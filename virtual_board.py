import time
from copy import deepcopy
import cv2
import numpy as np
import mediapipe as mp

from config import (
    WINDOW_NAME,
    TOOLS,
    COLORS,
    PARAMETRIC_TOOLS,
    TOOLS_3D,
    DEFAULT_TOOL,
    DEFAULT_COLOR,
)
from gestures import (
    init_hand_detector,
    find_laptop_camera_index,
    open_camera,
    detect_hand,
    get_distance,
    fingers_up,
    draw_hand_skeleton,
    GestureStabilizer,
    PointSmoother,
    get_hand_roll_angle,
    get_hand_tilt_angle,
)
from shapes import create_parametric_shape, create_3d_shape, recognize_auto_shape, scale_shape, render_shape
from ui import draw_ui, draw_hud



def main():
    detector = init_hand_detector()

    cam_index, cam_name = find_laptop_camera_index()
    try:
        cap = open_camera(cam_index)
    except RuntimeError as error:
        print(f"LỖI CAMERA: {error}")
        return
    time.sleep(0.5)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    freehand_canvas = None
    shapes = []
    current_stroke = []
    prev_x, prev_y = 0, 0
    start_point = None
    selected_shape_idx = None
    drag_offset = (0, 0)

    scaling_shape_idx = None
    initial_pinch_dist = 0.0
    smooth_pinch_dist = 0.0
    initial_shape_pts = None

    # Trạng thái xoay hình 3D
    rotating_shape_idx = None
    prev_roll_angle = 0.0
    prev_tilt_angle = 0.0
    rotate_action_active = False

    current_tool = DEFAULT_TOOL
    current_color = DEFAULT_COLOR
    gesture_stabilizer = GestureStabilizer()
    point_smoother = PointSmoother()
    undo_history = []
    redo_history = []
    max_history = 50
    erase_action_active = False
    drag_action_active = False
    scale_action_active = False
    menu_action = None

    def current_snapshot():
        return freehand_canvas.copy(), deepcopy(shapes)

    def record_undo_state():
        """Lưu trạng thái trước một thao tác hoàn chỉnh; thao tác mới sẽ xóa nhánh redo."""
        if freehand_canvas is None:
            return
        undo_history.append(current_snapshot())
        if len(undo_history) > max_history:
            undo_history.pop(0)
        redo_history.clear()

    def restore_undo():
        nonlocal freehand_canvas, shapes
        if not undo_history:
            return
        redo_history.append(current_snapshot())
        freehand_canvas, shapes = undo_history.pop()

    def restore_redo():
        nonlocal freehand_canvas, shapes
        if not redo_history:
            return
        undo_history.append(current_snapshot())
        freehand_canvas, shapes = redo_history.pop()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if freehand_canvas is None or freehand_canvas.shape != frame.shape:
            freehand_canvas = np.zeros_like(frame)

        temp_canvas = np.zeros_like(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detect_hand(detector, mp_image)

        tool_regions, color_regions = draw_ui(frame, current_tool, current_color)

        gesture_state = "NO HAND DETECTED"
        state_color = (130, 130, 130)

        if results.hand_landmarks:
            hand_lms = results.hand_landmarks[0]
            draw_hand_skeleton(frame, hand_lms, w, h)

            ix, iy = point_smoother.update("index", (int(hand_lms[8].x * w), int(hand_lms[8].y * h)))
            tx, ty = point_smoother.update("thumb", (int(hand_lms[4].x * w), int(hand_lms[4].y * h)))
            mx, my = (ix + tx) // 2, (iy + ty) // 2
            cx, cy = point_smoother.update("palm", (int(hand_lms[9].x * w), int(hand_lms[9].y * h)))

            up = fingers_up(hand_lms)
            pinch_dist = get_distance((ix, iy), (tx, ty))
            is_fist = (not any(up)) and (pinch_dist > 45)
            is_thumb_index_pose = (not up[1] and not up[2] and not up[3]) and not is_fist

            # Kiểm tra cử chỉ 2 ngón (cái + trỏ) đè tâm vào hình để phóng to / thu nhỏ
            is_scaling = False
            if is_thumb_index_pose and start_point is None and len(current_stroke) < 10:
                if scaling_shape_idx is not None and scaling_shape_idx < len(shapes):
                    shape_center = np.mean(shapes[scaling_shape_idx]["pts"], axis=0)
                    if get_distance(shape_center, (mx, my)) < 160:
                        is_scaling = True
                    else:
                        scaling_shape_idx = None
                if scaling_shape_idx is None:
                    for idx, s in enumerate(shapes):
                        shape_center = np.mean(s["pts"], axis=0)
                        if get_distance(shape_center, (mx, my)) < 65:
                            scaling_shape_idx = idx
                            initial_pinch_dist = max(pinch_dist, 25.0)
                            smooth_pinch_dist = initial_pinch_dist
                            initial_shape_pts = s["pts"].astype(np.float64).copy()
                            # Hình 3D cần lưu kích thước ban đầu riêng
                            if s.get("type") in TOOLS_3D:
                                s["_initial_size"] = s["size"]
                            prev_x, prev_y = 0, 0
                            current_stroke = []
                            is_scaling = True
                            break
            else:
                scaling_shape_idx = None

            if up[0] and up[1] and not up[2] and not up[3]:
                raw_gesture = "HOVER"
            elif all(up):
                raw_gesture = "ERASE"
            elif is_scaling and scaling_shape_idx is not None:
                raw_gesture = "SCALE"
            elif is_fist:
                raw_gesture = "DRAG"
            elif current_tool in ["FREE", "AUTO"] and up[0] and not any(up[1:]):
                raw_gesture = "DRAW"
            elif (current_tool in PARAMETRIC_TOOLS or current_tool in TOOLS_3D) and pinch_dist < 42:
                raw_gesture = "SHAPE"
            elif up[0] and up[1] and up[2] and not up[3]:
                raw_gesture = "ROTATE"
            else:
                raw_gesture = "IDLE"
            stable_gesture = gesture_stabilizer.update(raw_gesture)

            if raw_gesture == stable_gesture == "HOVER":
                gesture_state = "HOVER / SELECT (2 Fingers)"
                state_color = (0, 255, 255)
                prev_x, prev_y = 0, 0
                selected_shape_idx = None
                scaling_shape_idx = None
                rotating_shape_idx = None

                if start_point is not None and (current_tool in PARAMETRIC_TOOLS or current_tool in TOOLS_3D):
                    if current_tool in TOOLS_3D:
                        new_shape = create_3d_shape(current_tool, start_point, (ix, iy), current_color)
                    else:
                        new_shape = create_parametric_shape(current_tool, start_point, (ix, iy), current_color)
                    if new_shape is not None:
                        record_undo_state()
                        shapes.append(new_shape)
                    start_point = None

                if len(current_stroke) > 0 and current_tool == "AUTO":
                    auto_shape = recognize_auto_shape(current_stroke, current_color)
                    if auto_shape is not None:
                        shapes.append(auto_shape)
                    else:
                        pts = np.array(current_stroke, np.int32)
                        cv2.polylines(freehand_canvas, [pts], isClosed=False, color=current_color, thickness=4)
                    current_stroke = []

                cv2.circle(frame, (ix, iy), 10, current_color, 2)
                hovered_action = None
                for name, (x1, y1, x2, y2) in tool_regions:
                    if x1 < ix < x2 and y1 < iy < y2:
                        hovered_action = ("tool", name)
                        if name == "CLEAR":
                            if menu_action != hovered_action and (shapes or np.any(freehand_canvas)):
                                record_undo_state()
                                freehand_canvas = np.zeros_like(frame)
                                shapes.clear()
                        else:
                            current_tool = name
                for col, (x1, y1, x2, y2) in color_regions:
                    if x1 < ix < x2 and y1 < iy < y2:
                        hovered_action = ("color", col)
                        current_color = col
                menu_action = hovered_action

            elif raw_gesture == stable_gesture == "ERASE":
                gesture_state = "ERASING (5 Fingers)"
                state_color = (0, 80, 255)
                if not erase_action_active:
                    record_undo_state()
                    erase_action_active = True
                cv2.circle(frame, (cx, cy), 50, (255, 255, 255), 2)
                cv2.circle(freehand_canvas, (cx, cy), 50, (0, 0, 0), -1)
                shapes = [s for s in shapes if get_distance(np.mean(s["pts"], axis=0), (cx, cy)) > 55]
                prev_x, prev_y = 0, 0
                start_point = None
                current_stroke = []
                selected_shape_idx = None
                scaling_shape_idx = None

            elif raw_gesture == stable_gesture == "SCALE" and scaling_shape_idx is not None:
                smooth_pinch_dist = 0.45 * pinch_dist + 0.55 * smooth_pinch_dist
                scale_factor = np.clip(smooth_pinch_dist / initial_pinch_dist, 0.25, 4.0)
                gesture_state = f"ZOOMING SHAPE ({int(scale_factor * 100)}%)"
                state_color = (255, 0, 255)
                selected_shape_idx = scaling_shape_idx

                if not scale_action_active:
                    record_undo_state()
                    scale_action_active = True
                scale_shape(shapes[scaling_shape_idx], initial_shape_pts, scale_factor, w, h)

                cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 255), 2)
                cv2.circle(frame, (tx, ty), 7, (255, 0, 255), -1)
                cv2.circle(frame, (ix, iy), 7, (255, 0, 255), -1)
                cv2.circle(frame, (mx, my), 6, (0, 255, 255), -1)

            elif raw_gesture == stable_gesture == "DRAG":
                gesture_state = "DRAGGING SHAPE (Fist)"
                state_color = (0, 165, 255)
                cv2.circle(frame, (cx, cy), 14, (0, 165, 255), -1)
                if selected_shape_idx is None:
                    for idx, s in enumerate(shapes):
                        center = np.mean(s["pts"], axis=0)
                        if get_distance(center, (cx, cy)) < 75:
                            selected_shape_idx = idx
                            drag_offset = (cx, cy)
                            break
                else:
                    dx = cx - drag_offset[0]
                    dy = cy - drag_offset[1]
                    if dx or dy:
                        if not drag_action_active:
                            record_undo_state()
                            drag_action_active = True
                        pts = shapes[selected_shape_idx]["pts"] + np.array([dx, dy], dtype=np.int32)
                        # Không cho phép hình bị kéo ra ngoài vùng nhìn thấy.
                        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                        shapes[selected_shape_idx]["pts"] = pts
                    drag_offset = (cx, cy)

            elif raw_gesture == stable_gesture == "DRAW":
                gesture_state = "DRAWING FREEHAND" if current_tool == "FREE" else "AUTO-SHAPE SKETCHING"
                state_color = (0, 255, 0)
                selected_shape_idx = None

                # Phát hiện "nhấc bút": nếu ngón tay nhảy quá xa so với điểm vẽ trước
                # (ví dụ > 50px), coi đây là bắt đầu nét mới, không vẽ đường nối.
                PEN_LIFT_THRESHOLD = 50
                jump = get_distance((ix, iy), (prev_x, prev_y)) if (prev_x != 0 or prev_y != 0) else 0
                is_new_stroke = (prev_x == 0 and prev_y == 0) or (jump > PEN_LIFT_THRESHOLD)

                if is_new_stroke:
                    if prev_x == 0 and prev_y == 0:
                        record_undo_state()
                    # Nếu đang vẽ AUTO và có nét cũ bị gián đoạn, lưu nét đó lại
                    if current_tool == "AUTO" and len(current_stroke) > 5:
                        auto_shape = recognize_auto_shape(current_stroke, current_color)
                        if auto_shape is not None:
                            shapes.append(auto_shape)
                        else:
                            pts = np.array(current_stroke, np.int32)
                            cv2.polylines(freehand_canvas, [pts], isClosed=False, color=current_color, thickness=4)
                        current_stroke = []
                    prev_x, prev_y = ix, iy

                smooth_x = int(0.45 * ix + 0.55 * prev_x)
                smooth_y = int(0.45 * iy + 0.55 * prev_y)
                if current_tool == "FREE":
                    if not is_new_stroke:
                        cv2.line(freehand_canvas, (prev_x, prev_y), (smooth_x, smooth_y), current_color, 4)
                else:
                    current_stroke.append((smooth_x, smooth_y))
                    if len(current_stroke) > 1:
                        pts = np.array(current_stroke, np.int32)
                        cv2.polylines(temp_canvas, [pts], isClosed=False, color=current_color, thickness=3)
                prev_x, prev_y = smooth_x, smooth_y

            elif raw_gesture == stable_gesture == "SHAPE":
                gesture_state = f"DRAWING {current_tool} (Pinch)"
                state_color = (0, 255, 0)
                selected_shape_idx = None
                rotating_shape_idx = None
                if start_point is None:
                    start_point = (ix, iy)
                # Hình 3D hay hình phẳng?
                if current_tool in TOOLS_3D:
                    preview = create_3d_shape(current_tool, start_point, (ix, iy), current_color)
                    if preview is not None:
                        render_shape(temp_canvas, preview)
                else:
                    preview = create_parametric_shape(current_tool, start_point, (ix, iy), current_color)
                    if preview is not None:
                        ddx = abs(ix - start_point[0])
                        ddy = abs(iy - start_point[1])
                        is_snapped = (current_tool == "LINE") and (ddx == 0 or ddy == 0 or abs(ddx - ddy) < 5)
                        render_shape(temp_canvas, preview, highlight=is_snapped)
                cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

            elif raw_gesture == stable_gesture == "ROTATE":
                # ── Xoay hình 3D bằng 3 ngón (trỏ + giữa + áp út) ──────────────
                gesture_state = "ROTATING 3D (3 Fingers)"
                state_color = (255, 140, 0)
                prev_x, prev_y = 0, 0
                start_point = None

                # Tìm hình 3D gần nhất
                if rotating_shape_idx is None or rotating_shape_idx >= len(shapes):
                    rotating_shape_idx = None
                    for idx, s in enumerate(shapes):
                        if s.get("type") in TOOLS_3D:
                            shape_center = np.mean(s["pts"], axis=0)
                            if get_distance(shape_center, (cx, cy)) < 130:
                                rotating_shape_idx = idx
                                prev_roll_angle = get_hand_roll_angle(hand_lms, w, h)
                                prev_tilt_angle = get_hand_tilt_angle(hand_lms, w, h)
                                break

                if rotating_shape_idx is not None:
                    roll = get_hand_roll_angle(hand_lms, w, h)
                    tilt = get_hand_tilt_angle(hand_lms, w, h)

                    def _delta(a, b):
                        d = a - b
                        return d - 360 if d > 180 else (d + 360 if d < -180 else d)

                    dr = _delta(roll, prev_roll_angle)
                    dt = _delta(tilt, prev_tilt_angle)

                    if abs(dr) + abs(dt) > 0.4:
                        if not rotate_action_active:
                            record_undo_state()
                            rotate_action_active = True
                        rot = shapes[rotating_shape_idx]["rotation"]
                        rot[1] = (rot[1] + dr * 1.5) % 360   # xoay Y (roll)
                        rot[0] = (rot[0] + dt * 1.5) % 360   # xoay X (tilt)

                    prev_roll_angle = roll
                    prev_tilt_angle = tilt
                    selected_shape_idx = rotating_shape_idx

                    # Vẽ đường hiển thị trục xoay (knuckle line)
                    p5  = (int(hand_lms[5].x  * w), int(hand_lms[5].y  * h))
                    p17 = (int(hand_lms[17].x * w), int(hand_lms[17].y * h))
                    cv2.line(frame, p5, p17, (255, 140, 0), 3)
                    rot = shapes[rotating_shape_idx]["rotation"]
                    cv2.putText(frame, f"Ry={rot[1]:.0f} Rx={rot[0]:.0f}",
                                (cx - 50, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

            else:
                gesture_state = "IDLE / READY"
                state_color = (200, 200, 200)
                rotating_shape_idx = None
                if start_point is not None and (current_tool in PARAMETRIC_TOOLS or current_tool in TOOLS_3D):
                    if current_tool in TOOLS_3D:
                        new_shape = create_3d_shape(current_tool, start_point, (ix, iy), current_color)
                    else:
                        new_shape = create_parametric_shape(current_tool, start_point, (ix, iy), current_color)
                    if new_shape is not None:
                        record_undo_state()
                        shapes.append(new_shape)
                    start_point = None
                if len(current_stroke) > 0 and current_tool == "AUTO":
                    auto_shape = recognize_auto_shape(current_stroke, current_color)
                    if auto_shape is not None:
                        shapes.append(auto_shape)
                    else:
                        pts = np.array(current_stroke, np.int32)
                        cv2.polylines(freehand_canvas, [pts], isClosed=False, color=current_color, thickness=4)
                    current_stroke = []
                prev_x, prev_y = 0, 0
                selected_shape_idx = None
                scaling_shape_idx = None
        else:
            scaling_shape_idx = None
            rotating_shape_idx = None
            point_smoother.reset()
            gesture_stabilizer.update("NONE")

        is_stable_hover = bool(results.hand_landmarks and raw_gesture == stable_gesture == "HOVER")
        erase_action_active  = bool(results.hand_landmarks and raw_gesture == stable_gesture == "ERASE")
        drag_action_active   = bool(results.hand_landmarks and raw_gesture == stable_gesture == "DRAG")
        scale_action_active  = bool(results.hand_landmarks and raw_gesture == stable_gesture == "SCALE")
        rotate_action_active = bool(results.hand_landmarks and raw_gesture == stable_gesture == "ROTATE"
                                    and rotating_shape_idx is not None)
        if not is_stable_hover:

            menu_action = None

        for idx, s in enumerate(shapes):
            render_shape(temp_canvas, s, highlight=(idx == selected_shape_idx))

        combined_canvas = cv2.bitwise_or(freehand_canvas, temp_canvas)
        gray = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, combined_canvas)

        draw_hud(frame, gesture_state, state_color, current_tool, cam_index, cam_name)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('z') or key == ord('Z'):
            restore_undo()
        elif key == ord('y') or key == ord('Y'):
            restore_redo()
        elif key == ord('c') or key == ord('C'):
            cap.release()
            try:
                cam_index, cam_name = find_laptop_camera_index()
                cap = open_camera(cam_index)
                freehand_canvas = None
            except RuntimeError as error:
                print(f"LỖI CAMERA: {error}")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
