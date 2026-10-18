import math
import cv2
import numpy as np

# ─── Dữ liệu hình học 3D ───────────────────────────────────────────────────
# Hệ tọa độ: x sang phải, y lên trên (lật về phía dưới khi chiếu), z vào màn hình.

# Hình hộp chữ nhật: 8 đỉnh của hình lập phương đơn vị
BOX_VERTICES = np.array([
    [-0.5, -0.5, -0.5],  # 0 trái-trên-trước
    [ 0.5, -0.5, -0.5],  # 1 phải-trên-trước
    [ 0.5,  0.5, -0.5],  # 2 phải-dưới-trước
    [-0.5,  0.5, -0.5],  # 3 trái-dưới-trước
    [-0.5, -0.5,  0.5],  # 4 trái-trên-sau
    [ 0.5, -0.5,  0.5],  # 5 phải-trên-sau
    [ 0.5,  0.5,  0.5],  # 6 phải-dưới-sau
    [-0.5,  0.5,  0.5],  # 7 trái-dưới-sau
], dtype=np.float64)

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # mặt trước
    (4, 5), (5, 6), (6, 7), (7, 4),  # mặt sau
    (0, 4), (1, 5), (2, 6), (3, 7),  # cạnh nối
]

# Hình chóp tứ giác: 4 đáy + 1 đỉnh
PYR_VERTICES = np.array([
    [-0.5,  0.5, -0.5],  # 0 đáy trước-trái
    [ 0.5,  0.5, -0.5],  # 1 đáy trước-phải
    [ 0.5,  0.5,  0.5],  # 2 đáy sau-phải
    [-0.5,  0.5,  0.5],  # 3 đáy sau-trái
    [ 0.0, -0.5,  0.0],  # 4 đỉnh chóp
], dtype=np.float64)

PYR_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # đáy
    (0, 4), (1, 4), (2, 4), (3, 4),  # cạnh bên
]

def create_parametric_shape(tool, pt1, pt2, color):
    x1, y1 = pt1
    x2, y2 = pt2

    if tool == "LINE":
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(-dy, dx))
        length = int(math.hypot(dx, dy))
        if abs(angle) < 6 or abs(abs(angle) - 180) < 6:
            y2 = y1
        elif abs(abs(angle) - 90) < 6:
            x2 = x1
        elif abs(abs(angle) - 45) < 5 or abs(abs(angle) - 135) < 5:
            side = min(abs(dx), abs(dy))
            x2 = x1 + (side if dx >= 0 else -side)
            y2 = y1 + (side if dy >= 0 else -side)
        pts = np.array([[x1, y1], [x2, y2]], np.int32)
        return {"type": "LINE", "pts": pts, "color": color, "info": f"{length}px"}

    elif tool == "SQR":
        side = max(abs(x2 - x1), abs(y2 - y1))
        sx = x1 + side if x2 >= x1 else x1 - side
        sy = y1 + side if y2 >= y1 else y1 - side
        pts = np.array([[x1, y1], [sx, y1], [sx, sy], [x1, sy]], np.int32)
        return {"type": "SQR", "pts": pts, "color": color, "info": ""}

    elif tool == "TRI":
        pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], np.int32)
        return {"type": "TRI", "pts": pts, "color": color, "info": ""}

    elif tool == "PARA":
        offset = int(0.25 * (x2 - x1))
        pts = np.array([[x1 + offset, y1], [x2, y1], [x2 - offset, y2], [x1, y2]], np.int32)
        return {"type": "PARA", "pts": pts, "color": color, "info": ""}

    elif tool == "CIR":
        radius = max(int(math.hypot(x2 - x1, y2 - y1)), 10)
        # Lưu 4 điểm cực quanh tâm (x1, y1) để giữ nguyên tâm khi di chuyển và co giãn
        pts = np.array([
            [x1 - radius, y1],
            [x1 + radius, y1],
            [x1, y1 - radius],
            [x1, y1 + radius]
        ], np.int32)
        return {"type": "CIR", "pts": pts, "color": color, "info": f"R={radius}px"}

    elif tool == "AXIS":
        rx = max(abs(x2 - x1), 60)
        ry = max(abs(y2 - y1), 60)
        # Lưu 4 đầu mút trục Ox (trái, phải) và Oy (trên, dưới) đối xứng qua gốc O(x1, y1)
        pts = np.array([
            [x1 - rx, y1],
            [x1 + rx, y1],
            [x1, y1 - ry],
            [x1, y1 + ry]
        ], np.int32)
        return {"type": "AXIS", "pts": pts, "color": color, "info": "Oxy"}

    return None


def recognize_auto_shape(stroke, color):
    if len(stroke) < 15:
        return None
    pts = np.array(stroke, dtype=np.int32)
    peri = cv2.arcLength(pts, True)
    if peri < 100:
        return None
    approx = cv2.approxPolyDP(pts, 0.04 * peri, True)
    x, y, bw, bh = cv2.boundingRect(pts)

    if len(approx) == 3:
        tri_pts = np.array([[x + bw // 2, y], [x, y + bh], [x + bw, y + bh]], np.int32)
        return {"type": "TRI", "pts": tri_pts, "color": color, "info": ""}

    elif len(approx) == 4:
        v = approx.reshape(4, 2)
        vec1 = v[1] - v[0]
        vec2 = v[2] - v[1]
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        cos_angle = abs(np.dot(vec1, vec2) / (norm + 1e-6))
        if cos_angle < 0.25:
            side = max(bw, bh)
            sqr_pts = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], np.int32)
            return {"type": "SQR", "pts": sqr_pts, "color": color, "info": ""}
        else:
            offset = int(0.25 * bw)
            para_pts = np.array([[x + offset, y], [x + bw, y], [x + bw - offset, y + bh], [x, y + bh]], np.int32)
            return {"type": "PARA", "pts": para_pts, "color": color, "info": ""}

    elif len(approx) > 4:
        area = cv2.contourArea(pts)
        circularity = 4 * math.pi * area / (peri * peri + 1e-6)
        if circularity > 0.65:
            cx, cy = x + bw // 2, y + bh // 2
            radius = max((bw + bh) // 4, 15)
            cir_pts = np.array([
                [cx - radius, cy],
                [cx + radius, cy],
                [cx, cy - radius],
                [cx, cy + radius]
            ], np.int32)
            return {"type": "CIR", "pts": cir_pts, "color": color, "info": f"R={radius}px"}

    return None


def scale_shape(shape, initial_pts, scale_factor, w, h):
    # Hình 3D: phóng to/thu nhỏ trường size thay vì pts
    if shape.get("type") in ("BOX", "PYR"):
        init_size = shape.get("_initial_size", shape["size"])
        shape["size"] = int(np.clip(init_size * scale_factor, 20, min(w, h) * 0.6))
        return

    center = np.mean(initial_pts, axis=0)

    scaled_pts = center + (initial_pts - center) * scale_factor
    max_span = np.max(np.linalg.norm(scaled_pts - center, axis=1))

    if 15 <= max_span <= min(w, h) * 0.85:
        shape["pts"] = scaled_pts.astype(np.int32)
        if shape["type"] == "LINE":
            p0, p1 = shape["pts"]
            length = int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            shape["info"] = f"{length}px"
        elif shape["type"] == "CIR":
            radius = int(np.mean(np.linalg.norm(shape["pts"] - center, axis=1)))
            shape["info"] = f"R={radius}px"


def render_shape(img, shape, highlight=False):
    # Hình 3D được vẽ bởi hàm riêng
    if shape.get("type") in ("BOX", "PYR"):
        render_3d_shape(img, shape, highlight)
        return

    pts = shape["pts"]
    color = (0, 255, 0) if highlight else shape["color"]
    thickness = 4 if highlight else 3
    center = np.mean(pts, axis=0).astype(int)

    if shape["type"] == "LINE":
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, thickness)
        mid = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2 - 10)
        if shape["info"]:
            cv2.putText(img, shape["info"], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    elif shape["type"] == "CIR":
        radius = max(int(np.mean(np.linalg.norm(pts - center, axis=1))), 5)
        cv2.circle(img, tuple(center), radius, color, thickness)
        if shape["info"]:
            cv2.putText(img, shape["info"], (center[0] - 30, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    elif shape["type"] == "AXIS":
        ox, oy = center
        left_pt, right_pt, top_pt, bottom_pt = pts[0], pts[1], pts[2], pts[3]

        # Vẽ trục Ox và Oy có mũi tên
        cv2.arrowedLine(img, tuple(left_pt), tuple(right_pt), color, thickness, tipLength=0.04)
        cv2.arrowedLine(img, tuple(bottom_pt), tuple(top_pt), color, thickness, tipLength=0.04)

        # Vẽ vạch chia đơn vị (mỗi 50px)
        tick_step = 50
        rx = abs(right_pt[0] - ox)
        ry = abs(bottom_pt[1] - oy)
        for dx in range(tick_step, rx - 15, tick_step):
            cv2.line(img, (ox + dx, oy - 5), (ox + dx, oy + 5), color, 2)
            cv2.line(img, (ox - dx, oy - 5), (ox - dx, oy + 5), color, 2)
        for dy in range(tick_step, ry - 15, tick_step):
            cv2.line(img, (ox - 5, oy + dy), (ox + 5, oy + dy), color, 2)
            cv2.line(img, (ox - 5, oy - dy), (ox + 5, oy - dy), color, 2)

        # Nhãn O, x, y
        cv2.putText(img, "x", (right_pt[0] - 15, right_pt[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "y", (top_pt[0] + 12, top_pt[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "O", (ox - 22, oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    cv2.circle(img, tuple(center), 5, color, -1)


# ─── Hàm 3D ────────────────────────────────────────────────────────────────

def project_3d(vertices, rotation_xyz, center_2d, scale, fov=800):
    """
    Chiếu 3D → 2D với phép phối cảnh (perspective projection).
    rotation_xyz : [rx, ry, rz] tính bằng độ.
    center_2d    : (cx, cy) tâm hình trên màn hình.
    scale        : kích thước (px).
    fov          : tiêu cự (càng lớn càng ít biến dạng).
    Trả về: (danh sách (x,y) đã chiếu, mảng độ sâu z).
    """
    rx, ry, rz = (math.radians(a) for a in rotation_xyz)

    cx_r, sx_r = math.cos(rx), math.sin(rx)
    cy_r, sy_r = math.cos(ry), math.sin(ry)
    cz_r, sz_r = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx_r, -sx_r], [0, sx_r, cx_r]])
    Ry = np.array([[cy_r, 0, sy_r], [0, 1, 0], [-sy_r, 0, cy_r]])
    Rz = np.array([[cz_r, -sz_r, 0], [sz_r, cz_r, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rotated = (R @ vertices.T).T * scale          # (N, 3)

    z = rotated[:, 2]
    denom = np.maximum(fov + z, 1.0)              # tránh chia 0
    f = fov / denom

    x2d = (rotated[:, 0] * f + center_2d[0]).astype(int)
    y2d = (-rotated[:, 1] * f + center_2d[1]).astype(int)  # lật y cho màn hình

    return list(zip(x2d.tolist(), y2d.tolist())), z


def create_3d_shape(tool, pt1, pt2, color):
    """Tạo dict hình 3D từ hai điểm pinch.  pts[[cx,cy]] để tương thích drag/erase."""
    x1, y1 = pt1
    x2, y2 = pt2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = max(int(math.hypot(x2 - x1, y2 - y1) * 0.6), 35)
    return {
        "type": tool,                            # "BOX" hoặc "PYR"
        "size": size,
        "rotation": [20.0, 30.0, 0.0],          # góc mặc định trông đẹp
        "color": color,
        "pts": np.array([[cx, cy]], np.int32),   # tâm — drag sẽ cập nhật pts này
        "info": "",
    }


def render_3d_shape(img, shape, highlight=False):
    """Vẽ hình 3D lên ảnh với perspective projection và depth shading."""
    stype = shape["type"]
    color = (0, 255, 0) if highlight else shape["color"]
    thickness = 4 if highlight else 2

    # Tâm lấy từ pts để đảm bảo drag luôn đồng bộ
    center = (int(shape["pts"][0, 0]), int(shape["pts"][0, 1]))

    vertices, edges = (BOX_VERTICES, BOX_EDGES) if stype == "BOX" else (PYR_VERTICES, PYR_EDGES)

    pts2d, z_depths = project_3d(vertices, shape["rotation"], center, shape["size"])

    z_min = float(np.min(z_depths))
    z_max = float(np.max(z_depths))
    z_range = max(z_max - z_min, 1.0)

    for v1, v2 in edges:
        avg_z = (float(z_depths[v1]) + float(z_depths[v2])) / 2
        t = 1.0 - (avg_z - z_min) / z_range      # 1 = gần nhất, 0 = xa nhất
        brightness = 0.35 + 0.65 * t
        edge_color = tuple(int(c * brightness) for c in color)
        p1, p2 = pts2d[v1], pts2d[v2]
        cv2.line(img, p1, p2, edge_color, thickness)

    cv2.circle(img, center, 5, color, -1)
    cv2.putText(img, stype, (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
