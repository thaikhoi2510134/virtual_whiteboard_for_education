import os
import math
import platform
import subprocess
import json
import ctypes
import ctypes.util
import urllib.request
import time
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import MODEL_PATH, MODEL_URL, HAND_CONNECTIONS, CAM_WIDTH, CAM_HEIGHT, GESTURE_STABLE_MS, GESTURE_SMOOTHING_ALPHA


def init_hand_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand(detector, mp_image):
    """Nhận diện ở chế độ video để MediaPipe theo dõi bàn tay giữa các khung hình."""
    timestamp_ms = time.monotonic_ns() // 1_000_000
    return detector.detect_for_video(mp_image, timestamp_ms)


class GestureStabilizer:
    """Chỉ chuyển trạng thái khi cử chỉ mới tồn tại đủ lâu, chống rung/jitter."""

    def __init__(self, delay_ms=GESTURE_STABLE_MS):
        self.delay_ms = delay_ms
        self.candidate = None
        self.candidate_since = 0
        self.active = "NONE"

    def update(self, raw_gesture, now_ms=None):
        now_ms = time.monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        if raw_gesture != self.candidate:
            self.candidate = raw_gesture
            self.candidate_since = now_ms
        if raw_gesture != self.active and now_ms - self.candidate_since >= self.delay_ms:
            self.active = raw_gesture
        return self.active


class PointSmoother:
    """Làm mượt vị trí con trỏ mà không làm chậm quá nhiều thao tác vẽ."""

    def __init__(self, alpha=GESTURE_SMOOTHING_ALPHA):
        self.alpha = alpha
        self.points = {}

    def update(self, name, point):
        previous = self.points.get(name)
        if previous is None:
            smoothed = point
        else:
            smoothed = (
                int(self.alpha * point[0] + (1 - self.alpha) * previous[0]),
                int(self.alpha * point[1] + (1 - self.alpha) * previous[1]),
            )
        self.points[name] = smoothed
        return smoothed

    def reset(self):
        self.points.clear()


def get_avfoundation_cameras():
    """Lấy danh sách [(index, name, model_id)] theo đúng thứ tự OpenCV AVFoundation trên macOS."""
    if platform.system() != "Darwin":
        return []
    try:
        objc_path = ctypes.util.find_library("objc")
        if not objc_path:
            return []
        objc = ctypes.cdll.LoadLibrary(objc_path)
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        msg_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        msg_id_cstr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(("objc_msgSend", objc))
        msg_id_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        msg_uint = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        msg_id_uint = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(("objc_msgSend", objc))
        msg_cstr = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))

        NSString = objc.objc_getClass(b"NSString")
        sel_stringWithUTF8 = objc.sel_registerName(b"stringWithUTF8String:")
        sel_UTF8String = objc.sel_registerName(b"UTF8String")

        media_type_vide = msg_id_cstr(NSString, sel_stringWithUTF8, b"vide")
        AVCaptureDevice = objc.objc_getClass(b"AVCaptureDevice")
        sel_devicesWithMediaType = objc.sel_registerName(b"devicesWithMediaType:")
        devices = msg_id_id(AVCaptureDevice, sel_devicesWithMediaType, media_type_vide)
        if not devices:
            return []

        sel_count = objc.sel_registerName(b"count")
        sel_objectAtIndex = objc.sel_registerName(b"objectAtIndex:")
        sel_localizedName = objc.sel_registerName(b"localizedName")
        sel_modelID = objc.sel_registerName(b"modelID")

        count = msg_uint(devices, sel_count)
        cameras = []
        for i in range(count):
            dev = msg_id_uint(devices, sel_objectAtIndex, i)
            name_ns = msg_id(dev, sel_localizedName)
            model_ns = msg_id(dev, sel_modelID)
            name = msg_cstr(name_ns, sel_UTF8String).decode("utf-8", errors="ignore") if name_ns else f"Camera {i}"
            model = msg_cstr(model_ns, sel_UTF8String).decode("utf-8", errors="ignore") if model_ns else ""
            cameras.append((i, name, model))
        return cameras
    except Exception:
        return []


def find_laptop_camera_index():
    """Tìm index của camera tích hợp trên laptop (FaceTime HD / Built-in), bỏ qua camera điện thoại."""
    phone_keywords = ("iphone", "ipad", "continuity", "desk view", "droidcam", "iriun", "epoccam")
    laptop_keywords = ("facetime", "built-in", "macbook", "internal", "integrated")

    cameras = get_avfoundation_cameras()
    if cameras:
        print("Danh sách camera phát hiện được:")
        for idx, name, model in cameras:
            print(f"  [{idx}] {name} ({model})")

        # Ưu tiên 1: Camera có tên FaceTime / Built-in / MacBook và không phải điện thoại
        for idx, name, model in cameras:
            combined = f"{name} {model}".lower()
            if any(k in combined for k in laptop_keywords) and not any(p in combined for p in phone_keywords):
                print(f"-> Đã chọn camera laptop: [{idx}] {name}")
                return idx, name

        # Ưu tiên 2: Bất kỳ camera nào không chứa từ khóa điện thoại
        for idx, name, model in cameras:
            combined = f"{name} {model}".lower()
            if not any(p in combined for p in phone_keywords):
                print(f"-> Đã chọn camera: [{idx}] {name}")
                return idx, name

        return cameras[0][0], cameras[0][1]

    # Dự phòng bằng system_profiler trên macOS nếu không gọi được AVFoundation trực tiếp
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["system_profiler", "SPCameraDataType", "-json"], timeout=3)
            data = json.loads(out.decode("utf-8", errors="ignore")).get("SPCameraDataType", [])
            if len(data) > 1:
                # Nếu có camera điện thoại kết nối Continuity Camera, macOS thường đẩy FaceTime sang index 1
                names = [cam.get("_name", "") for cam in data]
                has_phone = any(any(p in n.lower() for p in phone_keywords) for n in names)
                if has_phone:
                    return 1, "FaceTime HD Camera"
        except Exception:
            pass

    return 0, "Laptop Camera"


def open_camera(cam_index):
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Không thể mở camera. Hãy kiểm tra quyền Camera cho Terminal/Python "
            "và thử đổi camera bằng phím C."
        )
    return cap


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]


def draw_hand_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c1, c2 in HAND_CONNECTIONS:
        cv2.line(frame, pts[c1], pts[c2], (180, 180, 180), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 200, 255), -1)


def get_hand_roll_angle(landmarks, w, h):
    """
    Góc xoay của bàn tay quanh trục cổ tay → điều khiển xoay Y của hình 3D.
    Đo góc của đường thẳng nối khớp ngón trỏ (5) và ngón út (17) so với trục ngang.
    """
    ix, iy = landmarks[5].x * w, landmarks[5].y * h    # index MCP
    px, py = landmarks[17].x * w, landmarks[17].y * h  # pinky MCP
    return math.degrees(math.atan2(py - iy, px - ix))


def get_hand_tilt_angle(landmarks, w, h):
    """
    Góc nghiêng lên/xuống của bàn tay → điều khiển xoay X của hình 3D.
    Đo góc của đường nối cổ tay (0) đến lòng bàn tay (9).
    """
    wx, wy = landmarks[0].x * w, landmarks[0].y * h    # wrist
    mx, my = landmarks[9].x * w, landmarks[9].y * h    # middle MCP
    return math.degrees(math.atan2(my - wy, mx - wx))
