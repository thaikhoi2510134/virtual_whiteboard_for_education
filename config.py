# Cấu hình chung cho Virtual Teaching Board

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

CAM_WIDTH = 1280
CAM_HEIGHT = 720
WINDOW_NAME = "Virtual Teaching Board"

# Tên công cụ trên thanh menu. Vị trí được tính động theo độ phân giải camera.
TOOLS = [
    "FREE", "AUTO", "LINE", "TRI", "SQR", "PARA", "CIR", "AXIS", "BOX", "PYR", "CLEAR"
]

# Các công cụ vẽ hình phẳng bằng cử chỉ Pinch (chụm ngón cái + ngón trỏ)
PARAMETRIC_TOOLS = ["LINE", "TRI", "SQR", "PARA", "CIR", "AXIS"]

# Các công cụ vẽ hình 3D (cũng dùng Pinch để định kích thước, xoay bằng 3 ngón)
TOOLS_3D = ["BOX", "PYR"]

# Bảng màu BGR. Vị trí được tính động theo độ phân giải camera.
COLORS = [
    (0, 255, 255), (0, 255, 0), (0, 100, 255), (255, 255, 255)
]

# Chỉ nhận một cử chỉ sau khi nó ổn định trong khoảng thời gian này.
GESTURE_STABLE_MS = 140
GESTURE_SMOOTHING_ALPHA = 0.45

DEFAULT_TOOL = "FREE"
DEFAULT_COLOR = (0, 255, 255)

# Các đường nối khớp bàn tay MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]
