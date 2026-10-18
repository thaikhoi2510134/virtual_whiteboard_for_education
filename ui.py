import cv2
from config import TOOLS, COLORS


def draw_ui(frame, current_tool, current_color):
    """Vẽ thanh công cụ co giãn theo khung hình và trả về vùng bấm của từng nút."""
    h, w, _ = frame.shape
    margin, gap, bar_height = 8, 4, min(50, max(34, h // 14))
    tool_width = max(1, (w - 2 * margin - gap * (len(TOOLS) - 1)) // len(TOOLS))
    tool_regions = []
    for i, name in enumerate(TOOLS):
        x1 = margin + i * (tool_width + gap)
        x2 = min(w - margin, x1 + tool_width)
        y1, y2 = margin, margin + bar_height
        tool_regions.append((name, (x1, y1, x2, y2)))
        is_active = (name == current_tool)
        bg_color = (0, 180, 255) if is_active else (50, 50, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2 if is_active else 1)
        font_scale = min(0.62, max(0.32, tool_width / 155))
        (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(frame, name, (x1 + max(2, (tool_width - text_w) // 2), y1 + (bar_height + text_h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    color_regions = []
    color_size = min(34, max(20, (w - margin * 2 - gap * (len(COLORS) - 1)) // len(COLORS)))
    color_y1, color_y2 = margin + bar_height + 5, margin + bar_height + 5 + color_size
    color_start = max(margin, w - (color_size * len(COLORS) + gap * (len(COLORS) - 1) + margin))
    for i, col in enumerate(COLORS):
        x1 = color_start + i * (color_size + gap)
        x2 = x1 + color_size
        y1, y2 = color_y1, color_y2
        color_regions.append((col, (x1, y1, x2, y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
        if col == current_color:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
    return tool_regions, color_regions


def draw_hud(frame, state_text, state_color, tool_name, cam_idx, cam_name="Laptop Camera"):
    h, w, _ = frame.shape
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, h - 85), (480, h - 10), (25, 25, 25), -1)
    cv2.rectangle(overlay, (w - 370, h - 188), (w - 10, h - 10), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.rectangle(frame, (10, h - 85), (480, h - 10), state_color, 2)
    cv2.circle(frame, (30, h - 58), 9, state_color, -1)
    cv2.putText(frame, f"STATE: {state_text}", (50, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"TOOL: {tool_name}  |  Cam #{cam_idx}: {cam_name}", (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    cv2.rectangle(frame, (w - 370, h - 188), (w - 10, h - 10), (120, 120, 120), 1)
    guides = [
        "2 Fingers (Index+Mid) : Hover / Menu",
        "1 Finger (Index)      : Draw (FREE/AUTO)",
        "Pinch (Thumb+Index)   : Draw Shape/Axis/3D",
        "Thumb+Index on Center : Zoom Shape",
        "Fist (Closed Hand)    : Move Shape",
        "5 Fingers (Open Palm) : Eraser",
        "3 Fingers near 3D     : Rotate 3D Shape",
        "Z / Y : Undo / Redo",
    ]
    for i, g in enumerate(guides):
        cv2.putText(frame, g, (w - 358, h - 165 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 240, 210), 1)

