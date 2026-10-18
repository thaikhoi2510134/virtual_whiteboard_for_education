# Virtual Teaching Board

Bảng giảng dạy điều khiển bằng cử chỉ tay qua webcam. Ứng dụng hỗ trợ vẽ tự do, nhận diện hình phác thảo, tạo hình học, kéo/phóng to hình và xóa bằng bàn tay mở.

## Cài đặt và chạy

Yêu cầu Python 3.9 trở lên và webcam. Trên macOS, hãy cấp quyền Camera cho ứng dụng Terminal hoặc IDE đang chạy Python.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python virtual_board.py
```

Lần đầu chạy, mô hình nhận diện bàn tay sẽ được tải tự động nếu chưa có tệp `hand_landmarker.task`.

## Điều khiển

| Cử chỉ / phím | Tác vụ |
| --- | --- |
| Ngón trỏ | Vẽ tự do hoặc phác thảo hình tự nhận diện |
| Ngón trỏ + giữa | Di chuyển trên thanh công cụ / chọn màu |
| Chụm ngón cái + trỏ | Vẽ đường, hình hoặc trục tọa độ |
| Nắm tay | Kéo hình đã có |
| Ngón cái + trỏ tại tâm hình | Phóng to/thu nhỏ hình |
| Xòe bàn tay | Tẩy |
| `Z` / `Y` | Hoàn tác / làm lại |
| `C` | Kết nối lại camera |
| `Esc` | Thoát |

Mỗi cử chỉ cần ổn định khoảng 140 ms trước khi có tác dụng để giảm thao tác nhầm. Lịch sử hoàn tác/làm lại lưu tối đa 50 thao tác.

## Kiểm tra

```bash
python -m unittest -v
```
