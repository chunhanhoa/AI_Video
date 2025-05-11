# Hướng dẫn sử dụng dự án Roop

## Cài đặt

1. Cài đặt các thư viện phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

2. Nếu muốn sử dụng GPU (khuyến nghị):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Sử dụng

### Bằng giao diện người dùng (UI)
```
python run.py
```

### Bằng dòng lệnh
```
python run.py --source [đường dẫn ảnh nguồn] --target [đường dẫn ảnh/video đích] --output [đường dẫn để lưu kết quả]
```

### Một số tùy chọn thêm:
- `--keep-fps`: Giữ nguyên FPS của video gốc
- `--keep-frames`: Giữ lại các frame tạm thời sau khi xử lý
- `--many-faces`: Xử lý tất cả các khuôn mặt trong video đích thay vì chỉ một
- `--gpu [ID]`: Sử dụng GPU cụ thể (mặc định: 0)

## Ví dụ
```
python run.py --source "avatar.jpg" --target "video.mp4" --output "output.mp4" --keep-fps
```

## Yêu cầu hệ thống
- Windows/Linux/Mac
- 8GB RAM (khuyến nghị)
- NVIDIA GPU với VRAM ít nhất 2GB (để tăng tốc độ xử lý)
