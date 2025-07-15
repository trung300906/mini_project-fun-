# Image Quality Analyzer - Hướng dẫn sử dụng

## 🚀 Cài đặt nhanh

### 1. Cài đặt thư viện cần thiết
```bash
pip install opencv-python numpy pillow
```

Hoặc sử dụng file requirements:
```bash
pip install -r requirements_simple.txt
```

### 2. Chạy chương trình

#### So sánh hai ảnh:
```bash
python main_simple.py image1.jpg image2.jpg
```

#### Phân tích một ảnh:
```bash
python main_simple.py --analyze-single image.jpg
```

#### Lưu kết quả ra file JSON:
```bash
python main_simple.py --save-json image1.jpg image2.jpg
```

## 📊 Các chỉ số đánh giá

### 1. **Sharpness (Độ sắc nét)**
- Sử dụng Laplacian variance
- Đánh giá độ rõ nét của ảnh
- Điểm cao = ảnh sắc nét

### 2. **Noise (Nhiễu)**
- Ước tính bằng high-pass filter
- Tính SNR (Signal-to-Noise Ratio)
- Điểm cao = ít nhiễu

### 3. **Contrast (Độ tương phản)**
- Độ lệch chuẩn của pixel
- Đánh giá sự khác biệt sáng tối
- Điểm cao = tương phản tốt

### 4. **Brightness (Độ sáng)**
- Giá trị sáng trung bình
- Đánh giá độ sáng phù hợp
- Điểm cao = độ sáng cân bằng

### 5. **Color (Màu sắc)**
- Cân bằng màu RGB
- Ước tính nhiệt độ màu
- Điểm cao = màu sắc cân bằng

### 6. **Exposure (Phơi sáng)**
- Phân tích histogram
- Tỷ lệ vùng quá sáng/tối
- Điểm cao = phơi sáng đúng

### 7. **Composition (Bố cục)**
- Quy tắc 1/3
- Cân bằng vùng sáng tối
- Điểm cao = bố cục hài hòa

## 🎯 Hệ thống chấm điểm

- **A++ (95-100)**: Outstanding (Xuất sắc)
- **A+ (90-94)**: Excellent (Tuyệt vời)
- **A (85-89)**: Very Good (Rất tốt)
- **B+ (80-84)**: Good (Tốt)
- **B (75-79)**: Above Average (Trên trung bình)
- **B- (70-74)**: Average (Trung bình)
- **C+ (65-69)**: Below Average (Dưới trung bình)
- **C (60-64)**: Fair (Khá)
- **D (50-59)**: Poor (Kém)
- **F (0-49)**: Unacceptable (Không chấp nhận được)

## 📋 Ví dụ output

```
🖼️  PROFESSIONAL IMAGE QUALITY ANALYZER v2.0.0
================================================================================

📊 ANALYSIS RESULTS
============================================================
📐 Image: example.jpg
📏 Size: 4000x3000 (12.0 MP)
💾 File Size: 2048.5 KB
📐 Aspect Ratio: 1.333

📷 Camera Information:
   • Camera Make: Canon
   • Camera Model: EOS R5
   • Aperture: f/2.8
   • Shutter Speed: 1/125s
   • Iso: 400

🏆 Quality Scores:
   • Sharpness: 85.2/100 🔵
   • Noise: 78.3/100 🟡
   • Contrast: 82.1/100 🔵
   • Brightness: 88.7/100 🔵
   • Color: 91.4/100 🟢
   • Exposure: 79.6/100 🟡
   • Composition: 74.2/100 🟡

⭐ Overall Score: 82.8/100
🎯 Grade: B+ (Good)
```

## 🔧 Tính năng

### ✅ Đã có:
- Phân tích chất lượng ảnh cơ bản
- So sánh hai ảnh
- Trích xuất thông tin EXIF
- Xuất kết quả JSON
- Giao diện dòng lệnh đơn giản

### 🚀 Có thể mở rộng:
- Thêm phân tích bokeh
- Phân tích màu sắc nâng cao
- Phát hiện khuôn mặt
- Batch processing
- GUI interface

## 🛠️ Yêu cầu hệ thống

- Python 3.7+
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Pillow >= 8.0.0

## 📁 Cấu trúc file

```
image_quality_analyzer/
├── main_simple.py           # File chính đơn giản
├── requirements_simple.txt  # Dependencies tối thiểu
├── USAGE.md                # Hướng dẫn này
└── (các file khác...)
```

## 🎯 Lưu ý

- Chương trình chỉ hỗ trợ các định dạng: JPG, PNG, TIFF, BMP, WebP
- Kết quả phụ thuộc vào chất lượng ảnh gốc
- Không cần GPU, chạy được trên CPU thường
- Phù hợp cho đánh giá nhanh chất lượng ảnh

## 🐛 Báo lỗi

Nếu gặp lỗi, hãy kiểm tra:
1. Đã cài đặt đủ thư viện chưa
2. Đường dẫn file ảnh có đúng không
3. Định dạng ảnh có được hỗ trợ không

---

**Image Quality Analyzer v2.0.0** - Đánh giá chất lượng ảnh chuyên nghiệp!
