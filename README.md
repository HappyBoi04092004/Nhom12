# Ứng Dụng Dự Đoán Điểm Toán (TestScore_Math)

Ứng dụng web để dự đoán **Điểm Toán (TestScore_Math)** dựa trên các thông tin đầu vào:

- TestScore_Reading (Điểm Đọc)
- TestScore_Science (Điểm Khoa Học)
- GPA (Điểm Trung Bình)
- StudyHours (Số Giờ Học)
- AttendanceRate (Tỷ Lệ Tham Gia)

Model được train với dữ liệu thực tế từ file `Exam_Score_Prediction.csv` (999,997 mẫu).

## 📁 Cấu trúc thư mục

- `data/`: Chứa dữ liệu thô và dữ liệu đã xử lý
- `notebooks/`: Các file Jupyter Notebook dùng để mô tả và phân tích dữ liệu.
- `src/`: Mã nguồn chạy thử demo
  - `src/templates`: code html và css
- `reports/`: báo cáo
- `.gitignore`: Loại bỏ file không cần thiết
- `README.md`: Giới thiệu dự án

## 📦 Công nghệ sử dụng

| Thành phần                      | Công nghệ                                           |
| ------------------------------- | --------------------------------------------------- |
| **Ngôn ngữ lập trình**          | Python 3.9+                                         |
| **Phân tích dữ liệu & mô hình** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn    |
| **Giao diện người dùng (UI)**   | Spark                                               |
| **Môi trường phát triển**       | Jupyter Notebook (EDA), PyCharm ,Visual Studio Code |
| **Quản lý thư viện**            | `requirements.txt`                                  |
| **Hệ điều hành**                | Windows                                             |

---

## 📋 Kế hoạch công việc & hướng dẫn cộng tác

### Công cụ làm việc

- **Code phân tích dữ liệu**: Jupyter Notebook (`/notebooks`)
- **Code giao diện dùng cho dự đoán**: Viết trong Pycharm (`/src`)

### Nhiệm vụ chính

| Nhiệm vụ              | Mô tả                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| Tiền xử lý            | Chuẩn hóa dữ liệu cho huấn luyện. Dữ liệu gốc để phân tích mô tả                                            |
| Phân tích mô tả       | Vẽ biểu đồ, phân tích mối liên hệ giữa các biến xuất ra các figure                                          |
| Dự đoán               | Giao diện nhập liệu trên website, dùng model để dự đoán                                                     |
| Giao diện             | Form nhập đầu vào (Flask) và kết nối mô hình                                                                |
| Dự đoán theo tiêu chí | Cho phép chọn `TestScore_Reading`, `TestScore_Science`, `GPA`,`StudyHours`, `AttendanceRate` để lọc dự đoán |

## Cài Đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Chạy Ứng Dụng

1. Chạy server Flask:

```bash
python app.py
```

2. Mở trình duyệt và truy cập:

```
http://localhost:5000
```

## Cấu Trúc File

- `app.py`: Backend Flask với API endpoints
- `templates/index.html`: Giao diện web frontend
- `train_model.py`: Script để train model với dữ liệu thực tế
- `Exam_Score_Prediction.csv`: File dữ liệu training (999,997 mẫu)
- `model_metrics.json`: File chứa R² và RMSE metrics (tự động tạo khi train)
- `model.pkl`: File model machine learning (tự động tạo khi train)
- `requirements.txt`: Danh sách các thư viện Python cần thiết

## Train Model

Để train lại model với dữ liệu:

```bash
python train_model.py
```

Script này sẽ:

- Đọc dữ liệu từ `Exam_Score_Prediction.csv`
- Train model Linear Regression
- Tính R² và RMSE trên test set
- Lưu model vào `model.pkl`
- Lưu metrics vào `model_metrics.json`

## Metrics Hiện Tại

Model hiện tại được train với:

- **R² Score**: 0.694715 (69.47% variance được giải thích)
- **RMSE**: 5.488027 (Root Mean Squared Error)

## Cập Nhật Model

Để sử dụng model của riêng bạn:

1. Thay thế file `Exam_Score_Prediction.csv` với dữ liệu của bạn
2. Chạy `python train_model.py` để train lại model
3. Model và metrics sẽ được cập nhật tự động
