# Xây dựng mô hình phát hiện ùn tắc giao thông từ video camera giám sát

Báo cáo Bài tập lớn học phần **Mạng nơ-ron và Học sâu** - Trường Công nghệ Thông tin Phenikaa (Phenikaa University).

---

## 📌 Tổng quan đề tài
Tình trạng ùn tắc giao thông là vấn đề nan giải tại các đô thị lớn. Dự án này tập trung nghiên cứu và phát triển một hệ thống tự động nhận diện, phân loại trạng thái giao thông và đếm số lượng phương tiện theo thời gian thực từ camera giám sát (CCTV). 

Hệ thống được phát triển dựa trên việc tái hiện nghiên cứu khoa học gốc kết hợp mở rộng thêm mô hình phát hiện đối tượng và giao diện tương tác người dùng:
1. **Mô hình cốt lõi (CNN):** Phân loại nhị phân trạng thái giao thông thành `Ùn tắc (Jammed)` hoặc `Bình thường (Not Jammed)`.
2. **Mô hình mở rộng (YOLO):** Phát hiện và đếm số lượng các loại phương tiện (Ô tô, Xe máy, Xe buýt) để đánh giá mật độ.
3. **Giao diện đồ họa (GUI):** Hiển thị video và kết quả phân tích trực quan cho người dùng.

## 👥 Thành viên thực hiện
* **Nguyễn Thị Ngọc** - MSV: 20010788
* **Trịnh Đình Đức Trung** - MSV: 23010108
* **Phạm Hùng Mạnh** - MSV: 22010259
* **Giảng viên hướng dẫn:** TS. Phạm Tiến Lâm

---

## 🏗️ Kiến trúc hệ thống
Hệ thống được thiết kế theo cấu trúc tầng logic bảo đảm luồng xử lý dữ liệu xuyên suốt:
* **Tầng dữ liệu (Data Layer):** Tiếp nhận ảnh CCTV để huấn luyện CNN và Video Stream thực tế cho YOLO.
* **Tầng huấn luyện (Training Layer):** Tiền xử lý dữ liệu (Resize 100x100, Grayscale, Chuẩn hóa) và huấn luyện mô hình CNN trên Google Colab, xuất file trọng số `.h5`.
* **Tầng triển khai (Deployment Layer):** Nạp model `.h5`, kiểm thử dữ liệu ngoại biên, kết hợp đồng thời mô-đun phát hiện đối tượng YOLO.
* **Tầng ứng dụng (Application Layer):** Giao diện GUI nhận diện, vẽ bounding box, đếm xe và kết luận trạng thái giao thông.

---

## 💻 Công nghệ & Thư viện sử dụng
* **Ngôn ngữ lập trình:** Python
* **Học sâu & Trí tuệ nhân tạo:** TensorFlow, Keras, YOLOv8 (Ultralytics)
* **Xử lý ảnh/video:** OpenCV
* **Môi trường phát triển:** Google Colab (Huấn luyện), VS Code (Triển khai hệ thống)

---

## 📊 Mô hình & Thực nghiệm

### 1. Mô hình phân loại CNN
* **Kiến trúc:** Gồm 2 lớp Tích chập (Conv2D), lớp Max Pooling, 1 lớp Ẩn (Dense) với hàm kích hoạt `ReLU`, và lớp đầu ra sử dụng hàm `Sigmoid`.
* **Hàm mất mát:** Binary Cross Entropy.
* **Bộ tối ưu hóa:** Adam Optimizer.
* **Kết quả huấn luyện:** * Độ chính xác (Accuracy) trên tập kiểm tra nội bộ đạt **100%**.
  * Độ chính xác khi kiểm thử với 15 ảnh thực tế bên ngoài đạt **86.7%** (có hiện tượng Overfitting nhẹ do tập dữ liệu huấn luyện cố định).

### 2. Mô hình mở rộng YOLOv8
* Nhận diện chính xác các dòng phương tiện: `Car`, `Motorbike`, `Bus`.
* Cung cấp số lượng xe theo thời gian thực (Định lượng) bổ trợ cho kết quả phân loại trạng thái của CNN (Định tính).

---

## 🚀 Hướng dẫn cài đặt và khởi chạy

### 1. Yêu cầu hệ thống
Cài đặt Python (phiên bản khuyến nghị `>= 3.8`) và các thư viện phụ thuộc bằng cách chạy lệnh sau:

```bash
pip install tensorflow keras opencv-python ultralytics
