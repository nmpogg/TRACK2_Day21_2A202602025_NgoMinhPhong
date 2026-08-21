# BÁO CÁO THỰC HÀNH MLOPS LAB - DAY 21
**Course:** AIInAction - VinUni | **Chủ đề:** CI/CD cho AI Systems & Continuous Training  
**Họ và tên:** Ngô Minh Phong  
**GitHub Repository:** https://github.com/nmpogg/TRACK2_Day21_2A202602025_NgoMinhPhong  

---

## 1. Kết Quả Thực Nghiệm Cục Bộ & Lựa Chọn Siêu Tham Số (Bước 1)

Trong quá trình thực nghiệm cục bộ với MLflow tracking trên tập dữ liệu `train_phase1.csv` (2,998 mẫu) và đánh giá trên `eval.csv` (500 mẫu), các bộ siêu tham số đã được thử nghiệm:

| Run ID | Thuật toán | `n_estimators` | `max_depth` | `min_samples_split` | Accuracy | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | Random Forest | 100 | 5 | 2 | 0.5840 | 0.5780 |
| Run 2 | Random Forest | 200 | 10 | 2 | 0.6520 | 0.6498 |
| **Run 3 (Tối ưu)** | **Random Forest** | **350** | **20** | **2** | **0.6800** | **0.6791** |

> **Lý do lựa chọn bộ siêu tham số tối ưu:**
> - `n_estimators = 350`: Tăng số lượng cây quyết định giúp giảm phương sai (variance) và tăng tính ổn định của mô hình rừng ngẫu nhiên (Random Forest).
> - `max_depth = 20`: Cho phép cây học sâu hơn các mối quan hệ phi tuyến phức tạp giữa 12 thuộc tính hóa lý của rượu vang mà không bị underfitting.
> - `min_samples_split = 2`: Giữ độ chi tiết phân nhánh tối đa ở các nút lá.

---

## 2. Kết Quả Huấn Luyện Liên Tục & So Sánh Hiệu Năng (Bước 2 vs Bước 3)

| Chỉ số đánh giá | Bước 2 (2,998 mẫu) | Bước 3 (5,996 mẫu) | Mức cải thiện ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Accuracy** | `0.6800` (68.00%) | `0.7560` (75.60%) | **+7.60%** 🚀 |
| **F1-Score (Weighted)** | `0.6791` | `0.7551` | **+7.60%** 🚀 |

**Nhận xét:** Khi bổ sung thêm 2,998 mẫu mới từ `train_phase2.csv` vào pipeline thông qua DVC versioning và kích hoạt GitHub Actions tự động, độ chính xác của mô hình đã vượt qua ngưỡng kiểm định ($0.70$), đạt **75.60%** và tự động được triển khai thành công lên máy chủ Google Compute Engine.

---

## 3. Hoàn Thành 5 Thách Thức Nâng Cao (Bonus 1 – Bonus 5: +20 Điểm)

### 🌟 Bonus 1: Tracking MLflow Từ Xa Với DagsHub (+4đ)
- Đã xây dựng cơ chế tự động phát hiện biến môi trường `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` trong hàm `setup_mlflow()` tại `src/train.py`.
- Tự động trỏ `mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")` khi có cấu hình, và linh hoạt fallback về file cục bộ khi chạy offline.

### 🌟 Bonus 2: Thí Nghiệm & So Sánh Nhiều Thuật Toán (+4đ)
- Mở rộng `src/train.py` và `params.yaml` với trường `model_type`, hỗ trợ 3 thuật toán: `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`.
- Kết quả so sánh trên tập dữ liệu hoàn chỉnh (5,996 mẫu):
  - **Random Forest:** `Accuracy = 0.7560` | `F1 = 0.7551` (Tốt nhất)
  - **Gradient Boosting:** `Accuracy = 0.6900` | `F1 = 0.6892`
  - **Logistic Regression:** `Accuracy = 0.5220` | `F1 = 0.5042`

### 🌟 Bonus 3: Tự Động Sinh Báo Cáo Hiệu Suất & Confusion Matrix (+4đ)
- Pipeline tự động sinh file `outputs/report.txt` chứa chi tiết Precision, Recall, F1-score cho từng lớp chất lượng rượu (0: thấp, 1: trung bình, 2: cao) và ma trận nhầm lẫn (Confusion Matrix).
- GitHub Actions tự động đính kèm file báo cáo này vào Artifact `metrics-and-report` ở mỗi lần chạy.

### 🌟 Bonus 4: Cơ Chế Rollback & Chống Hồi Quy Hiệu Năng (+4đ)
- Khi deploy, metadata `metrics.json` của model hiện tại được lưu trên GCS tại `gs://<bucket>/models/latest/metrics.json`.
- Ở các lần huấn luyện tiếp theo, job `eval` trong `mlops.yml` tự động tải metrics của model trước đó và so sánh: nếu model mới bị giảm sút chất lượng so với model đang chạy trên production, pipeline sẽ **tự động từ chối deploy (Halt Deploy)** để bảo vệ hệ thống.

### 🌟 Bonus 5: Phát Hiện Mất Cân Bằng Dữ Liệu (Data Drift / Imbalance Warning) (+4đ)
- Tích hợp hàm `check_data_drift()` kiểm tra phân bố tần suất nhãn trước khi huấn luyện.
- Nếu bất kỳ lớp nào chiếm $< 10\%$ tổng số mẫu, hệ thống sẽ in cảnh báo rõ ràng `[CANH BAO DATA IMBALANCE]` vào log và lưu trường `label_distribution` vào `outputs/metrics.json` cùng MLflow tracking.

---

## 4. Khó Khăn Kỹ Thuật Gặp Phải & Giải Pháp Xử Lý

Trong quá trình xây dựng pipeline CI/CD tự động, một số sự cố đã phát sinh và được xử lý triệt để:

1. **Lỗi xác thực DVC (`401 Invalid Credentials`):**
   - *Nguyên nhân:* Cấu hình `.dvc/config` trỏ `credentialpath = ../sa-key.json` (thư mục gốc), trong khi workflow CI ban đầu chỉ ghi key vào `/tmp/sa-key.json`.
   - *Giải pháp:* Cập nhật job `train` trong `mlops.yml` để tạo file `sa-key.json` ở cả thư mục gốc và `/tmp/sa-key.json`.
2. **Lỗi DNS Lookup SSH Deploy (`dial tcp: lookup ***: no such host`):**
   - *Nguyên nhân:* Secret `VM_HOST` trên GitHub Actions được đặt nhầm tên máy ảo nội bộ thay vì Public External IP.
   - *Giải pháp:* Cập nhật secret `VM_HOST` thành địa chỉ IP công khai của máy ảo (`35.232.83.79`).
3. **Lỗi kiểm tra sức khỏe Deploy (`Process exited with status 1`):**
   - *Nguyên nhân:* Khi service `mlops-serve` restart, server cần 10–15 giây để tải mô hình mới từ GCS về và khởi động Uvicorn. Lệnh `sleep 5` quá ngắn khiến `curl /health` bị từ chối kết nối.
   - *Giải pháp:* Thêm vòng lặp thử lại (retry loop) tối đa 10 lần (mỗi lần 3 giây) trong script deploy để đảm bảo server sẵn sàng trước khi kết thúc job.

---

## 5. Danh Sách Minh Chứng (Screenshots)

Tất cả các hình ảnh minh chứng được lưu trong thư mục `submissions/screenshots/`:
1. **`mlflowUI.png`**: Giao diện MLflow UI ghi nhận đầy đủ các lượt chạy thực nghiệm cục bộ với các siêu tham số khác nhau (Bước 1).
2. **`github_action1.png`**: Pipeline CI/CD tự động ở Bước 2 hoàn thành thành công cả 4 jobs màu xanh (`Unit Test` $\rightarrow$ `Train` $\rightarrow$ `Eval` $\rightarrow$ `Deploy`).
3. **`github_action2.png`**: Pipeline Continuous Training ở Bước 3 tự động được kích hoạt khi có commit dữ liệu DVC mới (`train_phase2`).
4. **`accuracy_fail.png`**: Minh chứng Eval Gate hoạt động chính xác: tự động chặn và hủy deploy khi độ chính xác mô hình không đạt yêu cầu.
5. **`health_predict.png`**: Minh chứng gọi thực tế thành công tới VM qua REST API: endpoint `/health` (`{"status":"ok"}`) và `/predict` (`{"prediction":0,"label":"thap"}`).
6. **`gcs_bucket.png`**: Giao diện Google Cloud Storage Console hiển thị đầy đủ thư mục dữ liệu `dvc/` và model `models/latest/model.pkl` đã upload lên bucket `mlops-lab-nmp-202602025`.
7. **`dagshub.png`**: Giao diện MLflow Remote Server trên DagsHub lưu trữ và theo dõi các thí nghiệm trên đám mây từ xa (Bonus 1).
