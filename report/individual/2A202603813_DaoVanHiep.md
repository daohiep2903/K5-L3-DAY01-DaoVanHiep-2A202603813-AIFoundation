# AI Foundation Lab – Day 01 Report

## 1. Thông tin cá nhân

- Họ và tên: Đào Văn Hiệp
- MSSV: 2A202603813
- Chương trình: AI Foundation Lab
- Buổi học: Day 01
- SEED: 3221

## 2. Nội dung đã hoàn thành

- Khám phá và tiền xử lý bộ dữ liệu SMS Spam Collection.
- Xây dựng mô hình Logistic Regression bằng NumPy.
- Xây dựng mạng MLP bằng PyTorch.
- Thực hành Sentence Embedding với MiniLM.
- So sánh các phương pháp BoW, MLP, Embedding và kNN.
- Tìm hiểu LLM Zero-shot, Few-shot và Prompt Injection.
- Thực hiện thử thách cá nhân số 6: phân loại SMS bằng JSON.

## 3. Kết quả thực nghiệm

### 3.1. Khám phá dữ liệu

- Số dòng dữ liệu ban đầu: 5.572.
- Số tin nhắn trùng lặp: 403.
- Tổng dữ liệu sau khi bỏ trùng: 5.169.
- Số tin Spam: 653.
- Số tin Ham: 4.516.
- Train: 3.619 mẫu.
- Validation: 775 mẫu.
- Test: 775 mẫu.
- SUB: 40 mẫu.

### 3.2. So sánh các mô hình

| Mô hình | F1 Test | F1 SUB | p95 (ms) |
|---|---:|---:|---:|
| BoW + Logistic Regression | 0.880 | 0.824 | 0.094 |
| BoW + MLP | 0.924 | 0.947 | 0.234 |
| MiniLM + kNN | 0.867 | 1.000 | 72.056 |
| MiniLM + Logistic Regression | 0.904 | 0.974 | 45.531 |
| MiniLM + MLP Head | 0.918 | 0.974 | 43.500 |

**Nhận xét:**

- BoW + MLP đạt F1 Test cao nhất: 0.924.
- MiniLM + kNN đạt F1 SUB cao nhất: 1.000, nhưng p95 là 72.056 ms, không đáp ứng SLA 50 ms.
- MiniLM + Logistic Regression đạt p95 45.531 ms, đáp ứng SLA.
- BoW + Logistic Regression có tốc độ xử lý rất nhanh, p95 chỉ 0.094 ms.
- Mô hình không có ReLU đạt F1 Test 0.918.

### 3.3. Đánh giá Pareto và SLA

Các phương pháp nằm trên đường Pareto theo kết quả notebook:

- BoW + Logistic Regression.
- BoW + MLP.
- MiniLM + MLP Head.
- MiniLM + kNN.

Các phương pháp đáp ứng SLA p95 dưới 50 ms:

- BoW + Logistic Regression.
- BoW + MLP.
- MiniLM + Logistic Regression.
- MiniLM + MLP Head.

Nhận xét: BoW + MLP có chất lượng tốt và độ trễ rất thấp. MiniLM + MLP Head cũng đạt SLA và có F1 SUB 0.974.

### 3.4. Tính toán bằng tay

| Đại lượng | Kết quả |
|---|---:|
| baseline_accuracy | 0.87 |
| sigmoid_0 | 0.50 |
| f1_example | 0.73 |
| cosine_example | 0.60 |
| llm_cost_per_day | 10.50 USD |

Chi phí LLM được tính với 1.000.000 SMS/ngày, mỗi tin có 100 token input và 1 token output, giá lần lượt là 0.10 USD và 0.50 USD cho mỗi 1 triệu token.

### 3.5. Thử thách cá nhân số 6 – JSON Classification

Đã xây dựng:

- Hàm `build_classification_prompt_json()` để yêu cầu LLM trả về JSON.
- Hàm `parse_json_label()` để trích xuất nhãn spam/ham và lý do.
- Thực hiện ước tính token và chi phí giữa cách trả lời một từ và JSON.

Kết quả ước tính:

| Chỉ số | Trả lời một từ | JSON |
|---|---:|---:|
| Input tokens | 62.1 | 71.5 |
| Output tokens | 2 | 28 |
| Chi phí/1.000 tin | 0.00721 USD | 0.02115 USD |

Bản JSON có chi phí ước tính cao gấp khoảng **2.93 lần** bản trả lời một từ.

Lưu ý: Các lần gọi API thử nghiệm gặp lỗi 429 do hết quota, vì vậy chưa có kết quả đo F1 thực tế để so sánh hai cách.

## 4. Bài học rút ra

- Việc loại bỏ dữ liệu trùng lặp trước khi chia train/test giúp hạn chế data leakage.
- Logistic Regression có thể đạt kết quả tốt với dữ liệu BoW mà không cần mạng nơ-ron phức tạp.
- MLP cải thiện F1 Test từ 0.880 lên 0.924 so với BoW + Logistic Regression.
- Embedding giúp biểu diễn ngữ nghĩa của câu, nhưng có thể làm tăng độ trễ xử lý.
- LLM có khả năng phân loại theo ngữ cảnh nhưng cần quan tâm đến chi phí, latency và prompt injection.
- Khi lựa chọn mô hình cần cân bằng chất lượng, tốc độ và chi phí triển khai.

## 5. Khó khăn và hướng cải thiện

**Khó khăn:**
- Làm quen với NumPy, PyTorch và Sentence Embedding.
- Hiểu cách đánh giá mô hình qua Precision, Recall, F1-score.
- API LLM gặp lỗi 429 do không còn quota.

**Hướng cải thiện:**
- Tiếp tục thực hành NumPy, Pandas và PyTorch.
- Tìm hiểu sâu hơn về Transformer và Sentence Embedding.
- Cải thiện khả năng xử lý dữ liệu và đánh giá mô hình.
- Tìm hiểu cách xây dựng hệ thống AI có khả năng chống Prompt Injection.

## 6. AI Usage Log

- Công cụ AI: ChatGPT.
- Mục đích: Hỗ trợ giải thích thuật toán, công thức, kiểm tra lỗi code và phân tích kết quả thực nghiệm.
- Nội dung đã kiểm chứng: Đối chiếu công thức với code, chạy các ô kiểm tra trong notebook và kiểm tra kết quả thực nghiệm.
- Người học chịu trách nhiệm kiểm tra, hiểu và giải thích lại nội dung đã sử dụng.
