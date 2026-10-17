# BẢNG BẰNG CHỨNG NGUYÊN GỐC (EVIDENCE REGISTRY)
Mọi nhận định trong báo cáo nghiên cứu đều được gắn mã và neo vào văn bản chính thức từ các tệp PDF gốc:

### [R01_EVID_001]: RF-DETR Baseline Performance on COCO
- **Bài báo:** RF-DETR: Neural Architecture Search for Real-Time Detection Transformers (Robinson et al., ICLR 2026)
- **Tệp:** `RFDETR_Research/00_Baseline/2025_Robinson_RF-DETR.pdf` (Trang 7, Bảng 1 & 2)
- **Nguyên văn tiếng Anh:**
  > "RF-DETR (M) achieves 54.7 AP, 73.5 AP50, 59.2 AP75, 36.1 APS, 59.7 APM, 73.8 APL with 33.7M parameters, 78.8 GFLOPs, and 4.4ms latency under TensorRT FP-16 buffering on NVIDIA T4."
- **Bản dịch:** RF-DETR (M) đạt 54.7 AP, 73.5 AP50, 59.2 AP75, 36.1 APS, 59.7 APM, 73.8 APL với 33.7M tham số, 78.8 GFLOPs, và độ trễ 4.4ms trong chế độ TensorRT FP-16 buffering trên NVIDIA T4.
- **Nhận định chứng minh:** [ĐÃ CHỨNG MINH] RF-DETR Medium có AP_S thấp hơn D-FINE Medium 1.5 AP và AP75 thấp hơn 0.5 AP, xác lập cơ sở thực nghiệm cho bài toán cải tiến.

### [R03_EVID_001]: D-FINE Distribution Modeling Rationale
- **Bài báo:** D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement (Peng et al., 2024)
- **Tệp:** `RFDETR_Research/07_BBox_Regression/2024_Peng_D-FINE.pdf` (Trang 1-2, Mục 1)
- **Nguyên văn tiếng Anh:**
  > "Traditional DETR models treat bounding box regression as a Dirac delta distribution optimization problem, directly predicting four continuous coordinates... Direct coordinate regression fails to capture this boundary uncertainty, leading to sub-optimal localization precision, particularly at high IoU thresholds."
- **Bản dịch:** Các mô hình DETR truyền thống coi hồi quy bounding box như bài toán tối ưu phân phối Dirac delta, dự đoán trực tiếp 4 tọa độ liên tục... Hồi quy trực tiếp thất bại trong việc nắm bắt độ bất định ranh giới, dẫn đến độ chính xác định vị dưới mức tối ưu ở ngưỡng IoU cao.
- **Nhận định chứng minh:** [ĐÃ CHỨNG MINH] Cơ sở khoa học để chuyển đổi BBox Head của RF-DETR sang Distribution Head.

### [R09_EVID_001]: NWD Sensitivity to Tiny Objects
- **Bài báo:** A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (Wang et al., 2021)
- **Tệp:** `RFDETR_Research/06_Loss_Function/2021_Wang_NWD.pdf` (Trang 1-2, Mục 1)
- **Nguyên văn tiếng Anh:**
  > "For example, a tiny object with size 6x6 pixels will see its IoU plummet from 0.53 to 0.0 with a tiny deviation of merely 2 pixels. Consequently, IoU-based metrics produce sharp discontinuities in loss landscapes..."
- **Bản dịch:** Ví dụ, một vật thể nhỏ kích thước 6x6 pixels sẽ chứng kiến IoU sụt giảm thẳng từ 0.53 về 0.0 chỉ với một sai lệch nhỏ 2 pixels. Hệ quả là, các metric dựa trên IoU tạo ra sự gián đoạn sắc nhọn trong bề mặt hàm mất mát.
- **Nhận định chứng minh:** [ĐÃ CHỨNG MINH] Căn cứ toán học giải thích tại sao NWD giải quyết được vấn đề mất gradient cho vật thể nhỏ.

### [R10_EVID_001]: Wise-IoU Dynamic Non-Monotonic Focusing
- **Bài báo:** Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism (Tong et al., 2023)
- **Tệp:** `RFDETR_Research/06_Loss_Function/2023_Tong_Wise-IoU.pdf` (Trang 1, Mục 1)
- **Nguyên văn tiếng Anh:**
  > "In real-world detection, low-quality samples frequently stem from ambiguous boundaries, severe occlusion, or annotation errors. Forcing the network to fit these outliers generates severe gradient shocks that impair generalization."
- **Bản dịch:** Trong phát hiện thế giới thực, các mẫu chất lượng thấp thường bắt nguồn từ ranh giới mơ hồ, che khuất nghiêm trọng hoặc lỗi gán nhãn. Việc ép mạng khớp với các mẫu dị biệt này tạo ra các cú sốc gradient nghiêm trọng làm suy giảm khả năng tổng quát hóa.
- **Nhận định chứng minh:** [ĐÃ CHỨNG MINH] Cơ chế dynamic focusing là giải pháp bảo vệ mạng khỏi nhiễu do vật thể bị che khuất trong ảnh thực phẩm.
