# CHIẾN LƯỢC ĐỀ XUẤT CẢI TIẾN RF-DETR MEDIUM CHO BÀI TOÁN THỰC PHẨM ĐBSCL

Dựa trên toàn bộ kết quả phân tích tài liệu và thực nghiệm kiểm chứng, chúng tôi đề xuất lộ trình 3 giai đoạn:

### Giai đoạn 1: Triển khai tức thì Phương án A (RF-DETR-NWD)
- **Mục tiêu:** Nâng cao ngay chỉ số $AP_S$ (hiện tại RF-DETR Medium chỉ đạt 36.1 trên COCO, thấp hơn D-FINE 1.5 AP).
- **Hành động:** Viết thêm hàm `nwd_loss_xyxy` vào file huấn luyện `modal_rfdetr_m_unified_3in1_a100.py`, đặt hệ số $\lambda_{\text{nwd}} = 0.5$.
- **Thời gian:** 1 ngày lập trình và kiểm chứng.

### Giai đoạn 2: Tối ưu hóa toàn diện Loss (Phương án B — DualFocus)
- **Mục tiêu:** Xử lý triệt để các trường hợp nguyên liệu xếp chồng lấn và che khuất một phần trong đĩa ăn.
- **Hành động:** Tích hợp WIoU v3 dynamic focusing kết hợp với NWD Loss và Class-Balanced Weighting (CBW).

### Giai đoạn 3: Chuyển đổi kiến trúc sang FDR (Phương án C — RF-DETR-FDR)
- **Mục tiêu:** Tạo đóng góp tính mới khoa học cao nhất phục vụ công bố quốc tế / luận văn thạc sĩ.
- **Hành động:** Tái cấu trúc BBox Head sang Distribution Head 16 bins và bổ sung GO-LSD self-distillation.
