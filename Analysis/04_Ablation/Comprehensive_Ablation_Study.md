# TỔNG HỢP VÀ PHÂN TÍCH CHUYÊN SÂU ABLATION STUDY

Tài liệu này tổng hợp toàn bộ các nghiên cứu loại trừ (Ablation Studies) được công bố trong các bài báo nguồn, rút ra bài học cho việc cải tiến RF-DETR Medium:

### 1. D-FINE (FDR & GO-LSD) Ablation (Peng et al., Trang 8-9)
- **Đóng góp từng phần:**
  - FDR alone: $+1.2\%$ AP (Cải thiện rõ nét nhất ở $AP_{75}$ và $AP_{90}$).
  - GO-LSD alone: $+0.8\%$ AP (Tăng tốc độ hội tụ của các tầng nông 1-5).
  - FDR + GO-LSD: $+1.8\%$ AP và cho phép tỉa bớt kênh decoder để giảm 17% GFLOPs.
- **Ablation số lượng bins ($n$):**
  - $n=8$: Coarse resolution, thiếu độ mịn sub-pixel (+1.5% AP).
  - $n=16$: Điểm cân bằng hoàn hảo (+1.8% AP, chi phí tính toán thấp).
  - $n=32$: Over-smoothing phân phối, không tăng thêm mAP (+1.6% AP).
- **Rút ra cho RF-DETR:** Bắt buộc cố định $n=16$ khi triển khai Distribution Head cho RF-DETR.

### 2. NWD Loss Ablation (Wang et al., Trang 7)
- **Loss vs Matching:**
  - Chỉ áp dụng NWD trong Loss: $+6.4\%$ AP (Tăng mạnh khả năng định vị).
  - Chỉ áp dụng NWD trong Label Assignment: $+3.6\%$ AP (Tăng Recall).
  - Áp dụng cả Loss và Label Assignment: $+9.6\%$ AP.
- **Rút ra cho RF-DETR:** Trong giai đoạn đầu, chỉ nên tích hợp NWD vào Loss function để bảo toàn tính ổn định của Hungarian Matching. Sau khi hội tụ tốt mới thử nghiệm tích hợp vào Cost matrix.

### 3. Wise-IoU Version Ablation (Tong et al., Trang 8)
- WIoU v1 (Khoảng cách chuẩn hóa): $+0.4\%$ AP.
- WIoU v2 (Tiêu điểm đơn điệu): $+0.6\%$ AP.
- WIoU v3 (Tiêu điểm phi đơn điệu động): $+1.1\%$ AP.
- **Rút ra cho RF-DETR:** Bỏ qua v1 và v2, chỉ sử dụng phiên bản tối ưu WIoU v3 với tham số $\alpha=1.9, \delta=3.0$.
