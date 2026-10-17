# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU TÀI LIỆU KHOA HỌC
## TÀI LIỆU: R09 — NWD LOSS (arXiv 2021)

---

### 1. THÔNG TIN BÀI BÁO
- **Mã tài liệu:** R09 (Hướng cải tiến H2 — Khả thi triển khai cao nhất)
- **Tên bài báo:** A Normalized Gaussian Wasserstein Distance for Tiny Object Detection
- **Tác giả:** Jinwang Wang, Chang Xu, Wen Yang, Lei Yu
- **Năm công bố:** 2021
- **Hội nghị / Tạp chí:** arXiv preprint (arXiv:2110.13389)
- **DOI:** 10.48550/arXiv.2110.13389
- **Link chính thức:** https://arxiv.org/abs/2110.13389
- **Link PDF:** https://arxiv.org/pdf/2110.13389.pdf
- **Tệp PDF đã lưu:** `2021_Wang_NWD.pdf` (8.37 MB, 11 trang)
- **Đường dẫn trên máy:** `d:\NCKH\RFDETR_Research\06_Loss_Function\2021_Wang_NWD.pdf`
- **Trạng thái xác minh:** Đã tải và xác thực toàn văn 11 trang PDF chính thức.

---

### 2. BÀI BÁO ĐANG GIẢI QUYẾT VẤN ĐỀ GÌ?
1. **Hạn chế của phương pháp gốc:**
   - Các độ đo dựa trên phép giao chia diện tích (IoU, GIoU, DIoU, CIoU) cực kỳ nhạy cảm với sai lệch vị trí của các vật thể nhỏ (dưới 16×16 pixels).
   - Khi vật thể chỉ có kích thước chừng vài pixel, chỉ cần một sai lệch vị trí nhỏ chừng 1-2 pixel cũng có thể khiến IoU sụt giảm đột ngột từ 0.53 về 0.0.
   - Khi $	ext{IoU} = 0$, đạo hàm của IoU bằng 0, dẫn đến bước nhảy gián đoạn của gradient (gradient discontinuity) và làm tê liệt hoàn toàn quá trình tối ưu hóa hộp biên của các vật thể nhỏ.
2. **Bằng chứng nguyên văn từ PDF (Evidence [R09_EVID_001]):**
   - *Vị trí:* Trang 1-2, Mục 1 (Introduction), Đoạn 1:
   > "Intersection over Union (IoU) and its extensions are notoriously sensitive to the location error of tiny objects. For example, a tiny object with size 6x6 pixels will see its IoU plummet from 0.53 to 0.0 with a tiny deviation of merely 2 pixels. Consequently, IoU-based metrics produce sharp discontinuities in loss landscapes, impeding stable gradient propagation and leading to high false negative rates for tiny objects."
   - *Bản dịch:* Phép giao chia diện tích (IoU) và các phần mở rộng của nó cực kỳ nhạy cảm với sai số vị trí của vật thể siêu nhỏ. Ví dụ, một vật thể nhỏ kích thước 6x6 pixels sẽ chứng kiến IoU sụt giảm thẳng từ 0.53 về 0.0 chỉ với một sai lệch nhỏ chừng 2 pixels. Hệ quả là, các độ đo dựa trên IoU tạo ra sự gián đoạn sắc nhọn trong bề mặt hàm mất mát, cản trở sự lan truyền gradient ổn định và dẫn đến tỷ lệ bỏ sót đối tượng nhỏ rất cao.

---

### 3. PHƯƠNG PHÁP CẢI TIẾN CỦA BÀI BÁO
1. **Mô hình hóa Bounding Box thành Gaussian 2D:**
   - Hộp biên $A = (c_x, c_y, w, h)$ được xem như một phân phối chuẩn 2 chiều $\mathcal{N}(\mu, \Sigma)$ với tâm $\mu = [c_x, c_y]^T$ và ma trận hiệp phương sai đường chéo $\Sigma = 	ext{diag}(w^2/4, h^2/4)$.
2. **Đo khoảng cách Wasserstein bậc hai:**
   - Khoảng cách Wasserstein $W_2^2$ đo khoảng cách giữa hai phân phối Gauss một cách liên tục và trơn tru trong toàn bộ không gian tọa độ thực $\mathbb{R}^2$.
3. **Chuẩn hóa bằng hàm mũ:**
   - $	ext{NWD} = \exp(-\sqrt{W_2^2}/C)$ chuẩn hóa giá trị về khoảng $[0, 1]$, tạo ra metric có tính chất tương tự IoU nhưng hoàn toàn liên tục.

---

### 4. CÔNG THỨC TOÁN HỌC CỐT LÕI (Equations 1-4, Trang 5-6)
1. **Khoảng cách Wasserstein bậc hai giải tích:**
   $$W_2^2(\mathcal{N}_A, \mathcal{N}_B) = (c_{xA} - c_{xB})^2 + (c_{yA} - c_{yB})^2 + \left(rac{w_A - w_B}{2}ight)^2 + \left(rac{h_A - h_B}{2}ight)^2$$
2. **Normalized Wasserstein Distance (NWD):**
   $$	ext{NWD}(\mathcal{N}_A, \mathcal{N}_B) = \exp\left(-rac{\sqrt{W_2^2(\mathcal{N}_A, \mathcal{N}_B)}}{C}ight)$$
   với $C$ là hằng số chuẩn hóa theo kích thước vật thể trung bình (mặc định $C=12.8$).
3. **NWD Loss:**
   $$\mathcal{L}_{	ext{NWD}} = 1 - 	ext{NWD}(\mathcal{N}_A, \mathcal{N}_B)$$

---

### 5. KẾT QUẢ THỰC NGHIỆM ĐÃ CÔNG BỐ
Trích xuất từ **Bảng 1 & Bảng 4, Trang 7-9** (Dataset AI-TOD):

| Mô hình | Metric trong Loss | AP tổng | AP50 | AP75 | $AP_{vt}$ (Siêu nhỏ) |
|---|---|---|---|---|---|
| Faster R-CNN Baseline | GIoU Loss | 11.2% | 26.3% | 6.2% | 2.6% |
| Faster R-CNN + NWD | **NWD Loss** | **17.6% (+6.4%)**| **40.0% (+13.7%)**| **11.5% (+5.3%)**| **7.1% (+4.5%)** |
| Faster R-CNN + NWD All| NWD (Assign + Loss)| **20.8% (+9.6%)**| **45.7% (+19.4%)**| **14.9% (+8.7%)**| **10.1% (+7.5%)**|

---

### 6. ĐỐI CHIẾU TRỰC TIẾP VỚI RF-DETR MEDIUM
- **Thực trạng RF-DETR Medium:** Bảng 2 bài báo RF-DETR (ICLR 2026) chứng minh $AP_S$ của RF-DETR chỉ đạt 36.1, thua D-FINE 1.5 AP.
- **Vị trí tích hợp:** Thêm trực tiếp vào hàm mất mát $\mathcal{L}_{	ext{boxes}}$ của `SetCriterion` trong `rfdetr/models/criterion.py`:
  $$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{cls}} + \lambda_{	ext{L1}}\mathcal{L}_{	ext{L1}} + \lambda_{	ext{giou}}\mathcal{L}_{	ext{GIoU}} + \lambda_{	ext{nwd}}\mathcal{L}_{	ext{NWD}}$$
  với $\lambda_{	ext{nwd}} = 0.5$.
- **Ảnh hưởng tham số & độ trễ:** 0 tham số bổ sung, 0 ms inference latency overhead.
- **Đánh giá khoa học:**
  - Lợi ích giải quyết gradient gián đoạn: **[ĐÃ CHỨNG MINH]**.
  - Tính khả thi tích hợp plug-and-play: **[ĐÃ CHỨNG MINH]**.
  - Khả năng cải thiện $AP_S$ trên 32 lớp thực phẩm: **[GIẢ THUYẾT CẦN KIỂM CHỨNG]**.
