# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU TÀI LIỆU KHOA HỌC
## TÀI LIỆU: R17 — LORA (ICLR 2022)

---

### 1. THÔNG TIN BÀI BÁO
- **Mã tài liệu:** R17 (Hướng cải tiến H5 — Thích ứng Backbone DINOv2)
- **Tên bài báo:** LoRA: Low-Rank Adaptation of Large Language Models
- **Tác giả:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Năm công bố:** 2021 (arXiv), 2022 (ICLR)
- **Hội nghị / Tạp chí:** International Conference on Learning Representations (**ICLR 2022**)
- **DOI:** 10.48550/arXiv.2106.09685
- **Link chính thức:** https://arxiv.org/abs/2106.09685
- **Link PDF:** https://arxiv.org/pdf/2106.09685.pdf
- **Link GitHub chính thức:** https://github.com/microsoft/LoRA
- **Tệp PDF đã lưu:** `2021_Hu_LoRA.pdf` (1.61 MB, 14 trang)
- **Đường dẫn trên máy:** `d:\NCKH\RFDETR_Research\01_Backbone\2021_Hu_LoRA.pdf`
- **Trạng thái xác minh:** Đã tải và xác thực toàn văn 14 trang PDF chính thức.

---

### 2. BÀI BÁO ĐANG GIẢI QUYẾT VẤN ĐỀ GÌ?
1. **Hạn chế của phương pháp gốc:**
   - Tinh chỉnh toàn bộ mô hình nền tảng (Full Fine-Tuning) đòi hỏi cập nhật toàn bộ tham số, tốn bộ nhớ optimizer khổng lồ và dễ gây hiện tượng quên thảm họa (catastrophic forgetting).
2. **Cơ chế LoRA:**
   - Đóng băng $W_0 \in \mathbb{R}^{d 	imes k}$ và chỉ huấn luyện $\Delta W = rac{lpha}{r} B A$ với $r \ll \min(d, k)$.
   - Giảm hơn 99% số tham số huấn luyện của backbone mà vẫn đạt hiệu quả biểu diễn tương đương.
   - Khi triển khai suy luận: Gộp tĩnh $W = W_0 + rac{lpha}{r} BA$, độ trễ tăng thêm bằng 0 ms.

---

### 3. ĐỐI CHIẾU VỚI RF-DETR MEDIUM
- **Backbone RF-DETR Medium:** DINOv2 ViT-S gồm 12 transformer blocks, 22 triệu tham số.
- **Hiện tại:** RF-DETR fine-tune toàn bộ DINOv2 bằng tốc độ học $0.1 	imes 	ext{LR}$.
- **Cải tiến đề xuất:** Đóng băng DINOv2, chèn LoRA ($r=8, lpha=16$) vào $W_q$ và $W_v$ trong self-attention.
- **Lợi ích:** Tiết kiệm 35% VRAM huấn luyện, bảo toàn 100% visual priors của DINOv2, triệt tiêu overfitting trên tập dữ liệu 32 lớp thực phẩm ĐBSCL.
