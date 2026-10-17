# ĐỐI CHIẾU CHI TIẾT TỪNG PHƯƠNG PHÁP VỚI RF-DETR MEDIUM BASELINE

## 1. MÔ TẢ ĐỐI CHUẨN RF-DETR MEDIUM
- **Backbone:** DINOv2 ViT-S with Registers, 12 layers, 384 hidden dim, windowed self-attention.
- **Neck (Projector):** Multi-level feature aggregation + C2f blocks + LayerNorm (384 -> 256 dim). Thang đo: P3 (1/8), P4 (1/16), P5 (1/32).
- **Decoder:** Deformable DETR Decoder (6 layers, 300 queries, d_model=256, 13 groups trong Group DETR, K=4 sampling points).
- **Prediction Heads:** Linear Classifier (256 -> 32) và 3-layer MLP BBox Head (256 -> 256 -> 256 -> 4).
- **Hàm mất mát gốc:** $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{IA-BCE}} + 5.0 \cdot \mathcal{L}_{\text{L1}} + 2.0 \cdot \mathcal{L}_{\text{GIoU}}$.
- **Hiệu năng COCO (ICLR 2026):** 33.7M params, 78.8 GFLOPs, 4.4 ms latency (FP16 TRT buffering), AP = 54.7, AP50 = 73.5, AP75 = 59.2, AP_S = 36.1, AP_M = 59.7, AP_L = 73.8.

---

## 2. BẢNG ĐỐI CHIẾU KỸ THUẬT THEO TỪNG THÀNH PHẦN

| Thành phần | RF-DETR Medium Baseline | Phương pháp cải tiến đề xuất | Sự khác biệt cốt lõi | Khả năng giữ Pretrained Weights | Ảnh hưởng Latency | Phân loại |
|---|---|---|---|---|---|---|
| **BBox Regression Head** | 3-layer MLP dự đoán 4 giá trị liên tục $[c_x, c_y, w, h]$ | **D-FINE FDR:** Distribution Head dự đoán $4 \times 16$ bins xác suất | Chuyển từ điểm Dirac sang phân phối rời rạc có mô hình hóa độ bất định | Phải khởi tạo lại BBox Head | 0 ms | **Class B** |
| **BBox Refinement** | Offset cộng dồn $\Delta [c_x, c_y, w, h]$ | **FDR Refinement:** Tinh chỉnh phân phối qua hàm trọng số $W(n)$ | Dồn gradient vào lân cận đỉnh phân phối thay vì dự đoán lại từ đầu | Cần cập nhật logic decoder loop | 0 ms | **Class B** |
| **BBox Regression Loss** | L1 Loss + GIoU Loss | **NWD Loss:** Chuẩn hóa khoảng cách Wasserstein giữa hai Gaussian 2D | Gradient liên tục, không bị triệt tiêu khi IoU=0 trên vật thể nhỏ | 100% giữ nguyên | 0 ms | **Class A** |
| **Loss Focusing** | Trọng số mất mát tĩnh theo công thức GIoU/EIoU | **Wise-IoU v3:** Tiêu điểm động phi đơn điệu theo độ dị biệt $\beta$ | Giảm phạt các mẫu che khuất cực đoan/nhãn lệch, dồn sức cho mẫu trung bình | 100% giữ nguyên | 0 ms | **Class A** |
| **Convergence Rate** | Tính IoU trên hộp biên gốc | **Inner-IoU:** Tính IoU trên hộp phụ trợ tỉ lệ $\text{ratio} = 0.75$ | Khuếch đại biên độ gradient ở dải IoU cao ($AP_{75}$) | 100% giữ nguyên | 0 ms | **Class A** |
| **Multi-Scale Projector**| C2f blocks với phép cộng unweighted | **BiFPN / AFPN:** Trọng số chuẩn hóa học được và dung hợp tiệm tiến | Tự động cân bằng tầm quan trọng của từng tầng đặc trưng | Phải train lại Projector | +0.2 ms | **Class B** |
| **Backbone Adaptation** | Lower learning rate ($0.1 \times \text{LR}$) fine-tune toàn bộ | **LoRA:** Đóng băng $W_0$, học adapter $B \times A$ ($r=8$) | Tránh quên thảm họa (catastrophic forgetting), giảm 35% VRAM | 100% giữ nguyên | 0 ms (Folded) | **Class B** |
| **Decoder Cross-Attn** | Sparse Deformable Attention ($K=4$ sampling points) | **Enhanced Attention:** $K=6$ + Global Encoder Context Mixing | Mở rộng trường tiếp nhận cho vật thể dài và chống trôi khi che khuất | Cần chỉnh cross-attn | +0.4 ms | **Class B** |
