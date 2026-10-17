# SO SÁNH TỔNG HỢP VÀ MA TRẬN TƯƠNG THÍCH KỸ THUẬT

### 1. BẢNG XẾP HẠNG TỔNG THỂ
1. **H2 (NWD Loss):** Điểm 8.55/10 — Khả thi cao nhất, chi phí bằng 0, giải quyết điểm nghẽn vật nhỏ đã chứng minh của RF-DETR Medium.
2. **H1 (D-FINE FDR):** Điểm 8.50/10 — Đột phá kiến trúc lớn nhất, cải thiện mạnh độ chính xác ranh giới hộp biên ($AP_{75}$).
3. **H4 (Wise-IoU v3):** Điểm 8.15/10 — Lọc nhiễu che khuất hiệu quả, dễ triển khai.
4. **H7 (Inner-IoU):** Điểm 7.80/10 — Gia tốc hội tụ ở dải IoU cao.
5. **H3 (BiFPN/AFPN):** Điểm 7.60/10 — Cải thiện đặc trưng đa tầng nhưng tăng độ trễ nhẹ.
6. **H5 (LoRA DINOv2):** Điểm 7.60/10 — Giảm VRAM, chống overfitting.
7. **H6 (Enhanced Cross-Attn):** Điểm 7.15/10 — Tiềm năng nhưng phức tạp triển khai CUDA kernel.

### 2. TỔ HỢP TƯƠNG THÍCH TỐI ƯU
- **Tổ hợp 1 (Plan A — An toàn & Tức thì):** RF-DETR Baseline + NWD Loss.
- **Tổ hợp 2 (Plan B — Cân bằng chất lượng & Tốc độ):** RF-DETR Baseline + NWD Loss + Wise-IoU v3 + CBW + LQH.
- **Tổ hợp 3 (Plan C — Đột phá khoa học cho Luận văn):** RF-DETR + D-FINE FDR + GO-LSD + NWD Loss + LoRA Backbone.
