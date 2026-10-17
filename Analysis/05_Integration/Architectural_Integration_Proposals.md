# PHƯƠNG ÁN TÍCH HỢP VÀ THAY ĐỔI MÃ NGUỒN RF-DETR MEDIUM

Tài liệu này xác định chính xác các file Python, class, function và tensor shape trong gói `rfdetr` cần chỉnh sửa khi triển khai thực tế:

### 1. TÍCH HỢP NWD LOSS (PHƯƠNG ÁN A)
- **Module can thiệp:** `rfdetr.models.criterion.SetCriterion` và `rfdetr.utilities.box_ops`.
- **File:** `rfdetr/models/criterion.py`.
- **Function:** `SetCriterion.loss_boxes()`.
- **Mã nguồn triển khai:**
  ```python
  def nwd_loss_xyxy(pred_boxes, target_boxes, C=12.8, eps=1e-7):
      # pred_boxes, target_boxes: (N, 4) in [x1, y1, x2, y2]
      cx_p, cy_p = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5, (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
      w_p, h_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps), (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)
      cx_t, cy_t = (target_boxes[:, 0] + target_boxes[:, 2]) * 0.5, (target_boxes[:, 1] + target_boxes[:, 3]) * 0.5
      w_t, h_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps), (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)
      
      w2 = (cx_p - cx_t).pow(2) + (cy_p - cy_t).pow(2) + ((w_p - w_t) * 0.5).pow(2) + ((h_p - h_t) * 0.5).pow(2)
      nwd = torch.exp(-torch.sqrt(w2.clamp(min=eps)) / C)
      return 1.0 - nwd
  ```
- **Rủi ro:** 0%. Không làm thay đổi forward pass của mô hình.

---

### 2. TÍCH HỢP D-FINE FDR & GO-LSD (PHƯƠNG ÁN C)
- **Module can thiệp:**
  - `rfdetr.models.lwdetr.LWDETR`: Thay thế `self.bbox_embed` bằng `DistributionHead(d_model=256, num_bins=16)`.
  - `rfdetr.models.criterion.SetCriterion`: Bổ sung `loss_dfl()` và `loss_go_lsd()`.
- **Tensor Shape Transformation:**
  - Forward output ban đầu: `pred_boxes` có shape `(B, 300, 4)`.
  - Với FDR: `pred_dist` có shape `(B, 300, 4, 16)`.
  - Chuyển đổi sang tọa độ hộp: `pred_boxes = (pred_dist.softmax(-1) * bin_values).sum(-1)`.
- **Rủi ro:** Cần re-initialize trọng số của BBox Head; không tái sử dụng được trọng số MLP ban đầu.
