# RF-DETR Architecture Improvement & Scientific Literature Survey
### Real-Time Detection Transformers for 32 Food Ingredient Classes in the Mekong Delta

[![ICLR 2026](https://img.shields.io/badge/Baseline-RF--DETR%20(ICLR%202026)-blue)](https://arxiv.org/abs/2511.09554)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20rfdetr-EE4C2C)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Research-Literature%20Survey%20%26%20Design-success)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Tổng quan Nghiên cứu (Research Overview)

Kho lưu trữ này chứa toàn bộ tài liệu khảo sát khoa học chuyên sâu, dữ liệu bằng chứng định lượng, báo cáo phân tích đối chuẩn và các phương án cải tiến kiến trúc cho mô hình **RF-DETR Medium** (*Neural Architecture Search for Real-Time Detection Transformers*, ICLR 2026).

- **Bài toán nghiên cứu:** Nhận diện và định vị đối tượng (Object Detection) cho **32 lớp nguyên liệu thực phẩm** đặc thù tại khu vực Đồng bằng sông Cửu Long (ĐBSCL).
- **Mô hình nền tảng (Baseline):** **RF-DETR Medium** (DINOv2 ViT-S backbone, 12 layers, $d_{\text{model}} = 256$, 33.7M tham số, độ trễ 4.4 ms).
- **Thách thức cốt lõi đã chứng minh:**
  1. **Định vị vật thể nhỏ ($AP_S$):** RF-DETR Medium đạt $AP_S = 36.1$ trên COCO (thua D-FINE Medium 1.5 AP) do hiện tượng gián đoạn gradient khi $\text{IoU} = 0$.
  2. **Độ chính xác ranh giới khắt khe ($AP_{75}$):** Mạng hồi quy MLP 3 tầng dự đoán 4 điểm cố định (Dirac delta) không mô hình hóa được sự bất định ranh giới thức ăn lởm chởm, mờ nhòe.
  3. **Che khuất & Chồng lấn:** Thức ăn xếp đè lên nhau gây nhiễu gradient khi tối ưu hàm mất mát đơn điệu.

---

## 📂 Cấu trúc Thư mục (Repository Structure)

```
RFDETR_Research/
├── 00_Baseline/                     # Bài báo nền tảng RF-DETR, LW-DETR, DINOv2, RT-DETR
│   ├── 2025_Robinson_RF-DETR.pdf    # Bài báo gốc RF-DETR (ICLR 2026, 24 trang)
│   ├── 2024_Chen_LW-DETR.pdf        # Nền tảng kiến trúc transformer nhẹ
│   ├── 2024_Zhao_RT-DETR.pdf        # Nền tảng RT-DETR (CVPR 2024)
│   └── 2023_Oquab_DINOv2.pdf        # Visual Foundation Model (TMLR 2024)
├── 01_Backbone/                     # Các nghiên cứu tối ưu hóa & thích ứng Backbone
│   ├── 2021_Hu_LoRA.pdf             # Low-Rank Adaptation cho Vision Transformer
│   └── 2023_Beyer_FlexiViT.pdf      # Thích ứng kích thước Patch đa tỷ lệ
├── 02_Feature_Fusion/               # Nâng cấp Neck & Multi-Scale Feature Projector
│   ├── 2020_Tan_BiFPN_EfficientDet.pdf # Dung hợp đặc trưng hai chiều có trọng số
│   └── 2023_Wang_AFPN.pdf           # Dung hợp tiệm tiến xóa bỏ semantic gap
├── 03_Attention/                    # Cơ chế chú ý biến dạng trong Transformer
│   └── 2020_Zhu_Deformable_DETR.pdf # Multi-Scale Deformable Cross-Attention
├── 04_Decoder/                      # Tối ưu hóa bộ giải mã & Query Assignment
│   ├── 2022_Zhang_DINO.pdf          # Contrastive DeNoising & Mixed Queries
│   └── 2023_Chen_Group_DETR.pdf     # One-to-Many Group Hungarian Matching
├── 06_Loss_Function/                # Các hàm mất mát hình học & điều tiết động học
│   ├── 2019_Cui_Class_Balanced_Loss.pdf # Trọng số mẫu hiệu dụng (CBW)
│   ├── 2021_Wang_NWD.pdf            # Normalized Gaussian Wasserstein Distance (Tiny objects)
│   ├── 2023_Tong_Wise-IoU.pdf       # Tiêu điểm động phi đơn điệu WIoU v3
│   ├── 2023_Zhang_Inner-IoU.pdf     # Hộp biên phụ trợ gia tốc hội tụ
│   └── 2024_Zhang_Shape-IoU.pdf     # Độ đo nhạy với tỷ lệ & hình dạng
├── 07_BBox_Regression/              # Tái cấu trúc hồi quy ranh giới bằng phân phối xác suất
│   ├── 2020_Li_GFL_DFL.pdf          # Distribution Focal Loss (DFL)
│   └── 2024_Peng_D-FINE.pdf         # Fine-grained Distribution Refinement (FDR & GO-LSD)
├── Analysis/                        # Hệ thống báo cáo phân tích chuyên sâu
│   ├── 01_Paper_Analysis/           # Báo cáo chi tiết độc lập từng bài báo
│   ├── 02_Baseline_Comparison/      # Bảng đối chuẩn từng thành phần với RF-DETR
│   ├── 03_Evidence/                 # Trích dẫn nguyên văn tiếng Anh & bản dịch PDF
│   ├── 04_Ablation/                 # Phân tích các thực nghiệm loại trừ (Ablation)
│   ├── 05_Integration/              # Hướng dẫn can thiệp mã nguồn gói rfdetr
│   ├── 06_Comparison/               # Xếp hạng tổng thể & Ma trận tương thích
│   ├── 07_Recommendations/          # Đề xuất chiến lược và lộ trình thực nghiệm
│   ├── Paper_Analysis_Index.csv     # Chỉ mục 18 bài báo khoa học
│   ├── Baseline_Comparison.csv      # Bảng so sánh 11 trường thông tin
│   ├── Experimental_Results.csv     # Số liệu thực nghiệm gốc đã kiểm chứng
│   ├── Ablation_Analysis.csv        # Dữ liệu ablation studies
│   ├── Integration_Feasibility.csv  # Đánh giá phân loại tính khả thi
│   ├── Compatibility_Matrix.csv     # Ma trận tương thích 13 cặp phương pháp
│   └── Final_Research_Report.md     # Báo cáo tổng thể 13 phần chuẩn mực
├── References.csv                   # Danh mục tài liệu tham khảo chính
├── Evidence.csv                     # Bảng 15 chứng cứ khoa học cốt lõi
├── Research_Report.md               # Báo cáo nghiên cứu tổng hợp
└── HUONG_DAN_NGHIEN_CUU_VA_DOC_TAI_LIEU.md # Cẩm nang hướng dẫn nghiên cứu & bảo vệ luận văn
```

---

## 🎯 Ba Phương án Kiến trúc Đề xuất (Proposed Architectures)

### 1. Phương án A: RF-DETR-NWD (Ưu tiên triển khai tức thì — Rủi ro: 0%)
- **Cơ chế:** Bổ sung hàm mất mát $\mathcal{L}_{\text{NWD}}$ ([Wang et al., 2021](06_Loss_Function/2021_Wang_NWD.pdf)) với $\lambda_{\text{nwd}} = 0.5$ vào `SetCriterion.loss_boxes`.
- **Lợi ích:** Mô hình hóa hộp dạng Gaussian 2D, khoảng cách Wasserstein luôn có đạo hàm liên tục khi $\text{IoU} = 0$, cứu vãn các vật thể siêu nhỏ (ớt hiểm, tép, đậu phộng).
- **Chi phí:** 0 tham số bổ sung, 0 ms inference overhead, tái sử dụng 100% pretrained weights.

### 2. Phương án B: RF-DETR-DualFocus (Cân bằng thực tế)
- **Cơ chế:** Kết hợp **Wise-IoU v3** ([Tong et al., 2023](06_Loss_Function/2023_Tong_Wise-IoU.pdf)) tiêu điểm động phi đơn điệu + **NWD Loss** + **Localization Quality Head (LQH)** + **Class-Balanced Weighting (CBW)**.
- **Lợi ích:** Tự động triệt tiêu gradient độc hại từ các vùng thức ăn xếp chồng lấn bị gán nhãn lệch, bảo vệ quá trình hội tụ.

### 3. Phương án C: RF-DETR-FDR (Đột phá kiến trúc cho Luận văn)
- **Cơ chế:** Tái cấu trúc BBox Head từ MLP 3 tầng sang **Distribution Head 16 bins** ($256 \to 64$ kênh), áp dụng **FDR residual refinement** từng tầng decoder và **GO-LSD self-distillation** ([Peng et al., 2024](07_BBox_Regression/2024_Peng_D-FINE.pdf)).
- **Lợi ích:** Giải quyết triệt để ranh giới mờ nhòe của thức ăn, thu hẹp khoảng cách mAP@50 và mAP@75, tạo đóng góp tính mới cao nhất.

---

## 📊 Bảng Đối chuẩn Số liệu Gốc (Ground-Truth Benchmarks)

*Trích xuất trực tiếp từ Bảng 1 & 2, Trang 7, bài báo gốc RF-DETR (ICLR 2026):*

| Mô hình | Số tham số | GFLOPs | Độ trễ (FP16 TRT) | AP | AP50 | AP75 | $AP_S$ (Nhỏ) | $AP_M$ (Vừa) | $AP_L$ (Lớn) |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR (M)** | **33.7M** | **78.8** | **4.4 ms** | **54.7** | **73.5** | **59.2** | **36.1** | **59.7** | **73.8** |
| D-FINE (M) | 19.2M | 56.6 | 5.4 ms | 55.0 | 72.6 | 59.7 | **37.6** | 59.4 | 71.7 |
| LW-DETR (M) | 28.2M | 83.9 | 4.4 ms | 52.6 | 72.0 | 56.6 | 32.5 | 57.6 | 70.5 |
| RT-DETR (R18)| 36.0M | 100.0 | 4.4 ms | 49.0 | 66.6 | 53.3 | 32.8 | 52.1 | 65.0 |

---

## 📖 Hướng dẫn Đọc nhanh (Quick Reading Guide)

1. **Hiểu toàn cảnh bài toán & giải pháp:** Đọc file [HUONG_DAN_NGHIEN_CUU_VA_DOC_TAI_LIEU.md](HUONG_DAN_NGHIEN_CUU_VA_DOC_TAI_LIEU.md).
2. **Xem đối chiếu kỹ thuật chi tiết:** Đọc file [Analysis/02_Baseline_Comparison/Detailed_Baseline_Comparison.md](Analysis/02_Baseline_Comparison/Detailed_Baseline_Comparison.md).
3. **Xem code triển khai NWD Loss:** Đọc file [Analysis/05_Integration/Architectural_Integration_Proposals.md](Analysis/05_Integration/Architectural_Integration_Proposals.md).

---

## 📜 Trích dẫn (Citation)

Nếu tài liệu này hỗ trợ nghiên cứu của bạn, xin vui lòng trích dẫn các tài liệu nền tảng tương ứng:

```bibtex
@inproceedings{robinson2025rfdetr,
  title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author={Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

@article{peng2024dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Wang, Shenghao and Wang, Boying and others},
  journal={arXiv preprint arXiv:2410.13842},
  year={2024}
}

@article{wang2021nwd,
  title={A Normalized Gaussian Wasserstein Distance for Tiny Object Detection},
  author={Wang, Jinwang and Xu, Chang and Yang, Wen and Yu, Lei},
  journal={arXiv preprint arXiv:2110.13389},
  year={2021}
}
```
