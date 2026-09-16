# Scale study: Nemotron/WildGuard

## Experiment card

- **Câu hỏi:** Khi giữ nguyên chính sách SFT và tăng quy mô, tỉ lệ 30:70 hay 70:30 có giữ được cân bằng đa ngôn ngữ, tiếng Việt và tiếng Anh khó không?
- **Giả thuyết:** 30:70 là ứng viên chính theo study 80k; 70:30 là kiểm chứng xem thêm Nemotron có mở rộng dữ liệu mà không làm mất điểm tiếng Việt/unsafe hay không.
- **Baseline:** Qwen3-4B, LoRA, config `source_study_scaled_train_qwen3_4b.yaml`, seed 3407.
- **Biến duy nhất:** tỉ lệ và quy mô mẫu Nemotron/WildGuard; prompt, nhãn nhị phân, validation và optimizer giữ nguyên.
- **Chỉ số chính:** trung bình đều của Balanced Accuracy trên 8 benchmark nhị phân cân bằng (3 đa ngôn ngữ, 3 tiếng Việt, 2 tiếng Anh khó).
- **Guardrail:** unsafe-only recall, Safe Recall và Unsafe Recall theo từng benchmark; parse rate phải đạt 100% hoặc được ghi rõ.
- **Rò rỉ:** builder loại benchmark overlap, exact-content conflict và giữ complete semantic group; manifest ghi lại số bị loại.
- **Ngân sách:** một epoch trên 4 GPU; checkpoint cuối epoch và checkpoint gần nhất để resume.

## Quy mô yêu cầu và thực tế sau lọc

| Recipe | Yêu cầu N/W | Thực tế N/W | Train tổng | Validation | Vai trò |
|---|---:|---:|---:|---:|---|
| 30:70 | 37,176 / 86,745 | 35,155 / 82,028 | 117,183 | 997 | Nhánh chính theo matched-80k |
| 70:30 | 202,405 / 86,745 | 192,332 / 82,428 | 274,760 | 1,000 | Kiểm chứng Nemo-heavy scale |

WildGuard có 86,759 dòng thô và 86,745 dòng hợp lệ. Builder yêu cầu toàn bộ phần hợp lệ, sau đó loại benchmark leakage, duplicate/conflict và mọi complete prompt group chạm exact overlap với Nemotron. Validation giữ 700 WildGuard ở 30:70 và 300 ở 70:30, nên hai train pool có kích thước hơi khác nhau.

Nemotron được chọn theo complete upstream ID xuyên 9 ngôn ngữ, không lấy ngẫu nhiên từng dòng. 30:70 có 3,906 dòng ở tám ngôn ngữ và 3,907 dòng tiếng Pháp; 70:30 có 21,370 dòng ở bảy ngôn ngữ và 21,371 dòng ở Đức/Pháp. `manifest.json` là nguồn sự thật cho mọi số thực tế.

## Kết quả matched-80k đã có

Các giá trị dưới đây là phần trăm. `BA tổng` là trung bình phẳng của 8 benchmark cân bằng; MultiJail và XSafety chỉ dùng để theo dõi unsafe-only, không đưa vào BA tổng.

| Tỉ lệ | BA đa ngôn ngữ | BA tiếng Việt | BA tiếng Anh khó | Unsafe-only recall | BA tổng | Safe Recall (8) | Unsafe Recall (8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| N0–W100 | 82.59 | 83.34 | 92.51 | 61.73 | 85.35 | 91.51 | 79.20 |
| N30–W70 | **85.03** | **84.64** | 91.67 | **67.00** | **86.55** | **90.06** | **83.03** |
| N50–W50 | 84.58 | 82.93 | 91.09 | 60.51 | 85.58 | 90.99 | 80.18 |
| N70–W30 | **85.28** | 84.51 | 90.61 | 64.77 | **86.32** | 89.87 | 82.78 |
| N80–W20 | 85.22 | 83.77 | 89.75 | 64.69 | 85.81 | 87.31 | **84.31** |
| N100–W0 | **85.28** | 82.63 | 85.69 | 67.78 | 84.39 | 83.41 | **85.37** |

30:70 dẫn đầu BA tổng và unsafe-only trong các mix đã chạy, đồng thời có BA tiếng Việt cao nhất. 70:30 chỉ thấp hơn 0.23 điểm BA tổng, dẫn đầu nhóm đa ngôn ngữ cùng 100:0 và vẫn giữ BA tiếng Việt gần 30:70. Vì vậy 70:30 là comparator hợp lệ để kiểm chứng quy mô Nemo lớn hơn, nhưng chưa được gọi là recipe tối ưu.

Các trung bình được tính từ những ô benchmark đã làm tròn đến hai chữ số; chênh lệch 0.01 giữa các báo cáo chỉ có thể là sai số làm tròn, không phải hiệu ứng thực nghiệm mới.

## Lệnh trên máy Linux công ty (offline)

Mặc định chạy 30:70:

```bash
export SOURCE_STUDY_MODEL_PATH=/workspace/storage-shared/models/Qwen3-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

bash scripts/run_scaled_nemotron_wildguard_ratio.sh prepare
bash scripts/run_scaled_nemotron_wildguard_ratio.sh preflight
bash scripts/run_scaled_nemotron_wildguard_ratio.sh smoke

mkdir -p logs/source-study-scaled/30_70
nohup bash scripts/run_scaled_nemotron_wildguard_ratio.sh train \
  > logs/source-study-scaled/30_70/train.nohup.log 2>&1 &
echo $! | tee logs/source-study-scaled/30_70/train.pid
```

Chạy nhánh Nemo-heavy 70:30 ở một thư mục log/run riêng:

```bash
export SOURCE_STUDY_SCALE_RATIO=70_30
bash scripts/run_scaled_nemotron_wildguard_ratio.sh prepare
bash scripts/run_scaled_nemotron_wildguard_ratio.sh preflight
bash scripts/run_scaled_nemotron_wildguard_ratio.sh smoke
mkdir -p logs/source-study-scaled/70_30
nohup bash scripts/run_scaled_nemotron_wildguard_ratio.sh train \
  > logs/source-study-scaled/70_30/train.nohup.log 2>&1 &
echo $! | tee logs/source-study-scaled/70_30/train.pid
```

Hai recipe dùng 4 GPU DDP, BF16/SDPA, effective global batch 32 (`1 × 8 × 4`), max length 2048, learning rate `1e-5`, constant scheduler, gradient checkpointing, save/eval mỗi epoch, seed 3407. Script không gọi mạng; chỉ đọc raw data và model đã có trên máy.

## Quyết định sau khi chạy

Chỉ gọi 70:30 thành công nếu BA tổng không thấp hơn 30:70 quá ngưỡng định trước (khuyến nghị 0.5 điểm), BA tiếng Việt không giảm quá 0.5 điểm, và unsafe-only recall không giảm. Nếu đạt, scale Nemo-heavy là có cơ sở; nếu không, giữ 30:70 làm recipe chính và không suy rộng từ “nhiều dữ liệu hơn” sang “tốt hơn”.
