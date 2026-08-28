# Báo cáo dữ liệu Phase 2 sau full build

Ngày inventory: 2026-08-28  
Nguồn VI được chọn cho run đầu: Gemini  
Seed cố định: 3407

## Inventory hiện có

Các số dưới đây là số ví dụ `P/PR` sau semantic registry và prompt-view dedupe, chưa render full instruction để tránh tạo artifact lớn trên máy local.

| Nguồn | Train | Validation | P | PR | Tổng |
|---|---:|---:|---:|---:|---:|
| V3 semantic replay, một language/upstream ID | 50.271 | 3.189 | 34.634 | 18.826 | 53.460 |
| Gemini VI full (test bị loại) | 50.637 | 3.195 | 33.215 | 20.617 | 53.832 |
| Content Safety Reasoning | 35.467 | 437 | 23.110 | 12.794 | 35.904 |
| Nemotron 3.5 selected | 9.605 | 107 | 5.856 | 3.856 | 9.712 |
| WildGuardTrain | 85.036 | 1.280 | 48.382 | 37.934 | 86.316 |
| **Tổng semantic registry** | **231.016** | **8.208** | **226.325** | **146.234** | **239.224** |

Nguồn reasoning local có chính xác 27.459 record trong file efficient-reasoning. Semantic registry giữ các interaction khác nhau dù upstream ID trùng, rồi dedupe prompt view. Con số “28k” trong kế hoạch/tài liệu là tên quy mô làm tròn, không được thay cho count local.

Nemotron 3.5 bắt đầu từ train parquet 88.688 record, sau đó chỉ giữ safety + text-only + synthetic/adversarial, loại topic-following, multimodal, duplicate và overlap nội bộ với các nguồn đã chọn. Kết quả còn 5.856 semantic record, render thành 9.712 P/PR. Có 46 ví dụ mang category ngoài N23 là `Economic Harm`; các ví dụ này bị ép taxonomy-off, không gán sai vào N23.

WildGuardTrain có 86.759 raw records; sau prompt-view dedupe còn 86.316 semantic P/PR. WildGuard luôn taxonomy-off và giữ metadata `response_refusal_label`, `adversarial`, `subcategory`, không fabricate N23.

## Revision và hash local

| Artifact | Revision / SHA-256 |
|---|---|
| V3 | revision `a3f7ecb3433d1933701a83f18de16c36934a7f51` |
| Reasoning JSONL | revision `792b0715f519c0750d63b73af2bf33ddd9ac3887`; SHA-256 `d7e9e43ea92a42cc95a20a710bb45072fd8eec75a331c1393ce96029ddfc5d0c` |
| Nemotron 3.5 parquet | revision `841f5023b8db12b484180c6121bb10fcf400d1c3`; SHA-256 `089052c8f1a63e45ac0d0825c47b13d5ce4b94de65bdea8d669bb18731805c53` |
| WildGuardTrain | revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`; parquet SHA-256 `02ECEA8A724A9146A1E473A95A7CDF262ADFE9C7D5408953CA86D2FCFBDC8953` |

## Quy tắc render

- V3: một ngôn ngữ được chọn ổn định theo hash của upstream ID; cùng ID luôn chọn cùng ngôn ngữ.
- VI: chỉ đúng một nguồn dịch trong mỗi build; test split không bao giờ được nạp.
- WildGuard: taxonomy-off bắt buộc.
- N23-only: taxonomy ON/OFF được chia ổn định theo seed.
- Category ngoài N23: taxonomy-off bắt buộc.
- Reasoning: THINK/NO-THINK được chia ổn định; NO-THINK vẫn là đường đánh giá mặc định.
- Toàn bộ nguồn được trộn bằng hash ổn định, không train tuần tự theo nguồn.

## Smoke test đã chạy

- Gemini: 160 ví dụ (80 train, 80 validation), 4 nguồn local, validate 0 lỗi.
- Full Gemini + WildGuard: 372.559 rendered rows, validate 0 lỗi.
- Luna/Sol: 160 ví dụ (80 train, 80 validation), 4 nguồn local, validate 0 lỗi.
- Unit tests: 6/6 pass, gồm taxonomy-off, unknown category, native THINK target, dual taxonomy/thinking views và cross-split leakage detection.
- Full build đã chạy và validate 0 lỗi; full training chưa chạy.

## Điều kiện còn thiếu

`training_ready=true`. `full_ready=false` chỉ vì `work/benchmarks` chưa được chuẩn bị để builder lập hash và loại mọi overlap với test benchmark.

Inventory máy đọc được nằm trong `reports/stage2_inventory_gemini.json`; manifest smoke nằm dưới `work/stage2/smoke_gemini` và `work/stage2/smoke_luna_sol` (gitignored).
