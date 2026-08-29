# Báo cáo dữ liệu Phase 2 sau full build

> Trạng thái lịch sử: báo cáo này mô tả bundle schema v2 đã tạo ngày
> 2026-08-28. Audit ngày 2026-08-29 xác nhận bundle đó không còn được xem là
> policy-correct hoặc validation-clean. Không dùng các câu “validate 0 lỗi” bên
> dưới làm bằng chứng cho causal comparison. Xem
> `reports/stage2_methodology_audit_20260829.md`.

Ngày inventory: 2026-08-28  
Nguồn VI được chọn cho run đầu: Gemini  
Seed cố định: 3407

Policy hiện hành: mỗi semantic record chỉ render một view; V3/VI chọn
taxonomy ON 75% và OFF 25% bằng hash ổn định (nhưng không thể ON nếu không có
category N23); Reasoning chọn THINK/NO-THINK 50/50 nếu có trace; WildGuard
luôn OFF; N35 ON khi có category N23, S24-only/không có category là OFF.

## Inventory hiện có

Các số dưới đây là số ví dụ `P/PR` sau semantic registry và prompt-view dedupe, chưa render full instruction để tránh tạo artifact lớn trên máy local.

| Nguồn | Train | Validation | P | PR | Tổng |
|---|---:|---:|---:|---:|---:|
| V3 semantic replay, một language/upstream ID | 53.053 | 3.356 | 36.445 | 19.964 | 56.409 |
| Gemini VI full (test bị loại) | 50.637 | 3.195 | 33.215 | 20.617 | 53.832 |
| Content Safety Reasoning | 35.467 | 437 | 23.110 | 12.794 | 35.904 |
| Nemotron 3.5 selected | 9.559 | 99 | 5.829 | 3.829 | 9.658 |
| WildGuardTrain | 85.036 | 1.280 | 48.382 | 37.934 | 86.316 |
| **Tổng semantic registry** | **233.752** | **8.367** | **146.981** | **95.138** | **242.119** |

Nguồn reasoning local có chính xác 27.459 record trong file efficient-reasoning. Semantic registry giữ các interaction khác nhau dù upstream ID trùng, rồi dedupe prompt view. Con số “28k” trong kế hoạch/tài liệu là tên quy mô làm tròn, không được thay cho count local.

Full render mới: 233.752 train + 8.367 validation = 242.119 rows; taxonomy ON 49.854, OFF 192.265; THINK 12.965, NO-THINK 229.154. ON thấp hơn 75% toàn cục vì V3/VI prompt-only views không mang category và WildGuard luôn OFF; 75/25 chỉ áp dụng cho các record có taxonomy N23 hợp lệ.

Nemotron 3.5 bắt đầu từ train parquet 88.688 record, sau đó chỉ giữ safety + text-only + synthetic/adversarial, loại topic-following, multimodal, duplicate và overlap với các nguồn đã chọn. Inventory độc lập là 9.712 P/PR; full build sau loại overlap còn 9.658 (5.829 P + 3.829 PR). Có 46 ví dụ mang category ngoài N23 là `Economic Harm`; các ví dụ này bị ép taxonomy-off, không gán sai vào N23.

WildGuardTrain có 86.759 raw records; sau prompt-view dedupe còn 86.316 semantic P/PR. WildGuard luôn taxonomy-off và giữ metadata `response_refusal_label`, `adversarial`, `subcategory`, không fabricate N23.

## Revision và hash local

| Artifact | Revision / SHA-256 |
|---|---|
| V3 | revision `a3f7ecb3433d1933701a83f18de16c36934a7f51` |
| Reasoning JSONL | revision `792b0715f519c0750d63b73af2bf33ddd9ac3887`; SHA-256 `d7e9e43ea92a42cc95a20a710bb45072fd8eec75a331c1393ce96029ddfc5d0c` |
| Nemotron 3.5 parquet | revision `841f5023b8db12b484180c6121bb10fcf400d1c3`; SHA-256 `089052c8f1a63e45ac0d0825c47b13d5ce4b94de65bdea8d669bb18731805c53` |
| WildGuardTrain | revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`; parquet SHA-256 `02ECEA8A724A9146A1E473A95A7CDF262ADFE9C7D5408953CA86D2FCFBDC8953` |

## Quy tắc render

- V3: một ngôn ngữ được chọn ổn định theo hash của upstream ID trong số các ngôn ngữ thực sự có ID đó; cùng ID luôn chọn cùng ngôn ngữ.
- VI: chỉ đúng một nguồn dịch trong mỗi build; test split không bao giờ được nạp.
- WildGuard: taxonomy-off bắt buộc.
- V3/Gemini VI: taxonomy ON/OFF 75/25 theo semantic ID; không nhân đôi view.
- N23-only reasoning: taxonomy ON; THINK/NO-THINK 50/50 theo semantic ID, không tạo cặp.
- N35: ON khi có N23 hợp lệ; S24-only/không có category và category ngoài N23-only là OFF.
- Category ngoài N23: taxonomy-off bắt buộc.
- Reasoning: native Qwen THINK target bắt đầu bằng rationale và đóng `</think>`; NO-THINK không chứa think tags.
- Toàn bộ nguồn được trộn bằng hash ổn định, không train tuần tự theo nguồn.

## Smoke test đã chạy

- Gemini: 160 ví dụ (80 train, 80 validation), 4 nguồn local, validate 0 lỗi.
- Full Gemini + WildGuard: 242.119 rendered rows, validate 0 lỗi.
- Luna/Sol: 160 ví dụ (80 train, 80 validation), 4 nguồn local, validate 0 lỗi.
- Unit tests: 7/7 pass, gồm taxonomy-off, unknown category, native THINK target, random non-paired dual thinking, stable taxonomy view và cross-split leakage detection.
- Full build đã chạy và validate 0 lỗi; full training chưa chạy.

## Điều kiện còn thiếu

`training_ready=true`. `full_ready=false` chỉ vì `work/benchmarks` chưa được chuẩn bị để builder lập hash và loại mọi overlap với test benchmark.

Inventory máy đọc được nằm trong `reports/stage2_inventory_gemini.json`; manifest smoke nằm dưới `work/stage2/smoke_gemini` và `work/stage2/smoke_luna_sol` (gitignored).
