# Audit Luna/Sol và Gemini cho nguồn tiếng Việt Phase 2

Ngày audit độc lập: 2026-08-28

## Kết luận chọn nguồn

Chọn **Gemini** cho run Phase 2 tốn GPU đầu tiên. Lý do không phải vì Luna/Sol lỗi: cả hai bộ đều qua kiểm tra cấu trúc. Gemini được ưu tiên vì đã có bằng chứng downstream cùng contract P/PR: Q2 train 101.274 ví dụ trong 1 epoch đạt SEA-VI 86,03%. Luna/Sol nên giữ làm ablation/pilot matched sau đó; hiện chưa có checkpoint downstream để chứng minh nó tốt hơn Gemini.

## Kết quả audit nguồn

| Thuộc tính | Luna/Sol | Gemini |
|---|---:|---:|
| Train UID | 40.007 | 40.007 |
| Validation UID | 2.445 | 2.445 |
| Test UID | 2.964 | 2.964 |
| Tổng UID | 45.416 | 45.416 |
| UID overlap giữa split | 0 | 0 |
| Hard error | 0 | 0 |
| Cảnh báo nguồn không render được | 2 | 2 |

Hai bộ có cùng UID set và cùng source contract (English prompt/response, split, nhãn, category) trên cả ba split. Hai cảnh báo là bản ghi nguồn có prompt trống; pipeline bỏ qua thay vì tạo ví dụ rỗng. Test split chỉ được audit, không được đưa vào Stage 2 train.

## Thành phần dịch

- Luna/Sol: `gpt-5.6-luna` 45.060 bản ghi; `gpt-5.6-sol-web` 356 bản ghi.
- Luna/Sol dispositions: hard-pass 45.004; warning-pass 372; manual-audit-pass 31; accepted-known-issue 2; revalidated-pass 7.
- Gemini theo provider: Gemini 44.814; Codex human review 174; manual web review 348; Gemini revision 29; manual Terra revision 51.

## Contract P/PR đã xác nhận

Khi chỉ lấy train + validation và dedupe prompt view theo semantic hash tiếng Anh, mỗi bộ cho cùng 53.832 ví dụ VI:

| Split/view | Số ví dụ |
|---|---:|
| Train | 50.637 |
| Validation | 3.195 |
| P | 33.215 |
| PR | 20.617 |

JSON audit đầy đủ, SHA-256 từng file và 100 lỗi/cảnh báo đầu (nếu có) nằm trong `reports/vi_translation_audit.json`.

## Số liệu train/SEA đã có trên máy này

| Run | Dữ liệu train | Epoch | SEA-VI accuracy |
|---|---|---:|---:|
| Q1 Qwen3Guard-Gen-4B zero-shot | Không train | — | 82,83% |
| Q2 Qwen3Guard-Gen-4B + Gemini LoRA | 101.274 EN-VI P/PR | 1 | **86,03%** |
| D1 Nemotron-8B-v3 zero-shot | Không train | — | 83,21% |
| D2 Nemotron pilot LoRA | 8.192 balanced EN-VI | pilot | 83,26% |
| D3 Nemotron full VI LoRA | 50.637 VI P/PR | 1 | — |

D3 được báo cáo 83,53% trên SEA paired toàn bộ 3.680 ví dụ và 86,04% trên tập Vietnamese gộp Nemotron+SEA; tài liệu hiện có không tách đúng SEA-VI 1.840 ví dụ, nên không điền suy đoán.

## So với các model đã train/đo SEA trên máy công ty

Nguồn: workbook báo cáo hiện tại, sheet `4-SEA_VI_chi_tiet`.

| Model | SEA-VI Acc | Macro-F1 | Unsafe Recall | Unsafe F1 | Parse |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct after 1ep | **84,40%** | 84,36% | 84,45% | 83,53% | 100% |
| Qwen3-8B after 5ep | 84,29% | 84,27% | 85,85% | 83,66% | 100% |
| Qwen3.5-4B after 1ep, merge đúng | 84,13% | 84,07% | 83,41% | 83,12% | 100% |
| Qwen3-8B after 1ep | 83,91% | 83,88% | 84,69% | 83,14% | 100% |
| Nemotron-8B-v3 | 83,04% | 82,87% | 77,84% | 81,14% | 99,40% |
| Qwen3-8B base | 82,77% | 82,63% | 78,77% | 81,07% | 100% |
| Qwen3.5-4B base | 82,50% | 82,43% | 81,32% | 81,32% | 100% |
| Qwen3-4B base | 81,36% | 80,94% | 71,00% | 78,11% | 100% |
| Llama-3.1-8B-Instruct base | 80,98% | 80,62% | 71,93% | 77,99% | 99,62% |
| Qwen3-1.7B base | 72,61% | 70,09% | 46,52% | 61,41% | 100% |

Q2 local cao hơn model công ty tốt nhất trong bảng 1,63 điểm phần trăm (86,03 so với 84,40). Đây chỉ là so sánh định hướng: model, checkpoint, thời điểm runner và recipe không hoàn toàn đồng nhất, nên Phase 2 vẫn phải đánh giá lại bằng chính runner/manifest hiện tại.
