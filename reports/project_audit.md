# Audit nhánh `no-dataset` trước Phase 2

Ngày audit: 2026-08-28  
Repository: `multilingual-safety-guard-runner`  
Nhánh: `no-dataset`

## Kết luận

Repository đúng là project Guard, không phải PIPPA. Sau khi thêm pipeline Phase 2, worktree sạch và `HEAD` khớp `origin/no-dataset` tại commit `cada7d3`.

Nhánh hiện tại đủ cho Stage 1: chuẩn bị Nemotron V3, SEA/XSafety, train LoRA/full, resume checkpoint, đánh giá Transformers/vLLM và merge LoRA cho vLLM. Tuy nhiên code cũ chưa thể thực hiện đúng kế hoạch Phase 2 vì chỉ có một prompt taxonomy-on, chỉ có target JSON no-think, chưa có semantic registry đa nguồn, chưa chọn một bộ dịch VI, chưa lọc Nemotron 3.5 và chưa audit leakage với benchmark test.

## Contract Stage 1 cần giữ nguyên

- Chỉ dùng view `P` và `PR`; `R` bị cấm.
- `NO-THINK` là đường đánh giá mặc định.
- Parser/metric hiện hữu không bị thay đổi.
- Checkpoint định kỳ mặc định mỗi 500 optimizer step, giữ tối đa 3 checkpoint.
- vLLM không đọc trực tiếp LoRA trong luồng hiện tại; runner merge adapter trước khi đánh giá.

## Khoảng trống đã được bổ sung

- `stage2.py` có các lệnh audit bản dịch, inventory, build và validate.
- `core/stage2_data.py` dựng semantic registry trước khi render; chọn một ngôn ngữ V3 theo hash ổn định; dedupe prompt view; tách train/validation theo semantic ID; lọc text-only synthetic Nemotron 3.5; loại topic-following và multimodal.
- Render hỗ trợ taxonomy ON/OFF và THINK/NO-THINK. WildGuard luôn taxonomy-off; category ngoài N23 luôn taxonomy-off.
- `core/train.py` vẫn đọc được JSONL Stage 1 cũ, đồng thời nhận `instruction`/`target` đã audit của Stage 2.
- Output Phase 2 nằm dưới `work/stage2` và `runs-stage2`, không ghi đè run Stage 1.

## Nguồn và revision đã khóa

| Nguồn | Revision | Trạng thái local |
|---|---|---|
| Nemotron Safety Guard Dataset V3 | `a3f7ecb3433d1933701a83f18de16c36934a7f51` | Có snapshot 9 ngôn ngữ |
| WildGuardMix / `wildguardtrain` | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` | Có local; 86.759 raw records |
| Nemotron Content Safety Reasoning | `792b0715f519c0750d63b73af2bf33ddd9ac3887` | Có file reasoning |
| Nemotron 3.5 Content Safety | `841f5023b8db12b484180c6121bb10fcf400d1c3` | Có train parquet |

## Chặn full-ready

1. Local chưa có `work/benchmarks`; vì vậy chưa thể hoàn thành leakage audit đối với toàn bộ benchmark test. Full build vẫn ghi rõ blocker thay vì bỏ qua điều kiện này.

Không có full training nào được khởi chạy trong audit này.
