# B200 cũ và decision-guard mới: mốc tiếp tục

Ngày ghi nhận: 2026-10-12. Trạng thái B200 dưới đây dựa trên log/status người dùng
đã gửi trong hội thoại, không phải kiểm tra trực tiếp cluster hôm nay. Không coi
file còn tồn tại hoặc GPU hiện rảnh là đã được xác nhận.

## 1. Data đã dùng ở B200

Root repository khi chạy:
`/workspace/storage-shared/nlp/anhhbt/multilingual-safety-guard-runner-b200-full-data-study`.

| Bộ đã cài | ZIP | Thư mục data | Số train |
| --- | --- | --- | ---: |
| Nemo:Wild 30:70 scale | `nemotron_wildguard_scaled_max_30_70_v2.gated.zip` | `work/source-study-scaled/nemotron30_wildguard70_scaled` | khoảng 117k; cần đọc manifest v2 ở máy train để chốt chính xác |
| Nemo:Wild 70:30 scale | `nemotron_wildguard_scaled_max_70_30_v2.gated.zip` | `work/source-study-scaled/nemotron70_wildguard30_scaled` | 274760 = 192332 Nemo + 82428 Wild; validation1000 |

SHA ZIP30:70: `7b65704a2e305d725b2da99b59eda0f89266d5d59aa5372b0ed319578305c628`.
SHA ZIP70:30: `61e43353485d1d0e3986507b2b4524a2f90afa5f42dc98c67575f8126e61ccf4`.

Nemo giữ 9 ngôn ngữ với lượng gần bằng nhau; chọn theo semantic group và bỏ
overlap/conflict upstream. Không phải raw420k hoặc full mọi nguồn. Hai tỷ lệ có
tổng số mẫu khác nhau, nên so scale này không cô lập tác động tỷ lệ thuần túy.

Cẩn thận: manifest30:70 local hiện có 117183/997 nhưng khác đời với bundle v2
đã chuyển lên B200. Không dùng con số của manifest local đó để khẳng định dữ liệu
thực tế trên cluster. Cần đọc manifest/hash từ đúng ZIP v2 hoặc thư mục B200.

Bundle eval chung:
`zip/qwen3_safety_benchmark_total_v2.zip` →
`work/eval-hosted/benchmark-total/qwen3_safety_benchmark_total.jsonl`.
129761 mẫu, 12 benchmark. SHA JSONL:
`bdb2b4d1ecac08d2b6f4915b7a3274282dd0833f0c9f6d6f25671cbbe35f635a`.
SHA ZIP: `d8b5dba2edbac87895c2709c79761929488ce959a513ecd0942ec663427897b7`.

## 2. Train B200 đã xong, nhưng chưa eval model sau train

Các run dưới `runs-source-study-scaled/qwen3_32b/`:

| Run | Steps | Thời gian train | Train loss | Validation loss | Trạng thái cuối được gửi |
| --- | ---: | --- | ---: | ---: | --- |
| `nemotron30_wildguard70_scaled_1epoch` | 3662 | ~1h09m | 0.02152 | 0.01899 | final adapter + checkpoint3662 + run_complete; chưa merge/eval |
| `nemotron70_wildguard30_scaled_1epoch` | 8587 | ~2h58m | 0.02604 | 0.02630 | final adapter + checkpoint8587 + run_complete; chưa merge/eval |

Qwen3-8B/4B scale: lần kiểm tra cuối chưa có smoke, train, merge hoặc eval.
Qwen235B hosted eval từng được khởi động, nhưng chưa có kết quả cuối trong bằng
chứng được gửi; không ghi là đã hoàn tất. Full-source SEA/Nemo/Wild là kế hoạch,
không phải các run B200 đã được chứng minh xong.

Đây là SFT LoRA decoder sinh output, **không phải full-parameter training**:
BF16/SDPA, Q/V r8 alpha32 dropout.05, 1epoch, constantLR1e-5, seq2048,
seed3407, globalbatch32 trên8GPU, gradient checkpointing. Lưu và validation cuối
epoch. `scripts/run_scaled_qwen_matrix_b200.sh train` chỉ train;
phase `run` mới nối train → merge → eval-smoke → eval-full.

## 3. Qwen3-32B base đã eval đầy đủ

Output: `runs/qwen3-32b-eval-unified/full_2`.
Đây là **base chưa train**, không phải hai adapter ở mục2.
No-think, max-model-len40000, max-new-tokens16, truncate2 mẫu, parse100%.

| Benchmark | n | BA | Safe recall | Unsafe recall |
| --- | ---: | ---: | ---: | ---: |
| CultureGuard-JB | 13266 | 88.31 | 88.70 | 87.92 |
| CultureGuard-Standard | 24993 | 78.60 | 73.56 | 83.64 |
| PolyGuard-Prompts | 30906 | 82.97 | 89.93 | 76.00 |
| LinguaSafe-VI | 3884 | 81.58 | 83.07 | 80.09 |
| SEA-VI | 1840 | 84.27 | 82.00 | 86.54 |
| SEALSBench-VI | 26644 | 89.48 | 99.46 | 79.51 |
| ViSafe-VI | 3212 | 94.53 | 97.95 | 91.12 |
| WildGuardTest-EN | 3408 | 86.52 | 87.97 | 85.07 |
| XSTest-EN | 450 | 85.90 | 72.80 | 99.00 |
| MultiJail | 1260 | - | - | 88.57 |
| XSafety | 19600 | - | - | 48.05 |
| bokhch_safety (prompt, gán unsafe theo yêu cầu) | 298 | - | - | 93.29 |
| ALL pooled | 129761 | 80.88 | 88.96 | 72.80 |

BA = (safe recall + unsafe recall)/2. Unsafe-only không có BA hai lớp.
ALL pooled bị ảnh hưởng tỷ lệ lớp/số mẫu giữa benchmark; không dùng nó một mình
để chọn model. Giữ cùng prompt, bundle/hash, truncation, decoding và độ đo khi so.
Đặc biệt ba bộ unsafe-only không đo được false positive/khả năng giữ safe.

## 4. Decision-guard mới và cách so cho sạch

`decision_guard/` dùng đúng data70:30/hash ở mục1, nhưng kiến trúc/prompt mới và
Qwen3-4B. Smoke pointer36 đã pass1step trên1L40S/NeMo26.08, checkpoint nạp lại và
forward khớp. Chưa train đầy đủ, chưa multiGPU hoặc smoke các arm khác.

Plan: JSON36 baseline; binary36/27/18; KEV-style pointer36/27/18. Mỗi arm khởi tạo
lại từ cùng Qwen3-4B, cùng data/seed/globalbatch và common policy. Đây là custom
Bridge/MCore/TE + torchDDP, không phải HFTrainer cũ; checkpoint gồm cả head, không
đưa thẳng vào `merge_adapter.py` hoặc vLLM cũ.

Để tiếp tục:

1. Đọc lại status/manifests trên B200 và giữ các adapter32B đã có; không train
   lại hai run complete chỉ vì chưa merge/eval.
2. Merge/eval hai adapter32B bằng bundle/setting của base ở mục3. Không dùng
   default8192 của matrix cũ nếu muốn so với base40000 một cách nhất quán.
3. Với nghiên cứu tốc độ/kiến trúc, dùng **Qwen3-4B JSON36 mới** làm baseline ghép
   cặp cho head arms. Kết quả32B cũ chỉ là tham chiếu thêm; model/prompt/backend
   khác thì không quy mọi chênh lệch cho việc thay head/cắt layer.
4. Smoke từng arm ở đúng GPU-count/config sẽ train; smoke1GPU không xác nhận
   multiGPU. Chỉ sau đó mới chạy study và benchmark/latency đồng nhất.

Máy công ty: chuyển code/data/model thủ công, chạy offline; không `git pull`,
HF download, curl/wget hoặc auto-install. Chỉ pip thật sự cần thiết là ngoại lệ.
