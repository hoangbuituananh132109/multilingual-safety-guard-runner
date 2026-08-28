# Runbook Phase 2 trên máy công ty

Mục tiêu của runbook là fail sớm trước khi tốn GPU. Không bỏ qua blocker và không chạy full train trước khi smoke 3 step thành công.

## 1. Chuẩn bị nguồn

Giữ đúng layout trong `stage2_config.yaml`. Hai file lớn đã kiểm tra trên máy local phải được chép sang máy công ty và xác minh SHA-256:

```bash
sha256sum input/stage2/reasoning/aegis_v2_efficient_reasoning.jsonl input/stage2/nemotron35/train.parquet
```

Kết quả phải lần lượt là:

```text
d7e9e43ea92a42cc95a20a710bb45072fd8eec75a331c1393ce96029ddfc5d0c
089052c8f1a63e45ac0d0825c47b13d5ce4b94de65bdea8d669bb18731805c53
```

Sau khi tài khoản Hugging Face đã được cấp quyền WildGuard và đã `hf auth login`, tải đúng file train bằng một lệnh (không tải WildGuardTest):

```bash
mkdir -p input/stage2/wildguard && hf download allenai/wildguardmix wildguard_train.parquet --repo-type dataset --revision d29c47f41c8b51348b5c8e8c81c039b3132b66d1 --local-dir input/stage2/wildguard
python3 scripts/prepare_wildguard.py --input input/stage2/wildguard/wildguard_train.parquet --output input/stage2/wildguard/wildguardtrain.jsonl
```

Không tải hoặc train `wildguardtest`.

Snapshot V3 phải nằm ở `../offline-bundle-work/snapshots/nemotron-9lang` và file `offline_source.json` phải ghi revision `a3f7ecb3433d1933701a83f18de16c36934a7f51`. Bộ Gemini phải nằm ở `../data/final`. Có thể override đường dẫn bằng cách sửa bản config dành riêng cho máy công ty, không sửa dữ liệu nguồn.

## 2. Chuẩn bị benchmark test trước leakage audit

Đặt các nguồn benchmark theo `config.yaml`, sau đó chạy:

```bash
python3 run.py --config config.yaml prepare
```

Kiểm tra `work/benchmarks` đã có CultureGuard, SEA-VI, XSafety và các benchmark published được chuẩn bị trên máy. Builder Phase 2 sẽ từ chối full-ready nếu đường dẫn test không tồn tại.

## 3. Audit và inventory, chưa dùng GPU

```bash
python3 stage2.py audit-translations --config stage2_config.yaml --output reports/vi_translation_audit_company.json
python3 stage2.py inventory --config stage2_config.yaml --translation-source gemini --output reports/stage2_inventory_company.json
```

Yêu cầu: Gemini `ready=true`, hard error 0, split overlap 0. Inventory không còn blocker.

## 4. Data smoke

```bash
python3 stage2.py build --config stage2_config.yaml --translation-source gemini --output-dir work/stage2/smoke_gemini --smoke-per-source 20
python3 stage2.py validate --data-dir work/stage2/smoke_gemini
```

Yêu cầu: `valid=true`, có đủ 5 dataset source, có cả train/validation, taxonomy ON/OFF và THINK/NO-THINK.

## 5. Full build, vẫn chưa dùng GPU

```bash
python3 stage2.py build --config stage2_config.yaml --translation-source gemini --output-dir work/stage2/full_gemini
python3 stage2.py validate --data-dir work/stage2/full_gemini
```

Mở `work/stage2/full_gemini/manifest.json` và lưu cùng run. Chỉ tiếp tục nếu `full_ready=true`, blocker rỗng, validation hợp lệ và count nguồn hợp lý so với inventory.

## 6. GPU smoke 3 optimizer step

`MODEL_PATH_STAGE1_MERGED` phải trỏ tới model Stage 1 đã merge hoàn chỉnh, không phải thư mục LoRA adapter:

```bash
export MODEL_PATH_STAGE1_MERGED=/absolute/path/to/stage1/merged
python3 -m torch.distributed.run --standalone --nproc-per-node=4 core/train.py --config stage2_train_qwen3_8b.yaml --max-steps 3 --skip-eval --no-checkpoints --no-final-save
```

Smoke này là disposable; không dùng kết quả để báo metric. Xác nhận model/tokenizer load được, tokenization không lỗi, loss hữu hạn và đủ 3 step.

## 7. Full train có checkpoint

Chỉ sau khi sáu bước trên pass:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=4 core/train.py --config stage2_train_qwen3_8b.yaml --skip-eval
```

Recipe mặc định: fresh LoRA trên merged Stage 1, 5 epoch, LR `3e-6`, per-device batch 1, gradient accumulation 8, effective global batch 32 trên 4 GPU, max length 2048, LoRA r8/alpha32/dropout0.05, checkpoint mỗi 500 step. Nếu bị gián đoạn, giữ nguyên toàn bộ run directory rồi resume:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=4 core/train.py --config stage2_train_qwen3_8b.yaml --skip-eval --resume
```

Không dùng `--no-checkpoints` cho full train. Không tự động đánh giá hay xóa checkpoint từ script train.

## 8. Chuyển bundle offline

Bundle training-only có thể giải nén và kiểm SHA-256 bằng một lệnh:

```bash
python3 scripts/unpack_stage2_bundle.py --zip stage2_bundle.zip --output-dir work/stage2/full_gemini_bundle
python3 stage2.py validate --data-dir work/stage2/full_gemini_bundle/data
```

Bundle policy mới có WildGuard hiện đã tạo local tại `D:\Downloads\Safety Dataset\multilingual-safety-guard-runner\company-transfer\stage2_gemini_policy_training_ready.zip` và upload tại [HF offline-zips](https://huggingface.co/datasets/TuanAnhHoangBui/safety-guard-offline-zips). Sau khi giải nén, dùng `stage2_train_qwen3_8b_bundle.yaml`; không đưa benchmark test vào archive hoặc HF dataset repo; bundle có WildGuard chỉ chia sẻ trong phạm vi đã được cấp phép.

## 9. Ablation để tách đóng góp dữ liệu

Giữ nguyên `MODEL_PATH_STAGE1_MERGED`, seed `3407`, LoRA, batch và benchmark
protocol; chỉ thay dataset/config. Các bundle `stage2_ablation_vi_gemini.zip`
và `stage2_ablation_reasoning_only.zip` đã nằm trong HF offline-zips.

| Run | Data | Epoch | Config |
|---|---|---:|---|
| VI-1E | Gemini VI only | 1 | `stage2_train_qwen3_8b_vi_gemini_1epoch.yaml` |
| VI-5E | Gemini VI only | 5 | `stage2_train_qwen3_8b_vi_gemini_5epoch.yaml` |
| Reasoning-5E | Reasoning only, random THINK/NO-THINK | 5 | `stage2_train_qwen3_8b_reasoning_only_5epoch.yaml` |
| Full-5E | V3 + VI + reasoning + N35 + WildGuard | 5 | `stage2_train_qwen3_8b_bundle.yaml` |

Chỉ báo cáo hiệu quả sau khi đánh giá cùng một bộ benchmark, cùng decoding và
tách rõ prompt-only/response-only; không so sánh loss giữa các run như metric
chính. Để chạy một ablation, giải nén bundle tương ứng vào đúng thư mục mà
YAML chỉ định rồi chạy GPU smoke 3 step trước full train.

```bash
hf upload <ORG>/<DATASET_REPO> stage2_bundle.zip stage2_bundle.zip --repo-type dataset
```
