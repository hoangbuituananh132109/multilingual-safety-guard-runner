# Qwen3-32B scaled safety study trên 8 B200

## Experiment card

- **Câu hỏi:** Tăng base model từ Qwen3-4B lên Qwen3-32B, giữ nguyên common-contract data và SFT recipe, có cải thiện bộ safety suite không?
- **Baseline:** Qwen3-32B chưa fine-tune, taxonomy-off/no-think, cùng 10 benchmark.
- **Variant:** Qwen3-32B LoRA một epoch với một scaled mixture đã khóa. Không đổi prompt, target, seed hoặc effective global batch giữa base comparison và trained model.
- **Chỉ số chính:** BA tổng trên 8 benchmark nhị phân cân bằng.
- **Guardrail:** BA tiếng Việt, Safe Recall, Unsafe Recall, unsafe-only recall và parse rate.
- **Success:** trained BA tổng tăng so với 32B base; BA tiếng Việt không giảm quá 0.5 điểm và unsafe-only recall không giảm. So 70:30 với 30:70 chỉ được gọi là scale sensitivity, không phải tối ưu toàn cục.
- **Seed:** 3407, một seed trong cửa sổ B200; kết luận về phương sai seed vẫn là exploratory.

## Dữ liệu private trên Hugging Face

Repo: `TuanAnhHoangBui/qwen3-safety-guard-scaled-mixtures-private`

WildGuard là gated upstream derivative nên repo được giữ private. Tải ZIP bằng trình duyệt, đặt vào `zip/` trên máy công ty; không dùng lệnh tải mạng trong cluster.

```bash
python3 scripts/source_study_bundle.py install \
  --zip zip/nemotron_wildguard_scaled_max_30_70_v2.gated.zip \
  --output work/source-study-scaled/nemotron30_wildguard70_scaled

python3 source_study_natural.py validate \
  --data-dir work/source-study-scaled/nemotron30_wildguard70_scaled
```

## Setting 32B

- Model local mặc định: `/workspace/storage-shared/models/Qwen3-32B`.
- LoRA `r=8`, alpha `32`, dropout `0.05`, `q_proj/v_proj`.
- BF16, SDPA, max length 2048, gradient checkpointing.
- 1 epoch, constant LR `1e-5`, seed 3407.
- GPU được chọn trong `config/qwen3_32b_b200.env.sh`; runner tự suy ra world size.
- Microbatch mặc định 2/GPU; gradient accumulation được tính tự động để effective global batch luôn là 32.
- Checkpoint và validation cuối epoch; resume từ checkpoint gần nhất.
- Eval vLLM TP=1 trên một B200 vì 32B BF16 vừa một B200; chọn GPU bằng `SOURCE_STUDY_32B_EVAL_GPU`.

## Chọn GPU và setting tại một chỗ

Sửa duy nhất file `config/qwen3_32b_b200.env.sh`, ví dụ node chỉ còn GPU 2, 4, 6, 7:

```bash
export SOURCE_STUDY_32B_TRAIN_GPUS="2,4,6,7"
export SOURCE_STUDY_32B_EVAL_GPU="5"
export SOURCE_STUDY_32B_MERGE_GPU="6"
export SOURCE_STUDY_32B_TARGET_GLOBAL_BATCH="32"
export SOURCE_STUDY_32B_MICROBATCH="2"
```

Runner sẽ dùng 4 process và tự đặt gradient accumulation thành 4. Với 8 GPU,
gradient accumulation là 2. Nếu lựa chọn GPU không thể giữ đúng global batch,
runner dừng trước khi load model. Xem cấu hình đã resolve bằng:

```bash
bash scripts/run_qwen3_32b_scaled.sh show-config
```

Có thể dùng file riêng mà không sửa file mặc định:

```bash
export SOURCE_STUDY_32B_SETTINGS_FILE=/workspace/my-qwen32-settings.env.sh
```

## Trình tự chạy 30:70

```bash
cd /workspace/multilingual-safety-guard-runner-no-dataset

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

bash scripts/run_qwen3_32b_scaled.sh show-config
bash scripts/run_qwen3_32b_scaled.sh preflight
bash scripts/run_qwen3_32b_scaled.sh eval-base-smoke

mkdir -p logs/source-study-scaled/qwen3_32b/30_70
nohup bash scripts/run_qwen3_32b_scaled.sh eval-base \
  > logs/source-study-scaled/qwen3_32b/30_70/eval_base.nohup.log 2>&1 &
echo $! | tee logs/source-study-scaled/qwen3_32b/30_70/eval_base.pid
```

Chỉ sau khi base eval hoàn tất:

```bash
bash scripts/run_qwen3_32b_scaled.sh smoke-train

export SOURCE_STUDY_ALLOW_32B_TRAIN=1
nohup bash scripts/run_qwen3_32b_scaled.sh train \
  > logs/source-study-scaled/qwen3_32b/30_70/train.nohup.log 2>&1 &
echo $! | tee logs/source-study-scaled/qwen3_32b/30_70/train.pid
```

Sau khi train complete:

```bash
bash scripts/run_qwen3_32b_scaled.sh merge
bash scripts/run_qwen3_32b_scaled.sh eval-trained-smoke

nohup bash scripts/run_qwen3_32b_scaled.sh eval-trained \
  > logs/source-study-scaled/qwen3_32b/30_70/eval_trained.nohup.log 2>&1 &
echo $! | tee logs/source-study-scaled/qwen3_32b/30_70/eval_trained.pid
```

Để chạy comparator 70:30, cài ZIP 70:30 vào `work/source-study-scaled/nemotron70_wildguard30_scaled` và đổi `SOURCE_STUDY_SCALE_RATIO=70_30`. Runner không có phase tự động nối base-eval sang train; train 32B luôn cần cờ opt-in rõ ràng.
