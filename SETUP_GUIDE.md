# Hướng dẫn cài môi trường & chạy train (multilingual-safety-guard-runner)

Tài liệu này dành cho máy GPU (H100/B200 công ty) để chạy:
- **Train LoRA SFT**: Qwen3-4B, Qwen3-8B, Llama-3.1-8B-Instruct (thành safety-guard theo phương pháp Nemotron).
- **Inference/compare**: Llama-3.1-Nemotron-Safety-Guard-8B-v3 (baseline, không train).

## 1. Yêu cầu
- Ubuntu 22.04+, Python 3.12+, NVIDIA GPU (H100/B200) có driver CUDA.
- Transformers 5.15.0 (bắt buộc để load Qwen3.5 arch). Code train đã fix tương thích 4.x/5.x.

## 2. Cài môi trường
```bash
bash setup_env.sh
```
Script tự:
- Tạo venv `.venv`, cài `requirements.txt`.
- Detect driver CUDA và cài torch bản khớp (`cu126` cho H100 driver 12.6, `cu130` cho B200/CUDA 13).
- Verify torch/transformers/peft.

> Nếu driver CUDA >= 13 (B200 công ty): script dùng `cu130`. Nếu H100 (12.6): dùng `cu126`.
> KHÔNG dùng torch `+cu130` trên driver 12.6 (lỗi CUDA driver không đủ).

## 3. Chuẩn bị dữ liệu
```bash
# Giải nén dữ liệu (nemotron, sea-helm, xsafety) — đặt zip trong ./zip
python run.py unpack --replace
# Chuẩn bị train/valid/test + benchmark
python run.py prepare
```

## 4. Chạy
```bash
# Đo likelihood (BPB/PPL) tiếng Việt
python run.py likelihood --model qwen3_4b --checkpoint before

# Eval zero-shot (stratified 1000 mẫu/benchmark)
python run.py evaluate --model qwen3_4b --checkpoint before --sample 1000

# Smoke train (kiểm tra lỗi, 3 steps)
python run.py train --model qwen3_4b --mode smoke

# Full LoRA train
python run.py train --model qwen3_4b --mode full

# Eval sau train
python run.py evaluate --model qwen3_4b --checkpoint after --method lora --run-mode full

# So sánh các model
python run.py compare --checkpoint before
```

## 5. Model train được / không train được
| Model | Train LoRA? | Ghi chú |
|---|---|---|
| Qwen3-4B | ✅ | causal LM |
| Qwen3-8B | ✅ | causal LM |
| Llama-3.1-8B-Instruct | ✅ | causal LM |
| Llama-3.1-Nemotron-v3 | ❌ (chỉ inference) | model đã train sẵn |
| Qwen3.5-4B | ❌ | multimodal, chỉ đo likelihood |
| Gemma-3-4b-it | ❌ | kém tiếng Việt, không hợp guard |

## 6. Lưu ý lỗi đã gặp
- **transformers 5.x**: `warmup_ratio`, `save_safetensors` không còn trong `TrainingArguments` -> code đã fix bằng filter động theo `inspect.signature`.
- **Qwen3.5-4B multimodal**: forward khác (cần pixel_values), LoRA q_proj/v_proj không map -> chỉ đo likelihood, không train.
- **PowerShell/SSH**: không dùng inline python có dấu nháy qua SSH; viết script file rồi scp.
- **Model cache**: đặt `HF_HOME` hoặc `MODEL_PATH_*` trỏ đúng để không tải lại.

## 7. Cấu hình train
- LoRA: `lora_r=8`, `lora_alpha=32`, `lora_dropout=0.05` (đã sửa từ 0.0).
- Scheduler: constant, warmup_ratio=0.0.
- Seed: 3407.
