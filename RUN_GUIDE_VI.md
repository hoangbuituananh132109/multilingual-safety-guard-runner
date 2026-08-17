# Quy trình chạy đầy đủ — Multilingual Safety Guard Runner

> Hướng dẫn từng bước: đặt file, giải nén, đo BPB/PPL, đánh giá, chọn model, train, đánh giá lại.
> Mọi lệnh chạy trong thư mục `multilingual-safety-guard-runner/` (nơi chứa `run.py`).

---

## 0. Cấu trúc thư mục đích

```
multilingual-safety-guard-runner/
├── run.py
├── status.py
├── compare_models.py
├── unpack_zips.py
├── config.yaml
├── requirements.txt
├── core/
├── zip/          <- ĐẶT CÁC ZIP DATA + MODEL VÀO ĐÂY
├── input/        <- (tự tạo bởi `run.py unpack`)
├── work/         <- (tự tạo: data đã prepare, benchmark, manifest)
└── runs/         <- (tự tạo: kết quả evaluate/likelihood/train)
```

---

## 1. Cài môi trường (làm 1 lần)

Yêu cầu: Python 3.12, GPU NVIDIA (B200), CUDA runtime + PyTorch có BF16.

```bash
# Tạo venv (nếu chưa có)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài package (đúng version trong requirements.txt)
pip install -r requirements.txt
```

> Nếu dùng Docker công ty, image phải có sẵn các package trong `requirements.txt`
> (torch, transformers, datasets, peft, accelerate, bitsandbytes, safetensors,
> huggingface-hub, scikit-learn, pyyaml). Không cần `trl` cho SFT này.

---

## 2. Đặt các file ZIP vào `zip/`

Đặt đúng tên file (hoặc tên khớp pattern trong `unpack_zips.py`):

| Loại | Tên file phải đặt | Giải nén vào |
|---|---|---|
| Data Nemotron 9 ngôn ngữ | `nemotron_9lang.zip` | `input/nemotron/` |
| SEA-HELM | `sea-helm.zip` | `input/sea-helm/` |
| XSafety | `xsafety.zip` | `input/xsafety/` |
| Qwen3-0.6B | `qwen3-0.6b.zip` | `input/models/qwen3-0.6b/` |
| Qwen3-1.7B | `qwen3-1.7b.zip` | `input/models/qwen3-1.7b/` |
| Qwen3-4B | `qwen3-4b.zip` | `input/models/qwen3-4b/` |
| Qwen3-8B | `qwen3-8b.zip` | `input/models/qwen3-8b/` |
| Llama-3.1-8B-Instruct | `llama31-8b-instruct.zip` | `input/models/llama31-8b-instruct/` |

> Nguồn data/model và revision đã pin trong `DOWNLOAD_LINKS.txt` và `config.yaml`.
> Llama 3.1 là model gated — chỉ đặt ZIP nếu tổ chức đã được cấp quyền.

---

## 2b. Dùng model có sẵn trên máy công ty (không tải lại, không commit path)

Nếu Qwen/Llama đã có sẵn ở một thư mục khác trên máy công ty, bạn **không cần**
đặt ZIP model vào `zip/` và cũng **không nên** sửa `config.yaml` (path đó sẽ bị
commit lên git). Thay vào đó, đặt **biến môi trường** trước khi chạy lệnh:

```
# Tên biến: MODEL_PATH_<TÊN_MODEL_IN_HOA>
#   qwen3_0_6b  -> MODEL_PATH_QWEN3_0_6B
#   qwen3_1_7b  -> MODEL_PATH_QWEN3_1_7B
#   qwen3_4b    -> MODEL_PATH_QWEN3_4B
#   qwen3_8b    -> MODEL_PATH_QWEN3_8B
#   llama31_8b  -> MODEL_PATH_LLAMA31_8B
```

Windows PowerShell:

```powershell
$env:MODEL_PATH_QWEN3_8B = "D:\models\qwen3-8b"
$env:MODEL_PATH_LLAMA31_8B = "D:\models\llama-3.1-8b-instruct"
```

Linux/macOS:

```bash
export MODEL_PATH_QWEN3_8B=/data/models/qwen3-8b
export MODEL_PATH_LLAMA31_8B=/data/models/llama-3.1-8b-instruct
```

> - Giá trị phải là **thư mục chứa model** (có `config.json`, `model.safetensors`...).
> - Khi biến được đặt, runner dùng đúng thư mục đó cho `evaluate`, `likelihood`
>   và `train`; không cần ZIP model.
> - Nếu không đặt biến, runner vẫn dùng `input/models/<model>` như bình thường.
> - Data (Nemotron/SEA/XSafety) vẫn cần đặt ZIP trong `zip/` như §2.

## 3. Giải nén + kiểm tra trạng thái

```bash
# Giải nén tất cả ZIP vào input/, chặn zip-slip, kiểm tra marker
python run.py unpack

# Xem trạng thái (GPU, package, ZIP/model thiếu, data đã prepare)
python status.py

# Kiểm tra GPU + BF16 thật
python run.py preflight
```

> `preflight` phải báo `GPU ready: True`, `bf16: true`, `bf16_matmul: ok`.
> Nếu thiếu package/CUDA, dừng và gửi `work/status_report.txt` cho hạ tầng.

---

## 4. Chuẩn bị data + benchmark

```bash
# Test nhanh 20 mẫu trước
python run.py prepare --limit 20

# Khi thành công, chạy toàn bộ
python run.py prepare
```

Kết quả:
```
work/training/{train,valid,test}.jsonl
work/training/manifest.json
work/sea/sea_safeguard_vi.jsonl
work/benchmarks/{cultureguard_test_9lang,sea_safeguard_vi,xsafety_multilingual}.jsonl
work/benchmarks/benchmark_manifest.json
```

---

## 5. Đo BPB/PPL trên SEA-Bench tiếng Việt (trước train)

Chạy cho từng model đã tải. Ví dụ cả 5 model:

```bash
python run.py likelihood --model qwen3_0_6b --checkpoint before
python run.py likelihood --model qwen3_1_7b --checkpoint before
python run.py likelihood --model qwen3_4b  --checkpoint before
python run.py likelihood --model qwen3_8b  --checkpoint before
python run.py likelihood --model llama31_8b --checkpoint before
```

> Đo NLL, perplexity, bits-per-byte trên `sea_vi` (tiếng Việt).
> Kết quả: `runs/<model>/base/likelihood/likelihood_metrics.json`.
> **BPB thấp hơn = tốt hơn** khi so model khác tokenizer.

---

## 6. Đánh giá guard trên 3 benchmark (trước train)

Chạy cho từng model:

```bash
python run.py evaluate --model qwen3_0_6b --checkpoint before
python run.py evaluate --model qwen3_1_7b --checkpoint before
python run.py evaluate --model qwen3_4b  --checkpoint before
python run.py evaluate --model qwen3_8b  --checkpoint before
python run.py evaluate --model llama31_8b --checkpoint before
```

> Đo accuracy, balanced accuracy, macro-F1, unsafe precision/recall/F1, parse rate
> trên **3 benchmark**: `cultureguard`, `xsafety`, `sea_vi`.
> Kết quả: `runs/<model>/base/guard/metrics.json`.

---

## 7. So sánh và chọn model

```bash
python run.py compare --checkpoint before
```

Kết quả: `runs/model_comparison_before.csv` + `.json`.

> - **Tiêu chí chính:** `guard_rank` = macro-F1 cao, rồi unsafe recall cao (trên SEA-VI).
> - **Diagnostic phụ:** `bpb_rank` = bits-per-byte thấp (khả năng tiếng Việt).
> - Không chọn model chỉ bằng BPB/PPL.

> Lưu ý: bảng luôn liệt kê **tất cả model trong `config.yaml`**; model chưa chạy
> hiện ô trống/null. Nếu muốn bảng chỉ gồm model đã chạy, bỏ model chưa chạy khỏi
> `config.yaml` rồi chạy lại `compare`.

---

## 8. Train model đã chọn (LoRA, recipe paper Nemotron)

Thay `qwen3_8b` bằng model bạn chọn.

```bash
# Smoke 3 step (kiểm tra pipeline)
python run.py train --model qwen3_8b --method lora --mode smoke

# Pilot 100 step (kiểm tra ổn định)
python run.py train --model qwen3_8b --method lora --mode pilot

# Full 5 epoch (chạy thật)
python run.py train --model qwen3_8b --method lora --mode full
```

> Recipe (trong `config.yaml`): LoRA r=8, alpha=32, dropout=0.0, q/v proj,
> 5 epochs, LR 1e-5 constant, batch 4, gradient accumulation 8 (effective 32),
> max_length 2048, BF16, seed 3407.
>
> Muốn sát paper Nemotron, đổi `lora_dropout` 0.0 → 0.05 trong `config.yaml`.
>
> Resume nếu bị gián đoạn:
> ```bash
> python run.py train --model qwen3_8b --method lora --mode full --resume
> ```

---

## 9. Đánh giá lại sau train (3 benchmark + BPB/PPL)

```bash
# Guard trên 3 benchmark
python run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full

# BPB/PPL trên SEA-VI
python run.py likelihood --model qwen3_8b --checkpoint after --method lora --run-mode full

# So sánh before/after
python run.py compare --checkpoint after --method lora --run-mode full
```

Kết quả:
```
runs/qwen3_8b/lora_full/guard/metrics.json
runs/qwen3_8b/lora_full/likelihood/likelihood_metrics.json
runs/model_comparison_after.csv
```

---

## 10. Tóm tắt thứ tự lệnh

```bash
# 1. Setup + unpack
python run.py unpack
python status.py
python run.py preflight

# 2. Prepare data
python run.py prepare --limit 20
python run.py prepare

# 3. Before train: likelihood + evaluate cho từng model
python run.py likelihood --model qwen3_0_6b --checkpoint before
python run.py likelihood --model qwen3_1_7b --checkpoint before
python run.py likelihood --model qwen3_4b  --checkpoint before
python run.py likelihood --model qwen3_8b  --checkpoint before
python run.py likelihood --model llama31_8b --checkpoint before
python run.py evaluate --model qwen3_0_6b --checkpoint before
python run.py evaluate --model qwen3_1_7b --checkpoint before
python run.py evaluate --model qwen3_4b  --checkpoint before
python run.py evaluate --model qwen3_8b  --checkpoint before
python run.py evaluate --model llama31_8b --checkpoint before

# 4. Compare + chọn model
python run.py compare --checkpoint before

# 5. Train model chọn (vd qwen3_8b)
python run.py train --model qwen3_8b --method lora --mode smoke
python run.py train --model qwen3_8b --method lora --mode pilot
python run.py train --model qwen3_8b --method lora --mode full

# 6. After train: đánh giá lại
python run.py evaluate --model qwen3_8b --checkpoint after --method lora --run-mode full
python run.py likelihood --model qwen3_8b --checkpoint after --method lora --run-mode full
python run.py compare --checkpoint after --method lora --run-mode full
```

---

## 11. Khi có lỗi

Gửi về: traceback text, `work/status_report.txt`, `work/unpack_manifest.json`,
manifest prepare, `runs/.../train_config.yaml`, `trainable_parameters.json`,
`guard/run_manifest.json`, và đầu ra `nvidia-smi`.

Không gửi model weights, dataset, token, credential hoặc cả thư mục run.
