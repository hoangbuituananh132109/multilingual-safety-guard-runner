# Evaluator benchmark qua model host

`scripts/evaluate_hosted_benchmark.py` không tải model và không khởi động vLLM. Script chỉ:

1. giải nén ZIP cục bộ;
2. kiểm tra từng dòng có đúng `id`, `prompt`, `ground_truth`;
3. gửi prompt tới endpoint HTTP do người dùng điền;
4. ghi predictions và các metric theo benchmark.

## Smoke, không gọi mạng

```bash
cd /workspace/multilingual-safety-guard-runner
python3 scripts/evaluate_hosted_benchmark.py \
  --zip zip/qwen3_safety_benchmark_total_v2.zip \
  --extract-dir work/eval-hosted/benchmark-total \
  --output-dir runs/hosted-benchmark-smoke \
  --limit-per-benchmark 1 \
  --dry-run
```

Smoke tạo `dry_run.json` và luôn ghi `"network_called": false`.

## Full eval qua endpoint đã host

Điền đúng địa chỉ model nội bộ vào biến dưới đây. Endpoint mặc định là OpenAI-compatible `/v1/chat/completions`.

```bash
cd /workspace/multilingual-safety-guard-runner
export HOSTED_MODEL_ENDPOINT="http://DIEN_DIA_CHI_MODEL:8000/v1/chat/completions"
export HOSTED_MODEL_API_KEY=""  # để trống nếu server nội bộ không cần key

nohup python3 scripts/evaluate_hosted_benchmark.py \
  --zip zip/qwen3_safety_benchmark_total_v2.zip \
  --extract-dir work/eval-hosted/benchmark-total \
  --output-dir runs/hosted-benchmark-total \
  --model "TEN_MODEL_NEU_SERVER_YEU_CAU" \
  --max-tokens 16 \
  --timeout 120 \
  --retries 2 \
  > logs/hosted-benchmark-total.nohup.log 2>&1 &
echo $! > logs/hosted-benchmark-total.pid
```

Nếu endpoint không cần trường `model`, bỏ `--model`. Không có giá trị mặc định nào gọi ra mạng; chạy thật chỉ xảy ra khi có endpoint và không dùng `--dry-run`.

Kết quả gồm `predictions.jsonl`, `metrics.json`, `metrics.csv` và `run_manifest.json`. Metric có `Balanced Accuracy`, `Safe Recall`, `Unsafe Recall`, `Safe F1`, `Unsafe F1`, `Macro F1`, `Parse Rate` theo benchmark/ngôn ngữ/view và dòng `ALL`.
