# Evaluator benchmark qua model host

`scripts/evaluate_hosted_benchmark.py` không tải model và không khởi động vLLM. Script chỉ:

1. giải nén ZIP cục bộ;
2. kiểm tra từng dòng có đúng `id`, `prompt`, `ground_truth`;
3. gửi prompt tới endpoint HTTP do người dùng điền;
4. ghi predictions và các metric theo benchmark.

## Smoke, không gọi mạng

```bash
cd /workspace/multilingual-safety-guard-runner-no-dataset
python3 scripts/evaluate_hosted_benchmark.py \
  --zip zip/qwen3_safety_benchmark_total_v2.zip \
  --extract-dir work/eval-hosted/benchmark-total \
  --output-dir runs/hosted-benchmark-smoke \
  --limit-per-benchmark 1 \
  --dry-run
```

Smoke tạo `dry_run.json` và luôn ghi `"network_called": false`.

## Cấu hình Qwen3-235B đã host

Sửa một file duy nhất: `config/hosted_235b_eval.env.sh`. Có thể điền URL base
`http://host:port/v1` hoặc URL đầy đủ `/v1/chat/completions`; evaluator tự chuẩn hóa.

```bash
cd /workspace/multilingual-safety-guard-runner-no-dataset
vi config/hosted_235b_eval.env.sh

bash scripts/run_hosted_benchmark_eval.sh show-config
bash scripts/run_hosted_benchmark_eval.sh dry-run
bash scripts/run_hosted_benchmark_eval.sh smoke
```

`dry-run` chỉ giải nén/validate ZIP và tuyệt đối không gọi mạng. `smoke` gọi hai
mẫu mỗi benchmark. Mặc định client gửi tối đa 8 request đồng thời; chỉnh
`HOSTED_EVAL_CONCURRENCY` theo sức chịu của server.

## Full eval qua endpoint đã host

```bash
source config/hosted_235b_eval.env.sh
mkdir -p logs
nohup python3 scripts/evaluate_hosted_benchmark.py \
  --zip zip/qwen3_safety_benchmark_total_v2.zip \
  --extract-dir work/eval-hosted/benchmark-total \
  --output-dir runs/hosted-benchmark-total \
  --endpoint "$HOSTED_MODEL_ENDPOINT" \
  --model "$HOSTED_MODEL_NAME" \
  --concurrency "$HOSTED_EVAL_CONCURRENCY" \
  --max-tokens 16 \
  --timeout 120 \
  --retries 2 \
  > logs/hosted-benchmark-total.nohup.log 2>&1 &
echo $! > logs/hosted-benchmark-total.pid
```

Hoặc dùng runner để không lặp tham số:

```bash
nohup bash scripts/run_hosted_benchmark_eval.sh full \
  > logs/hosted-qwen3-235b.nohup.log 2>&1 &
echo $! > logs/hosted-qwen3-235b.pid
```

Nếu endpoint không cần trường `model`, bỏ `--model`. Không có giá trị mặc định nào gọi ra mạng; chạy thật chỉ xảy ra khi có endpoint và không dùng `--dry-run`.

Kết quả gồm `predictions.jsonl`, `metrics.json`, `metrics.csv` và `run_manifest.json`. Metric có `Balanced Accuracy`, `Safe Recall`, `Unsafe Recall`, `Safe F1`, `Unsafe F1`, `Macro F1`, `Parse Rate` theo benchmark/ngôn ngữ/view và dòng `ALL`. Parse/request error được tính là dự đoán sai cho cả ground-truth safe lẫn unsafe.
