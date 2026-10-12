# Qwen3-4B binary36 / binary18: train rồi tự eval

## Phạm vi và bằng chứng

Chỉ chạy Qwen3-4B, data Nemotron:WildGuard **70:30 scale**, binary head.
Không chạy 8B, KEV/pointer, JSON hay bộ SFT cũ. Cố định image
`nvcr.io/nvidia/nemo:26.08`, backend Megatron Bridge + Transformer Engine.
Không cài package, không tải model/data từ CLI, không tự chuyển backend khi lỗi.

Code mới đã kiểm tra CPU và luồng Bash với các biên GPU được thay bằng fixture.
Đây **không phải** smoke GPU/B200. Bằng chứng GPU cũ chỉ là pointer36, 1 step,
1 L40S; không chứng minh binary36/binary18 hay 8 B200 chạy được. Vì vậy lệnh mới
có smoke và eval-smoke bắt buộc, lỗi thì dừng trước train thật.

## 1. Chuyển code mới vào thư mục đang dùng

### Tải ZIP trực tiếp từ GitHub (ưu tiên)

Tải bằng trình duyệt [ZIP nhánh b200-full-data-study](https://github.com/hoangbuituananh132109/multilingual-safety-guard-runner/archive/refs/heads/b200-full-data-study.zip),
upload thủ công bản ZIP **mới** vào thư mục `zip/` ở checkout cũ của máy công ty.
Không dùng `git pull`, `wget` hay tải mạng từ terminal máy công ty.
Thực hiện trước khi chạy binary; không thay code khi binary đang chạy.

```bash
OLD=/workspace/storage-shared/nlp/anhhbt/multilingual-safety-guard-runner-b200-full-data-study
ROOT=/workspace/storage-shared/nlp/anhhbt/decision-guard-study/multilingual-safety-guard-runner-b200-full-data-study
CODE_ZIP="$OLD/zip/multilingual-safety-guard-runner-b200-full-data-study.zip"
test -f "$CODE_ZIP" || { echo "ERROR: chưa upload ZIP GitHub mới: $CODE_ZIP"; exit 1; }
test -d "$ROOT/decision_guard" || { echo "ERROR: thiếu checkout đã chuẩn bị: $ROOT"; exit 1; }
STAGE=$(mktemp -d /workspace/storage-shared/nlp/anhhbt/decision-guard-study/code-binary-v2-XXXXXX)
python3 -m zipfile -e "$CODE_ZIP" "$STAGE" || exit 1
NEW="$STAGE/multilingual-safety-guard-runner-b200-full-data-study"
test -f "$NEW/decision_guard/run_binary_b200.sh" || { echo 'ERROR: ZIP cũ hoặc sai nhánh'; exit 1; }
cp -a "$NEW/decision_guard/." "$ROOT/decision_guard/"
cp -- "$NEW/docs/DECISION_GUARD_BINARY_B200_VI.md" "$ROOT/docs/"
cd "$ROOT"
bash -n decision_guard/run.sh
bash -n decision_guard/run_binary_b200.sh
```

Chỉ cập nhật folder binary và hướng dẫn mới, không thay code SFT đang chạy,
không động vào data/checkpoint/log. Thư mục giải nén tạm được giữ nguyên.

### Hoặc dùng ZIP patch cục bộ

ZIP patch chỉ chứa code/docs `decision_guard`, không chứa model/data. Chuyển
file `decision_guard_binary_b200_v2_20261012.zip` thủ công lên:

```text
/workspace/storage-shared/nlp/anhhbt/decision-guard-study/decision_guard_binary_b200_v2_20261012.zip
```

Không cần tải lại dataset/benchmark. Giải nén **một lần trước khi chạy binary**;
không sửa code trong lúc binary đang train vì checkpoint khóa fingerprint code.
Patch không thay `core/train.py` hay script SFT cũ đang chạy.

```bash
cd /workspace/storage-shared/nlp/anhhbt/decision-guard-study/multilingual-safety-guard-runner-b200-full-data-study
python3 -m zipfile -t ../decision_guard_binary_b200_v2_20261012.zip
python3 -m zipfile -e ../decision_guard_binary_b200_v2_20261012.zip .
bash -n decision_guard/run.sh
bash -n decision_guard/run_binary_b200.sh
```

Kiểm tra GPU không có job khác trước khi chạy. Các lệnh không kill tiến trình.

## 2. Full layer: 36 layer

Trong image NeMo đã chọn, trên node 8 GPU dành riêng cho thí nghiệm này:

```bash
cd /workspace/storage-shared/nlp/anhhbt/decision-guard-study/multilingual-safety-guard-runner-b200-full-data-study
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_SAVE_EVERY=2000 \
  bash decision_guard/run_binary_b200.sh binary36
```

## 3. Half layer: 18 layer

Chạy sau khi binary36 xong, hoặc trên **node khác** có 8 GPU riêng. Cùng storage
được, vì run root, cache, log và lock tách theo arm.

```bash
cd /workspace/storage-shared/nlp/anhhbt/decision-guard-study/multilingual-safety-guard-runner-b200-full-data-study
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_SAVE_EVERY=2000 \
  bash decision_guard/run_binary_b200.sh binary18
```

binary18 bắt đầu từ pretrained Qwen3-4B mới và bỏ vật lý 18 layer cuối, không
tiếp tục train từ binary36. Hai cấu hình giữ cùng data, prompt, seed, LoRA và
global batch. LoRA Q/V r8, alpha32, dropout0.05; backbone BF16, head và optimizer
FP32; 1 epoch, LR 1e-5 constant, global batch32, microbatch4/GPU với 8 GPU,
train max length2048. Đây vẫn là **LoRA**, không full-parameter fine-tuning.

## Mỗi lệnh thực sự làm gì?

1. Audit hash/schema benchmark tổng; kiểm tra và dùng lại data đã giải nén.
2. Preflight image/API/model, tạo receipt và cache token offline.
3. Smoke: train **1 optimizer step**, lưu checkpoint/optimizer/head/LoRA, nạp lại
   và kiểm tra forward. Khởi động torchrun lần hai để kiểm tra nạp checkpoint
   trong tiến trình mới; không train thêm step thứ hai.
4. Eval checkpoint smoke trên 8 mẫu đầu mỗi benchmark (96 mẫu tổng).
   Không dùng số đo nhỏ này để kết luận chất lượng.
5. Train mới trên toàn bộ **274.760 train**, validation1.000, **8.587 step**.
   Lưu step2.000/4.000/6.000/8.000/8.587, rồi `final.pt`.
6. Tự eval `final.pt` trên **129.761 mẫu / 12 benchmark** tổng, không so trùng
   hay xóa mẫu benchmark. Có BA, safe/unsafe recall và số safe/unsafe từng bộ.

Smoke dùng output riêng, không dùng trọng số smoke khởi tạo train thật.
Log vừa hiện terminal vừa ghi file, không `nohup`.

## Output và resume

Mặc định full checkpoint ở:

```text
runs-decision-guard/qwen3_4b_70_30_binary_v2/binary36/binary36/final.pt
runs-decision-guard/qwen3_4b_70_30_binary_v2/binary18/binary18/final.pt
```

Kết quả benchmark trong `eval_full/` ngay cạnh `final.pt`:
`metrics.json`, `metrics.csv`, `predictions.jsonl`, `run_manifest.json`.
Log ở `logs/decision-guard/binary-v2/binary36/` hoặc `binary18/`.

Nếu bị mất GPU, chạy lại **đúng lệnh cũ**, giữ nguyên số GPU, batch, code, input
và output. Train tự resume checkpoint hoàn chỉnh mới nhất (phần kể từ checkpoint
gần nhất chưa lưu phải chạy lại). Train xong rồi thì bỏ qua train. Eval giữ các
part đã lưu, tiếp tục mẫu chưa làm; eval hoàn chỉnh thì bỏ qua. Lệnh từ chối
checkpoint/output khác fingerprint, không overwrite một thí nghiệm khác.

Chỉ eval lại sau train, không chạy smoke/train:

```bash
bash decision_guard/run_binary_b200.sh binary36 eval
# hoặc
bash decision_guard/run_binary_b200.sh binary18 eval
```

Lấy bảng kết quả đã xong, không load GPU:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from decision_guard.evaluate import print_table
root = Path('runs-decision-guard/qwen3_4b_70_30_binary_v2')
for arm in ('binary36', 'binary18'):
    path = root / arm / arm / 'eval_full/metrics.json'
    if not path.is_file():
        print(f'{arm}: chưa có bảng eval hoàn chỉnh')
        continue
    print(f'\n===== {arm} =====')
    print_table(json.loads(path.read_text()))
PY
```

## Config và giới hạn so sánh

`decision_guard/settings.env.sh` là file config chính. `run_binary_b200.sh` đặt
mặc định riêng cho đợt binary v2. Export trước lệnh có ưu tiên: `DG_MODEL_PATH`,
`DG_DATA_ZIP`, `DG_DATA_DIR`, `DG_BENCHMARK`, `DG_RUN_ROOT`, `DG_CACHE_ROOT`,
`DG_LOG_ROOT`, `DG_MICRO_BATCH`, `DG_SAVE_EVERY`, `DG_EVAL_*`. Chỉ hỗ trợ model
Qwen3-4B đã khóa config; đổi đường dẫn không có nghĩa hỗ trợ Qwen3-8B.

Eval này **không dùng vLLM batch2000**. Nó dùng 8 bản sao model (DP), mặc định
batch tối đa8/GPU và padded-token budget32.768; câu dài chạy riêng. Max input
40.000 token, giữ head+tail và bảo vệ system prompt/câu hỏi/target. Dữ liệu tổng
chỉ có prompt đã serialize, không còn biên prompt/response đáng tin cậy; không
tách bằng chuỗi `assistant:` có thể xuất hiện trong nội dung. Với câu không
truncate, chat wrapper binary trùng training; câu dài giữ head+tail của toàn
conversation, ghi `truncated` để kiểm tra. Đây là khác biệt cần công bố so với
truncation training P/PR giữ ngân sách riêng cho hai phần.

Binary head đọc hidden của token quyết định cuối, softmax FP32 ra hai xác suất;
unsafe nếu P(unsafe)>0.5, bằng0.5 chọn safe. Không sinh token, không parse JSON.
Chưa calibrate ngưỡng. Benchmark chỉ có một lớp thì BA không xác định, hiển thị
`-`, không coi missing safe recall là0.

Khi so với SFT JSON cũ, phải ghi rõ prompt policy mới khác prompt cũ. Binary36
vs binary18 là so sánh matched trong nhánh này; số BA của các SFT cũ chỉ là tham
chiếu, chưa tách riêng ảnh hưởng prompt khỏi kiến trúc/head/depth.
