# Thử nghiệm công bằng nguồn dữ liệu — Qwen3-4B

## Thiết kế đã khóa

Ba mô hình bắt đầu độc lập từ cùng `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Mỗi arm có 80.000 train và
1.000 validation, seed 3407, taxonomy off, Qwen no-think và cùng bốn cell:

| Cell | Train | Validation |
| --- | ---: | ---: |
| P safe | 21.501 | 269 |
| P unsafe | 23.102 | 289 |
| PR safe | 27.594 | 345 |
| PR unsafe | 7.803 | 97 |

Nemotron chia quota từng cell gần đều cho 9 ngôn ngữ `en/ar/de/es/fr/hi/ja/th/zh`.
WildGuard là tiếng Anh. SEA Cultural chỉ lấy `language=Vietnamese`,
`quality_assurance=Pass`; nhãn `Sensitive` được ánh xạ prompt→safe và
response→unsafe theo semantics benchmark SEA. Dedupe và loại train/eval overlap
được thực hiện trước khi stable sampling.

Recipe chung: LoRA r=8, alpha=32, dropout=0,05, chỉ `q_proj/v_proj`; BF16,
SDPA, max length 2.048; learning rate 1e-5 constant, không warmup, batch 1/GPU,
gradient accumulation 8. Trên 4 GPU global batch là 32 và một epoch đúng 2.500
optimizer steps. Checkpoint và eval được lưu cuối epoch, sau đó còn thư mục
`final/`.

## Chuẩn bị dữ liệu

Bundle đã chọn sẵn được cài bằng:

```bash
wget -O nemotron_v3_9lang_80k_v1.zip \
  https://huggingface.co/datasets/TuanAnhHoangBui/qwen3-4b-source-study-nemotron-v3-80k/resolve/main/nemotron_v3_9lang_80k_v1.zip
python3 scripts/source_study_bundle.py install \
  --zip /duong-dan/nemotron_v3_9lang_80k_v1.zip \
  --output work/source-study/nemotron_v3_9lang
```

WildGuardMix và SEA Cultural-v3 là nguồn gated; không dùng mirror public để né
bước chấp thuận. Nếu máy công ty có thể nhận một HF token đã được cấp quyền:

```bash
export HF_TOKEN='hf_...'
python3 scripts/download_source_study_sources.py
python3 core/prepare_source_study_benchmarks.py --output-dir work/benchmarks
python3 source_study.py build --source wildguardtrain_en \
  --source-path input/source-study/raw/wildguardtrain_en/train/wildguard_train.parquet
python3 source_study.py build --source sea_cultural_vi \
  --source-path input/source-study/raw/sea_cultural_vi/Vietnam
```

Builder WildGuard nhận cả JSONL chuẩn và parquet gốc. Không đẩy SEA Cultural-v3
công khai khi upstream chưa công bố license.

Nếu chuyển file nội bộ, ba archive đã tạo trên máy chuẩn bị là:

| Archive | SHA-256 | Trạng thái |
| --- | --- | --- |
| `nemotron_v3_9lang_80k_v1.zip` | `68624e52d632f0a98fe6df31896113571365383ddf7fa99b16f44803ae14168a` | Public HF |
| `wildguardtrain_en_80k_v1.gated.zip` | `a5c8fa494f4f331f2b4cea613117e060aeebf366e8f81492fec9fbc28cc93645` | Chỉ chuyển nội bộ sau khi được cấp quyền |
| `sea_cultural_vi_80k_v1.gated.zip` | `031eb97e74f4e8a9b2149e94ae42ee7fba8e5f920e5aebf4dde167bfd09b35ad` | Chỉ chuyển nội bộ; không public |

Cài hai archive còn lại tương tự bằng `scripts/source_study_bundle.py install`.
Bốn benchmark bổ sung có thể chuyển bằng `source_study_benchmarks_v2.gated.zip`
(SHA-256 `a4c47e11b4c241167a31a1b71e04bdc9a1612b5e5d2ddfe34fa2effe6bad667c`)
và cài an toàn vào thư mục benchmark đang có:

```bash
wget -O source_study_benchmarks_v2.gated.zip \
  https://huggingface.co/datasets/TuanAnhHoangBui/qwen3-4b-source-study-nemotron-v3-80k/resolve/main/source_study_benchmarks_v2.gated.zip
python3 scripts/source_study_benchmark_bundle.py install \
  --zip source_study_benchmarks_v2.gated.zip \
  --output work/benchmarks
```

VISafe dùng NVIDIA Sample Data License cấm phân phối lại, nên không nằm trong
bundle trên. Sau khi đã cài đủ ba arm train, tải trực tiếp từ NVIDIA và tạo
binary guard view bằng:

```bash
python3 core/prepare_visafe_benchmark.py \
  --source-study-root work/source-study \
  --output-dir work/benchmarks
```

## Chạy an toàn trên 4 A30

Chạy từng cổng kiểm tra trước:

```bash
bash scripts/run_source_study.sh preflight
bash scripts/run_source_study.sh smoke
```

Smoke GPU của mỗi arm chỉ chạy đúng 2 optimizer steps và không ghi checkpoint.
Khi cả ba smoke pass, chạy train + merge + eval có log và tiếp tục sống sau khi
ngắt SSH:

```bash
mkdir -p logs/source-study
nohup bash scripts/run_source_study.sh study \
  > logs/source-study/orchestrator.log 2>&1 &
echo $! > logs/source-study/orchestrator.pid
tail -f logs/source-study/orchestrator.log
```

`study` có skip/resume: train tuần tự ba arm trên cả bốn A30, merge, rồi eval
đồng thời base + ba model (mỗi model một GPU). `all` chạy thêm preflight và smoke
từ đầu. Đánh giá luôn dùng taxonomy off/no-think để chỉ thay đổi nguồn data.

Muốn smoke eval trước khi chạy full benchmark:

```bash
SOURCE_STUDY_EVAL_SAMPLE=8 bash scripts/run_source_study.sh eval-smoke
```

Muốn chỉ chạy từng phase:

```bash
bash scripts/run_source_study.sh train
bash scripts/run_source_study.sh merge
bash scripts/run_source_study.sh eval
```

## Benchmark

Giữ nguyên sáu benchmark cũ: CultureGuard standard/jailbreak, MultiJail,
PolyGuard, XSafety và SEA-VI. Bổ sung:

- XSTest (450 prompt tiếng Anh): đo over-refusal qua cặp safe/unsafe khó.
- WildGuardTest (3.408 tác vụ P/PR): test tiếng Anh có nhãn người.
- SEALSBench-VI (26.644 prompt): kiểm độ bền tiếng Việt dịch máy; đây chỉ là
  diagnostic phụ, không thay SEA-VI vốn có ngữ cảnh văn hóa bản địa.
- LinguaSafe-VI (3.884 prompt): benchmark đa ngôn ngữ được NVIDIA dùng trong
  đánh giá Nemotron 3.5 Content Safety. Binary view dùng `L0=safe`,
  `L1/L2/L3=unsafe`; severity gốc vẫn nằm trong metadata. `macro-F1` của runner
  cho phép so cùng các guard benchmark; `linguasafe_severity_weighted_f1` và
  `linguasafe_severity_weighted_fpr` dùng đúng trọng số paper với alpha=0,6.
  Phần indirect generation/over-refusal vẫn nằm ngoài track classifier này.
  Upstream có một prompt VI giống hệt xuất hiện ba lần với nhãn L0/L2 xung đột;
  giữ nguyên để trung thành với bản benchmark NVIDIA đã dùng.
- NVIDIA VISafe (3.212 prompt tiếng Việt): bộ probe chính xác dành cho
  evaluation/red-team, không đưa vào bất kỳ arm train nào. Guard view ánh xạ
  `refuse/warn_and_refuse=unsafe`, `allow/neutral_response=safe`. Báo cáo cả
  `visafe_vi` đầy đủ để đối chiếu nội bộ và `visafe_vi_clean` gồm 3.164 mẫu;
  clean slice loại 48 hàng có prompt VI hoặc prompt EN trùng chính xác với ít
  nhất một arm train. Protocol chính thức sinh response + LLM judge vẫn tách
  khỏi phép đo input-classifier này.

Sau eval, bảng gọn nằm ở
`runs-source-study/qwen3_4b/source_study_results.md` với balanced accuracy,
macro-F1, unsafe recall và parse rate.

## WildGuard tiếng Việt ghép cặp

`scripts/prepare_wildguard_vi_translation.py export` tạo đúng 81.000 UID từ arm
WildGuard EN. Pipeline Gemini của workspace dịch prompt/response nhưng giữ split,
nhãn, group ID và source hash. Năm shard phải có checkpoint riêng. Chỉ khi đủ
81.000 bản dịch và không lỗi mới materialize thành arm VI; đây là thí nghiệm kế
tiếp, không trộn vào ba arm chính.
