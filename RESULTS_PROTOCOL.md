# Protocol danh gia theo paper Nemotron v3 (arXiv 2508.01710)

## Paper dung benchmark nao?
Paper Nemotron v3 KHONG dung XSTest. Cac benchmark chinh trong paper:

- CultureGuard (test split, 9 nguoi ngu) - prompt + response classification
- CultureGuard-JB (jail-break test, 8,883 mau)
- PGPrompts (PolyGuardPrompts)
- Cac benchmark phu: RTP-LX, MultiJail, XSafety, Aya Red-teaming

Metric chinh: harmful-F1 (F1 cua lop unsafe / harmful), bao trung binh
(average harmful-F1) tren cac nguoi ngu, tach rieng Prompt (P) va Response (PR).

## XSafety vs XSTest - dung nham
- XSafety (Jarviswang94/Multilingual_safety_benchmark): 14 loai, CHI chua
  unsafe prompts (recall diagnostic). Vi unsafe-only, balanced_accuracy/macro_f1
  la NaN (thieu lop safe). Day la hanh vi dung, KHONG phai bug.
  Chi nen doc harmful_f1 / unsafe_recall / unsafe_f1.
- XSTest (walledai/XSTest): 250 safe + 200 unsafe, do exaggerated safety.
  Khong nam trong paper Nemotron v3. Khong can them.

## Cac sua da ap dung (de so sanh cong bang voi NVIDIA)
1. Epochs: config mac dinh 1 epoch (de chay nhanh). Paper dung 5 epochs.
   Muon sat paper, doi `training.lora.epochs: 5` roi resume tiep tu checkpoint.
2. CultureGuard tach 2 benchmark rieng:
   - cultureguard_standard: tag != "jailbreaking"
   - cultureguard_jb: tag == "jailbreaking"
   prepare_training_data.py gio GIU field `tag` (generic/jailbreaking/adapted).
3. XSafety languages: mac dinh 7 nguoi ngu (en ar de fr hi ja zh) khop paper
   Table 15 (harmful-F1). Luu y Table 16 (harmful-recall) dung 9 nguoi ngu
   (them es, th).
4. Metric aggregation: `run.py nvidia-report` tinh average harmful-F1 per-language
   (khong pool), tach P/PR, giong paper.

## Lenh tao report NVIDIA-compatible
Sau khi chay evaluate, tao report:
```bash
python run.py nvidia-report --model <model> --checkpoint before
python run.py nvidia-report --model <model> --checkpoint after --method lora --run-mode full
```
Report nam tai runs/<model>/<checkpoint>/guard/nvidia_report.json.

So sanh voi so NVIDIA (Llama-3.1-Nemotron-Safety-Guard-8B-v3):
- CultureGuard standard P ~ 85.15, PR ~ 85.48
- CultureGuard-JB P ~ 91.77, PR ~ 94.35
- XSafety harmful-F1 ~ 66.97, harmful-recall ~ 53.27

## Luu y ve Nemotron model
Llama-3.1-Nemotron-Safety-Guard-8B-v3 tren may cong ty la base Llama-3.1-8B-Instruct
+ lora_adapter/. Phai merge adapter truoc khi eval vLLM:
```bash
python merge_adapter.py --base-model <LLAMA31_BASE> --adapter <NEMOTRON_DIR>/lora_adapter --output runs/llama31_nemotron/merged
```