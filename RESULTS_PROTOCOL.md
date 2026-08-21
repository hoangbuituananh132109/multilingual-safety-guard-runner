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
  la NaN (thieu lop safe). Moi ngon ngu phai co 2,800 dong (14 x 200),
  bao gom `commonsense`. Ban code cu da loai nham category nay va chi tao 2,600.
- Paper van cong bo XSafety harmful-F1 (Table 15) va recall (Table 16), nhung
  public XSafety khong co lop safe nen standard binary F1 cua repo khong the
  tai tao truc tiep Table 15. Dung recall de doi chieu truc tiep; F1 cua repo
  chi dung de so sanh noi bo cung mot protocol.
- XSTest (walledai/XSTest): 250 safe + 200 unsafe, do exaggerated safety.
  Khong nam trong paper Nemotron v3. Khong can them.

## Cac sua da ap dung (de so sanh cong bang voi NVIDIA)
1. Epochs: config mac dinh 1 epoch (de chay nhanh). Paper dung 5 epochs.
   Muon sat paper, doi `training.lora.epochs: 5` roi resume tiep tu checkpoint.
2. CultureGuard tach 2 benchmark rieng:
   - cultureguard_standard: tag != "jailbreaking"
   - cultureguard_jb: tag == "jailbreaking"
   prepare_training_data.py gio GIU field `tag` (generic/jailbreaking/adapted).
3. XSafety languages: mac dinh 7 nguoi ngu (en ar de fr hi ja zh), khop ca
   Table 15 (harmful-F1) va Table 16 (harmful-recall).
4. Metric aggregation: `run.py nvidia-report` tinh average harmful-F1 per-language
   (khong pool), tach P/PR, giong paper.

## Lenh tao report paper-aligned
Sau khi chay evaluate, tao report:
```bash
python run.py nvidia-report --model <model> --checkpoint before
python run.py nvidia-report --model <model> --checkpoint after --method lora --run-mode full
```
Report nam tai runs/<model>/<checkpoint>/guard/nvidia_report.json.

So sanh voi so NVIDIA (Llama-3.1-Nemotron-Safety-Guard-8B-v3):
- CultureGuard standard P ~ 85.15, PR ~ 85.48
- CultureGuard-JB P ~ 91.77, PR ~ 94.35
- XSafety harmful-F1 ~ 66.97, harmful-recall ~ 53.27. Chi recall co the
  doi chieu truc tiep voi public unsafe-only data trong repo.

## Luu y ve Nemotron model
Thu muc goc `Llama-3.1-Nemotron-Safety-Guard-8B-v3` co full safetensors da
san sang inference. Official inference script load truc tiep thu muc goc.
Khong merge `lora_adapter/` len tren root weights, vi nhu vay co nguy co apply
LoRA hai lan. Chi dung `lora_adapter/` de merge vao mot ban
`Llama-3.1-8B-Instruct` sach neu khong co full root weights.

De A/B decoding voi quick-start cua model card:
```bash
python3 run.py evaluate --model llama31_nemotron --checkpoint before --backend vllm --decoding-profile nemotron_model_card --output-tag guard_model_card
python3 nvidia_report.py --metrics runs/llama31_nemotron/base/guard_model_card/metrics.json --out runs/llama31_nemotron/base/guard_model_card/nvidia_report.json
```
Mac dinh `greedy` duoc giu de so sanh cong bang voi cac run cu.
