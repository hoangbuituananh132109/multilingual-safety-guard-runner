# Protocol đánh giá theo paper Nemotron v3 (arXiv 2508.01710)

## Paper dùng benchmark gì?
Paper Nemotron v3 KHONG dung XSTest. Cac benchmark chinh trong paper:

- CultureGuard (test split, 9 nguoi ngu) - prompt + response classification
- CultureGuard-JB (jail-break test, 8,883 mau)
- PGPrompts (PolyGuardPrompts)
- Cac benchmark phu: RTP-LX, MultiJail, XSafety, Aya Red-teaming

Metric chinh: harmful-F1 (F1 cua lop unsafe / harmful), bao trung binh
(average harmful-F1) tren cac nguoi ngu.

## XSafety vs XSTest - dung nham
- XSafety (Jarviswang94/Multilingual_safety_benchmark): 14 loai, 10 nguoi
  ngu, CHI chua unsafe prompts (recall diagnostic). Dung nhu code hien tai
  (prepare_benchmarks.py gan safety_label: unsafe cho moi dong).
  Vi unsafe-only, balanced_accuracy/macro_f1 la NaN (thieu lop safe).
  Day la hanh vi dung, KHONG phai bug. Chi nen doc unsafe_recall/unsafe_f1.
- XSTest (walledai/XSTest): 250 safe + 200 unsafe, do exaggerated safety.
  Khong nam trong paper Nemotron v3. Khong can them.

## So sanh cong bang voi paper
De so sanh model tu train voi Nemotron v3 theo dung paper, can:
1. Cung prompt template Nemotron (da khop 100% model card - verified).
2. Cung taxonomy 23 loai (da khop - verified).
3. Danh gia tren CultureGuard test (9 nguoi ngu) voi metric harmful-F1.
4. Nemotron v3 that = merge lora_adapter/ vao Llama-3.1-8B-Instruct.

## Luu y quan trong ve Nemotron model
Llama-3.1-Nemotron-Safety-Guard-8B-v3 tren may cong ty la base Llama-3.1-8B-Instruct
+ lora_adapter/. Phai merge adapter truoc khi eval (vLLM khong load PeftModel).