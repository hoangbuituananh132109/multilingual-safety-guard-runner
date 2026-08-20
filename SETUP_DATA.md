## 5. Xac nhan xu ly dung
Sau prepare, kiem tra manifest (co so dong + sha256):
```bash
cat work/benchmarks/benchmark_manifest.json | python3 -m json.tool
```
Va so dong tung benchmark:
```bash
wc -l work/benchmarks/*.jsonl
```

Luu y: moi sample Nemotron duoc tach thanh 2 dong (P = prompt, PR = prompt+response),
nen so dong trong file lon gap ~2 so sample. De doi chieu voi paper:
- CultureGuard standard: ~17,676 sample (paper)
- CultureGuard-JB: ~8,883 sample (paper)
- XSafety: chi unsafe prompts, 7 nguoi ngu (paper Table 15)
- SEA-VI: ~1,840 dong

Neu so dong lech bat thuong (vi du standard + jb khong xap xi 26,559 sample),
kiem tra lai zip/data truoc khi eval.

## 6. Chay eval toan bo
```bash
bash eval_all.sh
```