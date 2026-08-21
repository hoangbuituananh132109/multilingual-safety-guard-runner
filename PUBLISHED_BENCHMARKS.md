# Optional Nemotron v3 paper benchmarks

The core suite always contains CultureGuard standard/JB, XSafety, and SEA-VI.
Two additional public paper benchmarks can be prepared offline:

- `ToxicityPrompts/PolyGuardPrompts`: balanced prompt and response classification, 9 paper languages.
- `ToxicityPrompts/DAMO-MultiJail`: harmful jailbreak prompts, 4 overlapping paper languages.

## Download on an internet-connected machine

```bash
hf download ToxicityPrompts/PolyGuardPrompts --repo-type dataset --revision c5b466a95b64ff121db4398246b6abb7672696ec --local-dir published_raw/polyguard_prompts
hf download ToxicityPrompts/DAMO-MultiJail --repo-type dataset --revision 69d16bdef53a0a061d7e56d2b6edb361df7f1507 --local-dir published_raw/multijail
tar -czf published_benchmarks.tar.gz published_raw
```

Copy `published_benchmarks.tar.gz` to the company machine, then:

```bash
tar -xzf published_benchmarks.tar.gz
python3 core/prepare_published_benchmarks.py --polyguard-root published_raw/polyguard_prompts --multijail-root published_raw/multijail --output-dir work/benchmarks
```

`run.py evaluate` automatically includes these files when they exist in
`work/benchmarks`.

## Smoke and full Qwen3-4B

```bash
export MODEL_PATH_QWEN3_4B=/workspace/storage-shared/models/Qwen3-4B
python3 run.py evaluate --model qwen3_4b --checkpoint before --backend vllm --sample 100
nohup python3 run.py evaluate --model qwen3_4b --checkpoint before --backend vllm > qwen3_4b_base_eval.log 2>&1 & echo $! > qwen3_4b_base_eval.pid
python3 run.py nvidia-report --model qwen3_4b --checkpoint before
```
