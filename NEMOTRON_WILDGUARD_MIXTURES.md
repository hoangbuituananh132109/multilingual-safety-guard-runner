# Qwen3-4B fixed-80k Nemotron/WildGuard mixtures

## Research question

The completed natural-source study already supplies the two endpoints:

- 100% Nemotron / 0% WildGuard;
- 0% Nemotron / 100% WildGuard.

This follow-up adds four fixed-row-budget interpolation points while holding the
base model, output contract, optimizer, training steps and validation size fixed.

| Arm | Nemotron train | WildGuard train | Total train | Validation |
| --- | ---: | ---: | ---: | ---: |
| `nemotron30_wildguard70` | 24,000 | 56,000 | 80,000 | 300 / 700 |
| `nemotron50_wildguard50` | 40,000 | 40,000 | 80,000 | 500 / 500 |
| `nemotron70_wildguard30` | 56,000 | 24,000 | 80,000 | 700 / 300 |
| `nemotron80_wildguard20` | 64,000 | 16,000 | 80,000 | 800 / 200 |

Ratios are row ratios. Complete semantic groups are selected in stable SHA-256
order, so no upstream ID or WildGuard prompt group is split to hit a quota.

## Cross-source cleaning

The two installed 80k natural arms are not concatenated directly. The builder
first detects exact-content overlap between the two sources across both train
and validation. Every complete semantic group touching one of those hashes is
removed from both source pools. This also removes label conflicts and prevents
cross-source train/validation leakage before ratio sampling.

The builder writes counts, source quotas, distributions, parent-manifest hashes,
split hashes and the cleaning audit to every output `manifest.json`.

In the current pinned inputs, 642 normalized exact contents occur in both
sources and 113 of them have conflicting labels. Cleaning removes 5,711
Nemotron training rows and 632 WildGuard training rows because complete groups,
not isolated rows, are the unit of removal.

The selected sources do have different distributions. Their Safe/Unsafe priors
happen to be similar, so that marginal changes little; source, language, content
and P/PR exposure change substantially:

| Train arm | Safe | Unsafe | P | PR | English |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30% Nemotron / 70% WildGuard | 49,042 | 30,958 | 42,709 | 37,291 | 58,816 |
| 50% Nemotron / 50% WildGuard | 48,998 | 31,002 | 42,020 | 37,980 | 44,664 |
| 70% Nemotron / 30% WildGuard | 48,874 | 31,126 | 40,801 | 39,199 | 30,492 |
| 80% Nemotron / 20% WildGuard | 48,919 | 31,081 | 40,104 | 39,896 | 23,417 |

These are row-budget mixtures, not token-budget mixtures. Token length remains
a measured source characteristic instead of being artificially equalized.

## Controlled training recipe

- local offline Qwen3-4B base;
- taxonomy off and no-think Nemotron binary JSON contract for both sources;
- LoRA rank 8, alpha 32, dropout 0.05, `q_proj` and `v_proj`;
- BF16, SDPA, gradient checkpointing, maximum length 2,048;
- one epoch, constant learning rate `1e-5`, no warmup;
- per-device batch 1, gradient accumulation 32;
- one arm on each of GPUs 0, 1, 2 and 3; effective batch 32 per arm;
- 2,500 optimizer updates and one final-epoch checkpoint per arm;
- seed 3407.

The GPU mapping is fixed by array order: GPU 0 runs 50/50, GPU 1 runs 70/30,
GPU 2 runs 80/20, and GPU 3 runs 30/70. All four arms train concurrently.

## Fully offline company-machine commands

The source data must already exist under `work/source-study-natural`, and the
model must already exist under `/workspace/storage-shared/models/Qwen3-4B`.
The runner forces offline library modes and does not download anything.

```bash
cd /workspace/multilingual-safety-guard-runner-no-dataset
export SOURCE_STUDY_MODEL_PATH=/workspace/storage-shared/models/Qwen3-4B

bash scripts/run_nemotron_wildguard_mixtures.sh prepare
bash scripts/run_nemotron_wildguard_mixtures.sh preflight
bash scripts/run_nemotron_wildguard_mixtures.sh smoke
```

Only after all three commands pass:

```bash
cd /workspace/multilingual-safety-guard-runner-no-dataset
mkdir -p logs/source-study-mixtures

nohup bash scripts/run_nemotron_wildguard_mixtures.sh train \
  > logs/source-study-mixtures/train_master.nohup.log 2>&1 &

echo $! | tee logs/source-study-mixtures/train_master.pid
sleep 5
ps -fp "$(cat logs/source-study-mixtures/train_master.pid)"
tail -n 120 logs/source-study-mixtures/train_master.nohup.log
```

The train phase is idempotent: a completed arm is skipped, and an arm with an
existing checkpoint is resumed. The runner refuses to launch if GPUs 0-3 are
already using more than 2 GiB or if another mixture runner owns the lock.
