# Qwen3-4B fixed-80k Nemotron/WildGuard mixtures

## Research question

The completed natural-source study already supplies the two endpoints:

- 100% Nemotron / 0% WildGuard;
- 0% Nemotron / 100% WildGuard.

This follow-up adds three fixed-row-budget interpolation points while holding the
base model, output contract, optimizer, training steps and validation size fixed.

| Arm | Nemotron train | WildGuard train | Total train | Validation |
| --- | ---: | ---: | ---: | ---: |
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

## Controlled training recipe

- local offline Qwen3-4B base;
- taxonomy off and no-think Nemotron binary JSON contract for both sources;
- LoRA rank 8, alpha 32, dropout 0.05, `q_proj` and `v_proj`;
- BF16, SDPA, gradient checkpointing, maximum length 2,048;
- one epoch, constant learning rate `1e-5`, no warmup;
- per-device batch 1, gradient accumulation 32;
- one arm on each of GPUs 0, 1 and 2; effective batch 32 per arm;
- 2,500 optimizer updates and one final-epoch checkpoint per arm;
- seed 3407.

The fourth A30 is deliberately left free for a later orthogonal arm instead of
spending it on a duplicate seed. A WildGuard-majority 30/70 point can be added
only if the 50/50 result indicates the optimum lies toward the WildGuard endpoint.

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
existing checkpoint is resumed. The runner refuses to launch if GPUs 0-2 are
already using more than 2 GiB or if another mixture runner owns the lock.
