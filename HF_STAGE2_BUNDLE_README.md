---
license: cc-by-4.0
task_categories:
- text-generation
- text-classification
language:
- en
- vi
tags:
- content-safety
- safety-guard
---

# Stage-2 rendered safety training bundle

This repository contains ZIP archives of rendered training data for the
`multilingual-safety-guard-runner` Phase-2 experiment.

## Replacement schema-v3 bundles (use these)

These three archives were rebuilt on 2026-09-07 after fixing the taxonomy
prompt/label shortcut.  They also quarantine every exact-content group that
crosses train/validation or has contradictory labels, and remove configured
benchmark overlap.  Each embedded manifest has `training_ready=true`,
`full_ready=true`, empty blockers, and the exact post-filter counts/hashes.

| Archive | Train | Validation | ZIP SHA-256 |
|---|---:|---:|---|
| `stage2_corrected_vi_gemini_1e.zip` | 50,569 | 3,148 | `5fa471c13f4a3b36bff609e973b5486dca38c10ea8192ae099b24226ccca190a` |
| `stage2_corrected_reasoning_1e.zip` | 35,034 | 343 | `c519722747385f69c6d83f4c0f221512b7edf2dd373b9ab4b1e929bb4776b58d` |
| `stage2_corrected_full_1e.zip` | 230,419 | 7,467 | `b1d0a8de3442e5b4bef094a51e2b63a4687b707def198439ce069ff66efec032` |

`corrected_full` contains 55,801 V3 replay, 53,713 Gemini-VI, 34,726
reasoning, 9,323 selected text-only Nemotron 3.5, and 84,323 WildGuardTrain
rows after quarantine.  It contains 126,179 taxonomy-ON and 111,707
taxonomy-OFF rows; 12,620 rows retain teacher reasoning in THINK mode.

## Deprecated schema-v2 bundles (reproduction only)

`stage2_gemini_policy_training_ready.zip` and its two ablation archives use the
schema-v2 rendering policy. They are retained only to reproduce the already
launched exploratory run. Do not use them for a new training run or causal
comparison.

The deprecated archive contains 233,752 train and 8,367 validation rows
(242,119 rendered rows total), using V3 semantic replay, Gemini Vietnamese
translation, Nemotron Content Safety Reasoning, text-only synthetic Nemotron
3.5, and WildGuardTrain. It contains no model weights, benchmark test data, or
images. The manifest reports `training_ready=true` and `full_ready=false`
because benchmark leakage inputs are not embedded.

Known schema-v2 issues include taxonomy mode being confounded with safe/unsafe
labels, discarded prompt-only category annotations, incorrect V3 language
metadata, exact-content overlap across train/validation, conflicting labels
between sources, and smoke artifacts being able to mimic a completed run.

Uploaded ablation bundles:

- `stage2_ablation_vi_gemini.zip`: Gemini VI only, 50,637 train + 3,195 validation.
- `stage2_ablation_reasoning_only.zip`: Content Safety Reasoning only, 35,467 train + 437 validation.

Upstream V3 and reasoning data are attributed to NVIDIA under CC-BY-4.0.
The Gemini Vietnamese files are derivative translations and retain upstream
IDs/labels. Do not treat this archive as an evaluation benchmark. The
WildGuard-containing archive is uploaded only under the user's granted
access/permission and must not be redistributed outside that scope.
