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

This repository contains a ZIP archive of rendered training data for the
`multilingual-safety-guard-runner` Phase-2 experiment. The current archive is
`stage2_gemini_v2_no_wildguard_training_ready.zip`.

The archive contains 272,634 train and 13,609 validation rows (286,243
rendered rows total), using V3 semantic replay, Gemini Vietnamese translation,
Nemotron Content Safety Reasoning content-safety rows, and text-only synthetic
Nemotron 3.5 rows. It contains no model weights, benchmark test data, images,
or WildGuardTrain. The manifest therefore reports `training_ready=true` but
`full_ready=false` until the benchmark leakage inputs and gated WildGuardTrain
are handled separately.

Upstream V3 and reasoning data are attributed to NVIDIA under CC-BY-4.0.
The Gemini Vietnamese files are derivative translations and retain upstream
IDs/labels. Do not treat this archive as an evaluation benchmark or redistribute
gated WildGuard material without accepting its access terms.
