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
`multilingual-safety-guard-runner` Phase-2 experiment. The current policy-correct
archive is `stage2_gemini_policy_training_ready.zip`.

The policy-correct archive contains 233,752 train and 8,367 validation rows
(242,119 rendered rows total), using V3 semantic replay, Gemini Vietnamese
translation, Nemotron Content Safety Reasoning, text-only synthetic Nemotron
3.5, and WildGuardTrain. It contains no model weights, benchmark test data, or
images. The manifest reports `training_ready=true` and `full_ready=false`
because benchmark leakage inputs are not embedded.

Each semantic record is rendered once. V3/Gemini use deterministic 75/25
taxonomy ON/OFF where an N23 category exists; reasoning uses deterministic
50/50 THINK/NO-THINK assignment, not paired duplication; WildGuard is always
taxonomy-OFF. THINK targets use Qwen native thinking with a rationale followed
by `</think>` and the final JSON.

Upstream V3 and reasoning data are attributed to NVIDIA under CC-BY-4.0.
The Gemini Vietnamese files are derivative translations and retain upstream
IDs/labels. Do not treat this archive as an evaluation benchmark. The
WildGuard-containing archive is uploaded only under the user's granted
access/permission and must not be redistributed outside that scope.
