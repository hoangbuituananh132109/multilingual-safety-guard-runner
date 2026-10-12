# Decision Guard Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an offline, CPU-tested training package for the approved seven-arm Qwen3-4B study without running training.

**Architecture:** Bridge imports local Qwen weights and applies CanonicalLoRA Q/V. A custom DDP loop optimizes completion-only LM loss or trained two-class heads. Shared SQLite token caches preserve fixed data and physical depth cuts avoid hidden full-depth execution.

**Tech Stack:** NeMo 26.08, Megatron Bridge 0.6 API, PyTorch, Transformers tokenizer, Python standard library, Bash.

**Spec:** `docs/superpowers/specs/2026-10-11-decision-guard-bridge.md`

## Global Constraints

- Fixed image `nvcr.io/nvidia/nemo:26.08`; no installs/downloads/fallback in runtime.
- No training or GPU rental during implementation; CPU synthetic tests only.
- Default seven arms; 1 epoch, global batch 32, BF16, Q/V LoRA r8/a32/.05, LR1e-5, seed3407, max length2048.
- Preserve unrelated dirty files; create only the new `decision_guard/` package and these two documents.

### Task 1: Data contracts and safe preparation

Files: `decision_guard/protocol.py`, `data.py`, `tests/test_protocol.py`, `tests/test_data.py`.
Interfaces: `canonical(row) -> dict`, `Encoder(tokenizer, max_length).encode(row, kind) -> dict`, `install(zip, output) -> dict`, `prepare(data, model, output, limits) -> dict`.

- [x] Write behavior tests for PR truth, label non-leakage, protected suffix/markers, completion-only next-token shift, identical content truncation, archive hash/idempotence, cache retrieval.
- [x] Run `python -m unittest discover -s decision_guard/tests -v`; confirm the new imports are missing.
- [x] Implement the contracts and streaming archive/cache preparation; run the tests again.

### Task 2: Heads, physical cuts and portable state

Files: `decision_guard/model.py`, `checkpoint.py`, `tests/test_model.py`, `tests/test_checkpoint.py`.
Interfaces: `BinaryHead`, `PointerHead`, `cut_backbone(model, depth)`, `MasterAdam`, `save_checkpoint`, `load_checkpoint`, `step_slots`.

- [x] Test last-real-token pooling under padding; pointer option positions and nonzero gradient; actual removal of layers; FP32 master tiny updates; state roundtrip and rejection of incompatible contracts; disjoint last-batch coverage.
- [x] Run the tests to establish missing implementation failure, then implement and rerun.

### Task 3: Bridge backend, DDP runner and user handoff

Files: `bridge_backend.py`, `train.py`, `preflight.py`, `settings.env.sh`, `run.sh`, `README.md`, `provenance.json`, `tests/test_runner.py`.
Interfaces: `build_model(config, kind, depth)`, `python -m decision_guard.train`, Bash phases `install|preflight|prepare|smoke|train|all`.

- [x] Test microbatch/global-batch validation, arm names, fixed hyperparameters, validation aggregation and smoke bounds before implementing the runner helpers.
- [x] Implement local-only Bridge import, CanonicalLoRA, DDP sum-loss normalization, checkpoint/resume and per-arm initialization.
- [x] Run all CPU tests, Python compilation, Bash syntax checks and pinned dataset audit. Package code separately from the 206MB dataset ZIP.
- [x] Mark image execution and GPU smoke as unverified; provide explicit copyable user-run commands, no claim that real training passed.

Review follow-up: a native CanonicalLoRA subset-QKV missing-K branch was identified
from upstream v0.6 source. Added a parameter-free zero-K shim, CPU regression and
installed-wrapper preflight probe. Preserve 247 legitimate empty PR responses;
strict smoke/receipt/cache gates and trainable-only DDP init synchronization added.
No GPU/model training was run during implementation.

## Follow-on authorized smoke (2026-10-11)

After the preparation-only handoff, the user authorized exactly one optimizer
step and checkpoint saving. Added red-green tests and one-step save/reload checks.
The Modal wrapper reuses the existing Qwen3-4B cache, stages source data/token
caches on CPU, and runs a single L40S with the same NeMo image, no ML installs.
`pointer36` completed step1, saved adapters/head/FP32 optimizer and passed
same-process roundtrip/forward checks. 29 package tests pass. Other arms,
multi-GPU, fresh-process resume and full training remain unverified.
