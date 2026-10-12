# Verification evidence — 2026-10-11

Scope: CPU preparation plus one real optimizer step/save/reload on Modal.
No full training, latency experiment or benchmark inference.

Verified locally:

- 29 `unittest` CPU synthetic tests pass, including red→green tests for empty PR,
  native missing-K compatibility, stale smoke rejection and bounded smoke flag.
- Python compilation passes for the new package.
- `bash -n` passes for `run.sh` and `settings.env.sh`.
- Full pinned data audit: 274760 train / 1000 validation; zero cross-split group
  and content overlap, authoritative row labels preserved, all four file hashes
  match the pinned archive. Detail: `data_audit.json`.
- Read-only independent code review: missing-K issue fixed; no remaining
  confirmed Critical/Important finding. Review is not image/GPU runtime proof.
- Code ZIP includes only this new package and two design documents. It excludes
  data, models, local token caches, preexisting dirty files and `__pycache__`.

Verified on one L40S, Modal profile pippa3, app `ap-Kp5iZuy9XPiW9Lewye8f0M`:

- Fixed base `nvcr.io/nvidia/nemo:26.08`, no ML package installation or backend fallback.
- Bridge 0.6.1 / MCore 0.19.1 / Transformer Engine 2.18.0+15288857; all API,
  local checkpoint hashes and real tokenizer contracts passed image preflight.
- `pointer36`, physically full 36-layer Qwen3-4B; native Q/V r8 LoRA plus new FP32
  KEV-style pointer head, 4,259,840 trainable parameters.
- Real source cache 128 train/16 validation, max length 2048, global batch32,
  microbatch1, accumulation32. Exactly one optimizer step: loss0.65797559,
  grad norm17.80846 before clipping, 12.95 seconds.
- Saved `checkpoint-000001.pt` and `final.pt`; final62,489,638 bytes,
  SHA256 `9b64470b26be7c239376977ae5330de8bdbc6c99f899e9f34277210881ea62c9`.
- Zeroed trainables/masters, cleared optimizer state, restored checkpoint and
  checked exact weights/masters/moments and equal forward output in the same process.
- Smoke+validation/save/reload16.12 seconds after initialization;
  full GPU function250.8 seconds including preflight/import/load. Job completed.
- CPU-only setup initially failed because TE requires libcuda; GPU preflight
  performs that check. Existing HF cache symlinks were materialized on CPU into
  a separate study folder, preserving the original model cache and strict shard checks.
- Optional Triton/torchao import warnings appeared upstream; dense Qwen/TE
  preflight, backward and checkpoint verification nevertheless completed successfully.

Evidence: `modal-smoke-result.json`, `modal-pointer36/smoke_pass.json`,
`modal-pointer36/train.log`; checkpoint retained in the private user Modal volume
`guard-bridge-smoke-20261011` and downloaded locally. It is excluded from code ZIPs.

Not verified: binary/JSON/other-depth arms, multi-GPU training, fresh-process
checkpoint resume, full training, held-out benchmark accuracy or trained latency.

The small CPU test double is not presented as official Bridge runtime evidence.
Stop at a preflight/smoke failure and inspect logs before full training.
