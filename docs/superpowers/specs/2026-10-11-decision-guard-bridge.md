# Decision guard: approved first training study

Approved by the user on 2026-10-11. Prepare code and CPU checks only; do not train or rent GPU.

- Fixed image: `nvcr.io/nvidia/nemo:26.08`; no runtime dependency installs, downloads, or backend fallback.
- Local Qwen3-4B, initialized from its pretrained HF weights through Megatron Bridge. Native CanonicalLoRA Q/V, rank 8, alpha 32, dropout .05. BF16 compute, FP32 optimizer master weights. 1 epoch, LR 1e-5, global batch 32, max length 2048, seed 3407.
- Seven default arms: full 36-layer JSON generation; binary 36/27/18; KEV-style pointer 36/27/18. The pointer is a new two-option Qwen3 head, NOT the official pretrained KEV or OpenJEV model. An optional independently trained label-generation control is permitted, but is not a default arm.
- Physically retain the first N decoder layers, keeping the original final norm. No full-depth forward with intermediate readout. Fresh pretrained initialization for every arm; do not train the arms successively on one model.
- Genuine Bridge model import and CanonicalLoRA, with a custom PyTorch DDP loop for the new heads. Data parallel only, TP/PP/CP=1. The first version is single-node, configurable 1/2/4/8 GPUs; each GPU must hold the backbone.
- P: classify request; PR: classify response in request context. Use `safety_label`, never `prompt_safety_label` as PR truth. No taxonomy, no reasoning. Preserve original split, IDs, source/language distribution and binary labels.
- Dataset: pinned HF revision `9823486c583701c8d1438f2319269c4d8eeb93ba`, archive SHA-256 `61e43353485d1d0e3986507b2b4524a2f90afa5f42dc98c67575f8126e61ccf4`. Train 274760 (Nemo 192332, Wild 82428), validation 1000. Not proof that 70:30 is universally optimal.
- One offline prepare stage builds shared token caches per contract. All arms use the same retained conversation content, limited to fit the largest contract overhead. Policy, target, pointer markers and completion supervision are never truncated away.
- Exact data coverage: no dropped/repeated real training examples at the last global batch. Zero-weight padding slots only. A smoke cache contains 128/16 rows and runs one update, saving immediately and verifying adapter/head/optimizer reload, never a full epoch by accident.
- Checkpoints include adapters AND head, FP32 optimizer, per-rank RNG, sampler position, data/model/config fingerprints. Resume must reject changed study/world size. Logs appear on screen and in files. No nohup.
- CPU tests do not establish runtime compatibility with the image or guard accuracy. GPU smoke is an explicit user-run gate before real training.

Follow-on authorization: later on 2026-10-11 the user explicitly requested a
bounded GPU smoke, one optimizer step and a low save interval. `pointer36` passed
on one L40S in the fixed image, including save/reload/forward checks. This does
not authorize full training or establish accuracy, other-arm or multi-GPU proof.
