# Stage-2 methodology audit — 2026-08-29

## Scope and status

This audit covers the schema-v2 Gemini Stage-2 bundle used by the already
launched company run, commit `89ec5f7`, and the local schema-v3 corrective
code. The running job was not modified or stopped.

The schema-v2 run is reproducible but exploratory. It must not be used as a
clean causal estimate of the effect of Vietnamese data, reasoning, taxonomy
mode, or WildGuard.

## Confirmed schema-v2 defects

1. Taxonomy eligibility required a non-empty N23 category even for all-safe
   rows. This made taxonomy prompt presence strongly correlated with unsafe
   labels instead of implementing an unconditional 75/25 V3/VI mode split.
2. V3 and VI prompt-only views discarded `violated_categories`. All sampled
   unsafe prompt-only rows therefore became taxonomy-OFF despite having a
   valid upstream category.
3. The target renderer emitted `Safety Categories` for 2,246 all-safe
   reasoning rows that retained an auxiliary category, contradicting the V3
   and 3.5 instruction to omit categories when all classified turns are safe.
4. V3 selected the intended language text but stored the stale final loop
   variable as metadata. All 56,409 schema-v2 V3 rows were labeled `zh`.
5. The full rendered data contained 8,109 exact-content hashes shared across
   sources, 883 exact-content hashes present in both train and validation, and
   hundreds of same-content label conflicts. The old validator checked only
   `semantic_id`, so it reported a false clean result.
6. Smoke runs wrote `train_results.json` into real experiment directories.
   The suite used that file alone as its completion signal, so a smoke could
   cause a real one-epoch run to be skipped.
7. The bundle manifest had `benchmark_hash_count=0` because benchmark inputs
   were unavailable. Any benchmark result from this run remains preliminary
   until leakage is audited against the exact benchmark artifacts.

## Corrective code now implemented locally

- Safe examples may be taxonomy-ON without a category target.
- Unsafe taxonomy-ON examples require at least one valid N23 category.
- Prompt-only V3/VI views retain upstream category annotations.
- All-safe targets always omit `Safety Categories`.
- V3 records store the actually selected language.
- Smoke outputs are isolated under `runs-stage2/_smoke/*`.
- Full-run completion requires a final model artifact or `run_complete.json`.
- The validator and builder detect cross-split exact content and conflicting
  labels; invalid data cannot be bundled as training-ready by default.
- The Nemotron 3.5 text blend uses an explicit source allowlist. This preserves
  the previous 6,000 raw selected rows while documenting why Aegis V3 replay,
  reasoning duplicates, image-grounded data, and topic-following are excluded.

Twelve unit tests pass after these corrections. The exact Qwen tokenizer audit
found 975/233,760 training rows (0.42%) above the 2,048-token sequence budget;
no supervised target itself exceeded that budget. V3 is the largest affected
source at 579/53,061 rows (1.09%).

## Schema-v3 audit build

The local audit artifact is `work/stage2/audit_fixed_taxonomy_v3` and is not a
release bundle. Counts after correcting taxonomy/category/language rendering:

| Source | Rows | Taxonomy ON | Taxonomy OFF |
|---|---:|---:|---:|
| V3 replay | 56,425 | 42,432 | 13,993 |
| Gemini VI | 53,832 | 40,329 | 13,503 |
| Reasoning | 35,904 | 35,904 | 0 |
| Nemotron 3.5 selected | 9,658 | 9,601 | 57 |
| WildGuardTrain | 86,316 | 0 | 86,316 |
| **Total** | **242,135** | **128,266** | **113,869** |

The old bundle had only 49,854 taxonomy-ON rows. The corrected renderer adds
78,412 taxonomy-ON exposures, showing that the original error was material.

The audit build is deliberately `training_ready=false` because it still has:

- 883 content hashes across train and validation;
- 568 content hashes with conflicting binary labels;
- no configured benchmark hashes for leakage exclusion.

These are research-policy decisions, not mechanical errors to hide with an
arbitrary source precedence rule.

## Interpretation of the already running job

- Preserve every checkpoint, log, resolved config, and data manifest.
- Label the run `R0-schema-v2-exploratory`.
- Do not use internal validation loss as clean generalization evidence because
  exact content crosses train/validation.
- If smoke artifacts caused separate one-epoch runs to be skipped, use the
  epoch-1 checkpoint from each five-epoch run. The five-epoch configs use the
  same data, seed, optimizer settings, and save at every epoch.
- Benchmark every retained epoch using identical decoding and report parse
  rate, safe performance, unsafe recall/F1, taxonomy metrics where applicable,
  and SEA-VI separately. Mark all results preliminary until benchmark leakage
  is checked.

## Research plan for the replacement study

1. Build a canonical semantic registry before rendering. Group by upstream ID
   where available and exact/loose content hashes otherwise.
2. Quarantine label-conflict groups instead of silently choosing a source.
   Audit a stratified sample, then document any source precedence rule.
3. Assign train/validation once per semantic group, before source-specific
   rendering. Keep benchmark test sets completely external.
4. Make taxonomy sampling stratified by source, P/PR view, binary label, and
   language. Treat 75/25 as a project hypothesis: public NVIDIA artifacts show
   category/no-category and think/no-think modes but do not document this exact
   training ratio.
5. Keep one common replay control in every causal arm. Compare replay-only,
   replay+VI, replay+reasoning, replay+WildGuard, and full. VI-only versus
   reasoning-only is not enough because it confounds data contribution with
   forgetting and unequal training compute.
6. Match optimizer steps or target tokens across screening arms. One epoch of
   35k, 54k, and 234k rows is not a compute-matched comparison.
7. Run one-seed screening first, then repeat the best candidates with at least
   three seeds. Evaluate epoch/step learning curves instead of treating five
   epochs as one terminal result.
8. Evaluate taxonomy ON/OFF and Qwen THINK/NO-THINK as separate inference
   conditions on the same examples. Include classification, category tagging,
   parse reliability, over-refusal, latency, and token cost.
9. Test learning rate (`3e-6` versus `1e-5`) before changing LoRA rank/modules.
   Keep the current r=8 q/v adapter as the continuity baseline.

