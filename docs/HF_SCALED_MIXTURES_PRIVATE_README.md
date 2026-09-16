---
license: other
language:
- en
- ar
- de
- es
- fr
- hi
- ja
- th
- zh
task_categories:
- text-classification
tags:
- content-safety
- gated-upstream-derivative
- private-transfer
---

# Qwen3 safety-guard scaled Nemotron/WildGuard mixtures

Private offline-transfer bundles for the common-contract Qwen3 safety-guard study.

## Access and redistribution

These archives contain rendered derivatives of NVIDIA Nemotron Safety Guard V3
(CC-BY-4.0) and the gated AllenAI WildGuardMix release (ODC-BY plus the AI2
Responsible Use terms accepted by the account owner). Keep this repository
private. Do not republish the rendered WildGuard rows in a public dataset
repository.

## Bundles

| Archive | Train | Validation | Actual Nemotron | Actual WildGuard | SHA-256 |
|---|---:|---:|---:|---:|---|
| `nemotron_wildguard_scaled_max_30_70_v2.gated.zip` | 117,183 | 997 | 35,155 | 82,028 | `7b65704a2e305d725b2da99b59eda0f89266d5d59aa5372b0ed319578305c628` |
| `nemotron_wildguard_scaled_max_70_30_v2.gated.zip` | 274,760 | 1,000 | 192,332 | 82,428 | `61e43353485d1d0e3986507b2b4524a2f90afa5f42dc98c67575f8126e61ccf4` |

The released WildGuard file has 86,759 rows. Fourteen fail normalization, 402
overlap configured benchmarks, 839 are canonical duplicates, 31 belong to
conflicting-content groups, and 2,745 rows are removed because their complete
prompt groups touch exact Nemotron overlaps. The remaining training count also
depends on whether 700 or 300 WildGuard validation rows are reserved.

Nemotron is never sampled row-by-row. Selection uses complete upstream IDs,
keeps all surviving language/view rows for every selected ID, and targets equal
counts over the nine selected languages. Train/validation group overlap and
content overlap are both zero in the embedded validation reports.

## Offline installation

Download an archive manually in the browser, copy it into the company
repository's `zip/` directory, verify its SHA-256, then install locally:

```bash
python3 scripts/source_study_bundle.py install \
  --zip zip/nemotron_wildguard_scaled_max_30_70_v2.gated.zip \
  --output work/source-study-scaled/nemotron30_wildguard70_scaled

python3 source_study_natural.py validate \
  --data-dir work/source-study-scaled/nemotron30_wildguard70_scaled
```

Use the analogous `70_30` archive and output directory for the Nemo-heavy arm.
No cluster-side download command is required or permitted by this workflow.
