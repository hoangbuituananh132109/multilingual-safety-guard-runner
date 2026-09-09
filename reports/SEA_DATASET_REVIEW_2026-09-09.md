# SEA dataset review (2026-09-09)

## Conclusion

`aisingapore/SEA-Instruct-2602` is a general instruction-tuning corpus, not the
training corpus described in the SEA-Guard paper. It must not replace a safety
dataset directly. It can be tested as Vietnamese safe/benign retention data or
as prompt-only weak supervision after explicit filtering.

The paper-aligned safety corpus is
`aisingapore/SEA-Safeguard-Train-Cultural-v3`, newly made public on the AI
Singapore account. Access was granted and verified on 2026-09-09.
`aisingapore/SEASafeguardBench` is a separate held-out evaluation artifact,
remains pending for the current account, and must never be mixed into training.

## SEA-Instruct Vietnamese sample

A deterministic 10,000-row sample was collected from 100 equal-width strata
across the complete 1,153,050-row Vietnamese split. This avoids treating a
source-clustered prefix as representative.

| Field | Count | Percent |
| --- | ---: | ---: |
| Safe | 8,710 | 87.10% |
| Caution | 1,215 | 12.15% |
| Harmful adversarial | 39 | 0.39% |
| Harmful non-adversarial | 36 | 0.36% |
| Requires local cultural knowledge | 1,326 | 13.26% |
| Region explicitly Vietnam | 2,630 | 26.30% |
| Source `aisingapore/SEASafeguardMix` | 158 | 1.58% |

Of the 75 sampled harmful prompts, 42 (56.00%) came from
`aisingapore/SEASafeguardMix`. The dataset's safety supervision is therefore
sparse and strongly concentrated in a pre-existing safeguard source.

The sample contained no duplicate conversation IDs and no conversation parse
errors. All rows contained one user and one assistant turn, optionally preceded
by a system turn. Mean user length was 601.79 characters and mean assistant
length was 2,853.46 characters.

The metadata value `prompt_primary_language=English` must not be interpreted as
English output inside the Vietnamese config. For translated rows, the dataset
card says the tags describe the original English prompt; inspected translated
rows contained Vietnamese conversation text.

## What can and cannot be used

Safe first use:

- Treat only the user prompt as a candidate guard-training input.
- Use `Safe` rows as Vietnamese benign/hard-negative examples to control false
  positives and over-refusal.
- Use `Harmful_Adversarial` and `Harmful_NonAdversarial` as weak prompt-only
  unsafe supervision, with a separate provenance flag.
- Exclude `Caution` from the first clean binary experiment. Its mapping is a
  policy decision and should be a later ablation.
- Exclude sources `aisingapore/SEASafeguardMix` and
  `aisingapore/SEA-Safeguard-Train`, then deduplicate against every evaluation
  set once access is granted.

Unsafe use:

- Do not label the generated assistant response from `prompt_sensitivity`; that
  annotation applies only to the prompt. Manual inspection found both refusals
  and potentially harmful compliance among assistant responses.
- Do not map `Caution` silently to safe or unsafe.
- Do not populate Nemotron N23 categories from SEA-Instruct domains/tasks; they
  are different taxonomies.

## Paper-aligned cultural safety data

The SEA-Guard paper generates SEA-cultural prompts and responses, labels prompt
and response separately using ten stochastic MCRE passes, maps a five-level
ordinal risk scale to Safe/Sensitive/Harmful, performs cultural/topic/usage QA,
then removes shortcut-like examples with a bag-of-words classifier. The released
v3 Parquet names its third class `Unsafe`, not `Harmful`. The paper trains for
one epoch and evaluates with Sensitive prompt -> Safe and Sensitive response ->
Harmful/Unsafe.

The gated Vietnam config exposes these variants:

| Split | Rows | Compressed files |
| --- | ---: | ---: |
| `train` | 1,005,771 | 1.30 GB |
| `train_refined_v1` | 945,264 | 1.11 GB |
| `train_refined_v1_qa_pass_only` | 802,112 | 1.06 GB |
| `train_refined_v1_100k` | 95,510 | 132.66 MB |

The 802,112 QA-pass-only count exactly matches the Vietnam prompt-task and
response-task totals reported in the paper. All five shards (1.06 GB compressed)
were downloaded and audited.

The provided 95,510-row `train_refined_v1_100k` must not be used directly:

| Property | Count |
| --- | ---: |
| QA Pass | 76,686 |
| QA Not Pass | 18,824 |
| English rows | 48,473 |
| Vietnamese rows | 47,037 |
| Prompt tasks | 28,233 |
| Response tasks | 67,277 |
| Distinct semantic IDs | 14,117 |

The full QA-pass-only source contains 802,112 task rows, 136,226 semantic IDs,
405,986 English rows and 396,126 Vietnamese rows. It is already expanded into
prompt-only and response-classification tasks; it must not be expanded a second
time. IDs repeat across the English/Vietnamese pair and across multiple generated
responses. Sampling and train/validation splitting must therefore be grouped by
`id`, not performed independently by row.

The bilingual pairing is exact at the prompt level. All 136,226 IDs contain one
English prompt row and one Vietnamese prompt row. Within every pair, the topic,
prompt label and prompt risk score agree; 213 pairs also have identical surface
text (mostly language-neutral strings). Response rows are attached to the same
ID, but filtering is asymmetric: there are 269,760 English response tasks and
259,900 Vietnamese response tasks, so a training builder must not require every
response to have a surviving bilingual counterpart. Nor should paired responses
be treated as translations: among 218,022 unambiguous `(id, response_model)`
pairs with exactly one response per language, only 83.74% share the same
three-way label and only 43.20% share the exact risk score.

For the Vietnamese half alone, the released QA-pass source contains 396,126 task
rows: 136,226 prompt-classification rows and 259,900 response-classification
rows. Its three-way effective labels are 246,473 Safe, 97,563 Sensitive and
52,090 Unsafe. Under the paper's binary evaluation mapping, this becomes 312,163
safe and 83,963 unsafe rows.

The 53 intended topics appear alongside `Other` and 48 missing topic values.
Several released strings retain spelling errors (`Retriement`, `Promp injection`,
and `Grambling`); these should be normalized only as metadata, never used as a
Nemotron safety taxonomy target.

The released labels use risk thresholds Safe < 0.4, Sensitive 0.4-0.6, and
Unsafe > 0.6 with zero mismatches across the QA-pass-only split. This differs
from the published paper's stated 0.33/0.66 thresholds for 45,811 rows. Preserve
the released labels as ground truth; do not recompute them from the paper
thresholds.

There is also a reproducibility discrepancy between the paper, release variants
and uploaded model artifacts. The seven country `train_refined_v1` splits sum to
6,654,914 rows. The public 8B trainer state has 10,399 steps with per-device batch
5, while the earlier model card reports 64 devices and gradient accumulation 2:
`ceil(6,654,914 / (64*5*2)) = 10,399`. The public 4B state similarly has 8,666
steps with per-device batch 6, consistent with `ceil(6,654,914 / (64*6*2))`.
This is strong artifact-level evidence that those uploaded Qwen models used the
full seven-country `train_refined_v1` collection, whereas the paper's label
tables exactly match the smaller 5,658,506-row QA-pass-only collection. The
current model cards instead say 32 devices. Treat exact official reproduction as
underspecified; use QA-pass-only for the cleaner controlled experiment.

The preferred first SEA arm is a deterministic, semantic-ID-grouped sample from
the Vietnamese rows of `train_refined_v1_qa_pass_only`, preserving its natural
P/PR and label distribution. English companions are valuable but should be a
separate bilingual ablation. In both arms, each released row remains one SFT
classification example; ID grouping is for sampling and split isolation, not for
concatenating all responses into one long example.

## Risk-score supervision and label-quality spot check

The released score is an expected ordinal severity produced by the MCRE teacher,
not a calibrated probability that an example is unsafe. It nevertheless retains
more information than the thresholded three-way label. In the Vietnamese half,
99.87% of prompt scores and 99.80% of response scores lie on the expected 0.025
grid from ten votes over five severity levels; the small remainder uses other
fractions. Many rows sit exactly at the release boundaries: prompt tasks contain
7,737 rows at 0.4 and 1,646 at 0.6, while response tasks contain 2,913 and 1,232.

A scalar severity model is therefore a valid later modeling ablation. The clean
implementation is a one-logit sequence-classification head trained with a
bounded soft-label loss, followed by separate prompt and response calibration.
Emitting decimal strings from the current causal-LM completion trainer is not a
recommended substitute. Thresholds must be selected on an ID-disjoint
calibration split and frozen before benchmark testing.

The user's hypothesis about aligned response generators is supported. Vietnamese
response labels by source model are:

| Response model | Safe | Sensitive | Unsafe |
| --- | ---: | ---: | ---: |
| Gemma-SEA-LION-v4-27B | 89.03% | 8.82% | 2.14% |
| Gemma-3-27B | 84.06% | 10.98% | 4.97% |
| Llama-3.3-70B | 92.92% | 5.46% | 1.62% |
| GPT-OSS-20B | 66.21% | 26.13% | 7.66% |

Even among unsafe prompts, 63.86% of generated Vietnamese responses are labeled
Safe, 22.31% Sensitive and 13.82% Unsafe. Prompt and response tasks are therefore
meaningfully different and should receive separate policy thresholds.

The prompt-label distribution also exposes strong topic priors:

| Topic | Safe | Sensitive | Unsafe |
| --- | ---: | ---: | ---: |
| Politics | 1.59% | 39.84% | 58.57% |
| Religion | 9.08% | 56.13% | 34.79% |
| LGBTQ+ | 4.55% | 89.17% | 6.28% |
| Territorial dispute | 2.98% | 80.60% | 16.42% |
| Elections | 15.92% | 74.61% | 9.47% |
| Discrimination | 0.78% | 31.78% | 67.44% |

Manual inspection confirms genuine Vietnamese grounding (for example Hoang Sa,
Truong Sa, Hoa Hao, Cao Dai, Khmer Krom, the Central Highlands, Phu Quoc, Zalo,
MoMo and Vietcombank), but grounding does not imply label correctness. Clear
errors include a benign request for Vietnamese LGBTQ+ artist recommendations
labeled Sensitive at 0.5 and an inclusive story about a transgender woman and a
conservative aunt ending positively labeled Unsafe at 0.7. Neutral religious,
historical and civic-discussion prompts are also frequently assigned Sensitive.
Other rows correctly distinguish scam, covert manipulation and destructive
requests from factual or respectful cultural discussion. The dataset is useful
weak supervision, but the culturally sensitive high-score tail needs a targeted
re-annotation/filtering ablation before treating the score as policy truth.

## Clean first-stage dataset experiment

Use Qwen3-4B as the primary screening model, start every arm from the same base
checkpoint, use taxonomy-OFF and no-think for all arms, and keep the same
100k-example or token/update budget. One epoch is then comparable.

1. Nemotron V3 only.
2. WildGuard only.
3. SEA Cultural Vietnam QA-pass-only only.
4. Nemotron V3 + SEA Cultural Vietnam.
5. Nemotron V3 + WildGuard + SEA Cultural Vietnam.
6. Best arm + a small SEA-Instruct benign-Vietnamese component (for example
   10% or 25%), tested only after the first five arms.

Evaluate Vietnamese cultural safety first, then generic retention. Use the new
SEA-SafeguardBench only for evaluation, alongside the current SEA-VI,
CultureGuard/standard, WildGuardTest, XSafety and MultiJail checks. An epoch is
not a controlled budget if arms contain different numbers of examples.

## Reproduction

```powershell
$env:PYTHONUTF8='1'
python scripts/sample_sea_instruct_vi.py --rows 10000 --window-size 100 --seed 3407 --workers 12
```

The sample is an analysis artifact, not a prepared training set.
