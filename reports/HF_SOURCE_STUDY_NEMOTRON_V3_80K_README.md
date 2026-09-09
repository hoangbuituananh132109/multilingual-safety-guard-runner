---
license: cc-by-4.0
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
pretty_name: Qwen3 Guard Source Study - Nemotron V3 9-language 80k
---

# Nemotron V3 9-language matched 80k arm

This is the redistributable arm of a controlled Qwen3-4B safety-data source
study. It is derived from
[`nvidia/Nemotron-Safety-Guard-Dataset-v3`](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3)
at revision `a3f7ecb3433d1933701a83f18de16c36934a7f51` under CC-BY-4.0.

The archive contains 80,000 training examples and 1,000 validation examples.
Every view/label cell has a fixed quota and every cell is distributed nearly
equally over `en/ar/de/es/fr/hi/ja/th/zh`. Selection uses seed 3407 and stable
SHA-256 ordering after exact/loose benchmark-overlap removal, content dedupe,
conflict removal and train/validation group isolation.

All examples use the same Nemotron-compatible binary JSON output contract with
taxonomy off and Qwen thinking disabled. The archive manifest records row
counts and SHA-256 hashes.

## Install

Download `nemotron_v3_9lang_80k_v1.zip`, then from the
`no-dataset` branch of the runner:

```bash
python3 scripts/source_study_bundle.py install \
  --zip nemotron_v3_9lang_80k_v1.zip \
  --output work/source-study/nemotron_v3_9lang
python3 source_study.py validate \
  --data-dir work/source-study/nemotron_v3_9lang
```

Archive SHA-256:
`68624e52d632f0a98fe6df31896113571365383ddf7fa99b16f44803ae14168a`.

WildGuardMix and SEA-Safeguard-Train-Cultural-v3 arms are intentionally not
mirrored here because they are access-gated and the latter does not currently
declare a license in its pinned dataset card. Use the runner's pinned-source
downloader after obtaining upstream access.

## Attribution

Please cite the original NVIDIA Nemotron Safety Guard dataset/model work and
preserve CC-BY-4.0 attribution when redistributing derivatives.
