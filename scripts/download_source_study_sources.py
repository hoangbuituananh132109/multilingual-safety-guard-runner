"""Download pinned raw sources for rebuilding the matched 80k study arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import get_token, snapshot_download


SOURCES = {
    "nemotron_v3_9lang": {
        "repo_id": "nvidia/Nemotron-Safety-Guard-Dataset-v3",
        "revision": "a3f7ecb3433d1933701a83f18de16c36934a7f51",
        "allow_patterns": [f"{lang}/*.jsonl" for lang in ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh")],
        "gated": False,
    },
    "wildguardtrain_en": {
        "repo_id": "allenai/wildguardmix",
        "revision": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
        "allow_patterns": ["train/wildguard_train.parquet"],
        "gated": True,
    },
    "sea_cultural_vi": {
        "repo_id": "aisingapore/SEA-Safeguard-Train-Cultural-v3",
        "revision": "2a51a837855ff7ae7d383fa4af6fa11c2cee8a92",
        "allow_patterns": ["Vietnam/train_refined_v1_qa_pass_only-*.parquet"],
        "gated": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("input/source-study/raw"))
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    selected = args.source or list(SOURCES)
    outputs = {}
    for name in selected:
        spec = SOURCES[name]
        if spec["gated"] and not token:
            raise RuntimeError(f"{spec['repo_id']} is gated; export an accepted HF_TOKEN first")
        destination = args.output_root / name
        path = snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=spec["revision"],
            allow_patterns=spec["allow_patterns"],
            local_dir=destination,
            token=token if spec["gated"] else None,
        )
        outputs[name] = {**spec, "local_dir": str(Path(path).resolve())}
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
