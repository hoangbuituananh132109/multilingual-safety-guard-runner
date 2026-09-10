from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import get_token, hf_hub_download


XSTEST_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
WILDGUARD_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
SEALSBENCH_REVISION = "fc8125c1360f4aa407ef139bffaeaac5275137e8"
LINGUASAFE_REVISION = "7cb0bc217efd98293ff715294606522a8625321d"
XSTEST_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    f"{XSTEST_REVISION}/xstest_prompts.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            example_id = str(row["example_id"])
            if example_id in seen:
                raise ValueError(f"Duplicate benchmark example_id: {example_id}")
            seen.add(example_id)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            counts[f"language:{row['language']}"] += 1
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
            counts[f"subset:{row['subset']}"] += 1
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        **dict(sorted(counts.items())),
    }


def xstest_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt = str(row.get("prompt") or "").strip()
            label = str(row.get("label") or "").casefold()
            if not prompt or label not in {"safe", "unsafe"}:
                continue
            source_id = str(row.get("id") or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16])
            yield {
                "example_id": f"xstest:{source_id}:P",
                "source_id": source_id,
                "source_split": "test",
                "dataset_source": "paul-rottger/xstest",
                "language": "en",
                "view": "P",
                "subset": str(row.get("type") or "ALL"),
                "topic": row.get("note"),
                "text": f"Prompt: {prompt}",
                "prompt": prompt,
                "response": None,
                "safety_label": label,
                "prompt_safety_label": label,
                "categories": [],
                "category_scope": "prompt",
                "metadata": {"focus": row.get("focus")},
            }


def _harm_label(value: Any) -> str | None:
    return {"harmful": "unsafe", "unharmful": "safe"}.get(str(value or "").casefold())


def wildguard_test_rows(path: Path) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        prompt = str(row.get("prompt") or "").strip()
        response = row.get("response")
        response = str(response).strip() if response is not None else None
        prompt_label = _harm_label(row.get("prompt_harm_label"))
        response_label = _harm_label(row.get("response_harm_label"))
        source_id = str(index)
        subset = "adversarial" if bool(row.get("adversarial")) else "vanilla"
        metadata = {
            "subcategory": row.get("subcategory"),
            "prompt_harm_agreement": row.get("prompt_harm_agreement"),
            "response_harm_agreement": row.get("response_harm_agreement"),
            "response_refusal_label": row.get("response_refusal_label"),
            "response_refusal_agreement": row.get("response_refusal_agreement"),
        }
        if prompt and prompt_label:
            yield {
                "example_id": f"wildguardtest:{source_id}:P",
                "source_id": source_id,
                "source_split": "test",
                "dataset_source": "allenai/wildguardmix:wildguardtest",
                "language": "en",
                "view": "P",
                "subset": subset,
                "topic": row.get("subcategory"),
                "text": f"Prompt: {prompt}",
                "prompt": prompt,
                "response": None,
                "safety_label": prompt_label,
                "prompt_safety_label": prompt_label,
                "categories": [],
                "category_scope": "prompt",
                "metadata": metadata,
            }
        if prompt and response and response_label:
            yield {
                "example_id": f"wildguardtest:{source_id}:PR",
                "source_id": source_id,
                "source_split": "test",
                "dataset_source": "allenai/wildguardmix:wildguardtest",
                "language": "en",
                "view": "PR",
                "subset": subset,
                "topic": row.get("subcategory"),
                "text": f"Prompt: {prompt}\nResponse: {response}",
                "prompt": prompt,
                "response": response,
                "safety_label": response_label,
                "prompt_safety_label": prompt_label or "safe",
                "categories": [],
                "category_scope": "response",
                "metadata": metadata,
            }


def sealsbench_vi_rows(path: Path) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        if str(row.get("target_language") or "").casefold() != "vietnamese":
            continue
        prompt = str(row.get("prompt") or "").strip()
        label = str(row.get("label") or "").casefold()
        if not prompt or label not in {"safe", "unsafe"}:
            continue
        source_id = f"{row.get('ID')}:{row.get('Unnamed: 0', index)}"
        yield {
            "example_id": f"sealsbench:{source_id}:P",
            "source_id": source_id,
            "source_split": "test",
            "dataset_source": "MickyMike/SEALSBench",
            "language": "vi",
            "view": "P",
            "subset": str(row.get("origin") or "ALL"),
            "topic": row.get("category"),
            "text": f"Prompt: {prompt}",
            "prompt": prompt,
            "response": None,
            "safety_label": label,
            "prompt_safety_label": label,
            "categories": [],
            "category_scope": "prompt",
            "metadata": {
                "category_raw": row.get("category"),
                "unsafe_type_index": row.get("unsafe_type_index"),
                "label_index": row.get("label_index"),
            },
        }


def linguasafe_vi_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Adapt the Vietnamese LinguaSafe slice to the binary guard schema.

    LinguaSafe severity L0 is benign. L1-L3 are increasingly unsafe. The
    original level remains in metadata so severity-aware evaluation can be
    added without rebuilding the benchmark.
    """
    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        if str(row.get("lang") or "").casefold() != "vi":
            continue
        prompt = str(row.get("prompt") or "").strip()
        level = row.get("level")
        if not prompt or level not in {0, 1, 2, 3}:
            continue
        source = row.get("source")
        yield {
            "example_id": f"linguasafe:{index}:P:vi",
            "source_id": str(row.get("id") if row.get("id") is not None else index),
            "source_split": "train",
            "dataset_source": "zhiyuan-ning/linguasafe",
            "language": "vi",
            "view": "P",
            "subset": str(row.get("type") or "ALL"),
            "topic": row.get("subtype"),
            "text": f"Prompt: {prompt}",
            "prompt": prompt,
            "response": None,
            "safety_label": "safe" if level == 0 else "unsafe",
            "prompt_safety_label": "safe" if level == 0 else "unsafe",
            "categories": [],
            "category_scope": "prompt",
            "metadata": {
                "severity_level": level,
                "language_specific": row.get("specific"),
                "type_raw": row.get("type"),
                "subtype_raw": row.get("subtype"),
                "source": source,
            },
        }


def download_xstest(cache_dir: Path) -> Path:
    path = cache_dir / f"xstest-{XSTEST_REVISION}.csv"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(XSTEST_URL, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path


def hf_file(repo_id: str, filename: str, revision: str, token: str | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare extra source-study guard benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=Path("work/benchmarks"))
    parser.add_argument("--cache-dir", type=Path, default=Path("input/source-study/benchmarks"))
    parser.add_argument(
        "--skip-wildguard",
        action="store_true",
        help="Prepare only ungated XSTest, SEALSBench-VI, and LinguaSafe-VI.",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()

    xstest = download_xstest(args.cache_dir)
    seals = hf_file(
        "MickyMike/SEALSBench",
        "data/test-00000-of-00001.parquet",
        SEALSBENCH_REVISION,
        token=None,
    )
    linguasafe = hf_file(
        "zhiyuan-ning/linguasafe",
        "data/train-00000-of-00001.parquet",
        LINGUASAFE_REVISION,
        token=None,
    )
    manifest: dict[str, Any] = {
        "suite": "source_study_extra_benchmarks_v2",
        "sources": {
            "paul-rottger/xstest": XSTEST_REVISION,
            "MickyMike/SEALSBench": SEALSBENCH_REVISION,
            "zhiyuan-ning/linguasafe": LINGUASAFE_REVISION,
        },
        "notes": {
            "xstest": "English over-refusal/hard-negative diagnostic; 250 safe and 200 unsafe contrast prompts.",
            "sealsbench_vi": "Vietnamese machine-translated robustness diagnostic from the SEALGuard replication package; not a culturally native replacement for SEA-SafeguardBench.",
            "wildguardtest": "Human-annotated English prompt and response harm test; kept disjoint from WildGuardTrain.",
            "linguasafe_vi": "Vietnamese LinguaSafe prompt slice used by NVIDIA for multilingual guard evaluation. Binary mapping is L0=safe and L1-L3=unsafe; original L0-L3 severity and language-specific flags are preserved for the official alpha=0.6 severity-weighted F1/FPR. The upstream VI slice contains one exact prompt with conflicting L0/L2 labels (three rows); it is retained for upstream fidelity.",
        },
        "benchmarks": {},
    }
    manifest["benchmarks"]["xstest"] = write_jsonl(args.output_dir / "xstest_en.jsonl", xstest_rows(xstest))
    manifest["benchmarks"]["sealsbench_vi"] = write_jsonl(
        args.output_dir / "sealsbench_vi.jsonl",
        sealsbench_vi_rows(seals),
    )
    manifest["benchmarks"]["linguasafe_vi"] = write_jsonl(
        args.output_dir / "linguasafe_vi.jsonl",
        linguasafe_vi_rows(linguasafe),
    )

    if not args.skip_wildguard:
        if not token:
            raise RuntimeError("WildGuardTest is gated. Set HF_TOKEN after accepting allenai/wildguardmix, or use --skip-wildguard.")
        wildguard = hf_file(
            "allenai/wildguardmix",
            "test/wildguard_test.parquet",
            WILDGUARD_REVISION,
            token=token,
        )
        manifest["sources"]["allenai/wildguardmix:wildguardtest"] = WILDGUARD_REVISION
        manifest["benchmarks"]["wildguardtest"] = write_jsonl(
            args.output_dir / "wildguardtest_en.jsonl",
            wildguard_test_rows(wildguard),
        )

    output = args.output_dir / "source_study_benchmark_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
