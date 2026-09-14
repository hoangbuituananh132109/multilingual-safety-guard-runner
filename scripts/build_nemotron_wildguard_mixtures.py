from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_DIRS = {
    "nemotron": "nemotron_v3_9lang_natural",
    "wildguard": "wildguardtrain_en_natural",
}

MIXTURES = {
    "nemotron50_wildguard50": {"nemotron": 40_000, "wildguard": 40_000},
    "nemotron70_wildguard30": {"nemotron": 56_000, "wildguard": 24_000},
    "nemotron80_wildguard20": {"nemotron": 64_000, "wildguard": 16_000},
}

VALIDATION_MIXTURES = {
    "nemotron50_wildguard50": {"nemotron": 500, "wildguard": 500},
    "nemotron70_wildguard30": {"nemotron": 700, "wildguard": 300},
    "nemotron80_wildguard20": {"nemotron": 800, "wildguard": 200},
}

SMOKE_MIXTURES = {
    "nemotron50_wildguard50": {"nemotron": 50, "wildguard": 50},
    "nemotron70_wildguard30": {"nemotron": 70, "wildguard": 30},
    "nemotron80_wildguard20": {"nemotron": 80, "wildguard": 20},
}

SMOKE_VALIDATION_MIXTURES = {
    "nemotron50_wildguard50": {"nemotron": 10, "wildguard": 10},
    "nemotron70_wildguard30": {"nemotron": 14, "wildguard": 6},
    "nemotron80_wildguard20": {"nemotron": 16, "wildguard": 4},
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def study_group_id(row: dict[str, Any]) -> str:
    value = str((row.get("metadata") or {}).get("study_group_id") or "")
    if not value:
        raise ValueError(f"Missing study_group_id at source line {row.get('_source_line')}")
    return value


def content_hash(row: dict[str, Any]) -> str:
    value = str(row.get("content_sha256") or "")
    if not value:
        raise ValueError(f"Missing content_sha256 at source line {row.get('_source_line')}")
    return value


def select_complete_groups(
    rows: list[dict[str, Any]],
    budget: int,
    *,
    source: str,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[study_group_id(row)].append(row)

    remaining = budget
    selected: list[dict[str, Any]] = []
    order = sorted(
        groups,
        key=lambda group: stable_hash(f"nemo-wild-mixture:{seed}:{source}:{split}:{group}"),
    )
    for group in order:
        values = groups[group]
        if len(values) > remaining:
            continue
        selected.extend(values)
        remaining -= len(values)
        if remaining == 0:
            break

    if remaining:
        sizes = Counter(len(values) for values in groups.values())
        raise ValueError(
            f"Cannot select exact budget source={source} split={split} budget={budget} "
            f"remaining={remaining} available={len(rows)} group_sizes={dict(sorted(sizes.items()))}"
        )
    return selected


def output_row(row: dict[str, Any], *, source: str, arm: str) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value.pop("_source_line", None)
    metadata = dict(value.get("metadata") or {})
    original_group = str(metadata["study_group_id"])
    metadata.update(
        {
            "original_study_group_id": original_group,
            "study_group_id": f"{source}:{original_group}",
            "mixture_arm": arm,
            "mixture_source": source,
        }
    )
    value["metadata"] = metadata
    return value


def distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    groups: set[str] = set()
    for row in rows:
        source = str((row.get("metadata") or {}).get("mixture_source"))
        counts["rows"] += 1
        counts[f"source:{source}"] += 1
        counts[f"label:{row.get('safety_label')}"] += 1
        counts[f"view:{row.get('view')}"] += 1
        counts[f"language:{row.get('language')}"] += 1
        groups.add(study_group_id(row))
    counts["semantic_groups"] = len(groups)
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        **distribution(rows),
    }


def prepare_pools(source_root: Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    manifests: dict[str, Any] = {}
    for source, directory_name in SOURCE_DIRS.items():
        directory = source_root / directory_name
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifests[source] = json.loads(manifest_path.read_text(encoding="utf-8"))
        pools[source] = {
            split: read_jsonl(directory / f"{split}.jsonl")
            for split in ("train", "validation")
        }

    occurrences: dict[str, set[str]] = defaultdict(set)
    for source, by_split in pools.items():
        for rows in by_split.values():
            for row in rows:
                occurrences[content_hash(row)].add(source)
    cross_source_hashes = {value for value, sources in occurrences.items() if len(sources) > 1}

    excluded_groups: dict[str, set[str]] = defaultdict(set)
    conflicting_hashes: set[str] = set()
    labels: dict[str, set[str]] = defaultdict(set)
    for source, by_split in pools.items():
        for rows in by_split.values():
            for row in rows:
                value = content_hash(row)
                if value in cross_source_hashes:
                    excluded_groups[source].add(study_group_id(row))
                    labels[value].add(str(row.get("safety_label")))
    conflicting_hashes = {value for value, values in labels.items() if len(values) > 1}

    cleaned: dict[str, dict[str, list[dict[str, Any]]]] = {}
    removed_rows: dict[str, dict[str, int]] = {}
    for source, by_split in pools.items():
        cleaned[source] = {}
        removed_rows[source] = {}
        for split, rows in by_split.items():
            kept = [row for row in rows if study_group_id(row) not in excluded_groups[source]]
            cleaned[source][split] = kept
            removed_rows[source][split] = len(rows) - len(kept)

    audit = {
        "policy": "drop every complete semantic group touching an exact-content overlap across sources or splits",
        "cross_source_exact_hashes": len(cross_source_hashes),
        "cross_source_conflicting_hashes": len(conflicting_hashes),
        "excluded_semantic_groups": {source: len(values) for source, values in excluded_groups.items()},
        "removed_rows": removed_rows,
        "remaining_rows": {
            source: {split: len(rows) for split, rows in by_split.items()}
            for source, by_split in cleaned.items()
        },
    }
    return cleaned, {"audit": audit, "parent_manifests": manifests}


def build_arm(
    arm: str,
    train_budgets: dict[str, int],
    validation_budgets: dict[str, int],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    provenance: dict[str, Any],
    output_dir: Path,
    *,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, budgets in (("train", train_budgets), ("validation", validation_budgets)):
        for source, budget in budgets.items():
            rows = select_complete_groups(
                pools[source][split],
                budget,
                source=source,
                split=split,
                seed=seed,
            )
            selected[split].extend(output_row(row, source=source, arm=arm) for row in rows)
        selected[split].sort(
            key=lambda row: stable_hash(f"mixture-output:{seed}:{arm}:{split}:{row['example_id']}")
        )

    for split, rows in selected.items():
        owners: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        for row in rows:
            metadata = row.get("metadata") or {}
            owners[content_hash(row)].add(
                (
                    str(metadata.get("mixture_source")),
                    str(metadata.get("study_group_id")),
                    str(row.get("safety_label")),
                )
            )
        invalid = {value: references for value, references in owners.items() if len(references) > 1}
        if invalid:
            raise AssertionError(
                f"{arm}: {len(invalid)} exact contents cross source/group/label boundaries in {split}"
            )

    train_content = {content_hash(row) for row in selected["train"]}
    validation_content = {content_hash(row) for row in selected["validation"]}
    if train_content & validation_content:
        raise AssertionError(f"{arm}: exact content crosses train and validation")

    output_dir.mkdir(parents=True, exist_ok=False)
    split_reports = {
        split: write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in selected.items()
    }
    manifest = {
        "schema_version": 1,
        "study": "nemotron_wildguard_fixed_80k_mixture",
        "source": arm,
        "seed": seed,
        "smoke": smoke,
        "train_target_rows": sum(train_budgets.values()),
        "validation_target_rows": sum(validation_budgets.values()),
        "train_examples": len(selected["train"]),
        "validation_examples": len(selected["validation"]),
        "mixture": {"train": train_budgets, "validation": validation_budgets},
        "sampling": "stable SHA-256 sampling of complete semantic groups from cleaned natural arms",
        "rendering": {
            "label_contract": "nemotron_binary_json_taxonomy_off",
            "thinking_mode": "no_think",
        },
        "cross_source_cleaning": provenance["audit"],
        "parent_manifest_sha256": {
            source: stable_hash(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
            for source, manifest in provenance["parent_manifests"].items()
        },
        "splits": split_reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean fixed-budget Nemotron/WildGuard mixtures.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("work/source-study-natural"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/source-study-mixtures"),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    pools, provenance = prepare_pools(args.source_root)
    train_mixtures = SMOKE_MIXTURES if args.smoke else MIXTURES
    validation_mixtures = SMOKE_VALIDATION_MIXTURES if args.smoke else VALIDATION_MIXTURES
    root = args.output_root / "_smoke" if args.smoke else args.output_root

    reports: dict[str, Any] = {}
    for arm, train_budgets in train_mixtures.items():
        reports[arm] = build_arm(
            arm,
            train_budgets,
            validation_mixtures[arm],
            pools,
            provenance,
            root / arm,
            seed=args.seed,
            smoke=args.smoke,
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
