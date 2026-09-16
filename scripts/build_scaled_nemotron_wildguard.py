from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from core.source_study_data import StudyRow, benchmark_hashes, canonicalize, stable_hash
from core.source_study_natural_data import (
    _distribution,
    _render_binary,
    _v3_natural_rows,
    _wildguard_natural_rows,
    _write_jsonl,
    validate_natural_arm,
)


LANGUAGES = ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh")


def fit_mixed_budgets(
    *,
    requested_nemotron: int,
    requested_wildguard: int,
    nemotron_percent: int,
    wildguard_percent: int,
    nemo_capacity: int,
    wild_capacity: int,
    balanced_nemo_capacity: int | None = None,
) -> dict[str, int]:
    """Fit a mixed recipe to post-cleaning capacities while preserving its ratio.

    Raw source headlines are not capacities: benchmark leakage, cross-source
    overlap removal, and complete-group selection can reduce what is eligible.
    WildGuard is the limiting source in the scale recipes, so clamp it first,
    derive the proportional Nemotron budget, then re-check both capacities.
    """
    if not (0 < nemotron_percent < 100 and 0 < wildguard_percent < 100):
        raise ValueError("fit_mixed_budgets requires a mixed (non-pure) ratio")
    if nemotron_percent + wildguard_percent != 100:
        raise ValueError("mixture percentages must sum to 100")
    if min(requested_nemotron, requested_wildguard, nemo_capacity, wild_capacity) < 0:
        raise ValueError("budgets and capacities must be non-negative")

    wild = min(requested_wildguard, wild_capacity)
    nemo = min(requested_nemotron, round(wild * nemotron_percent / wildguard_percent))
    if balanced_nemo_capacity is not None:
        nemo = min(nemo, balanced_nemo_capacity)
    if nemo_capacity >= 0:
        nemo = min(nemo, nemo_capacity)

    # Re-derive the WildGuard side after any Nemotron cap so the realized pair
    # never claims more rows than the selected ratio can support.
    wild = min(wild, round(nemo * wildguard_percent / nemotron_percent))
    if nemo <= 0 or wild <= 0:
        raise ValueError(
            "mixed recipe has no positive feasible budget after capacity fitting: "
            f"nemo={nemo}, wildguard={wild}"
        )
    return {"nemotron": nemo, "wildguard": wild}


def group_rows(rows: Iterable[StudyRow]) -> dict[str, list[StudyRow]]:
    groups: dict[str, list[StudyRow]] = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    return dict(groups)


def select_complete_groups(
    rows: list[StudyRow],
    budget: int,
    *,
    seed: int,
    source: str,
    split: str,
) -> tuple[list[StudyRow], set[str]]:
    if budget == 0:
        return [], set()
    groups = group_rows(rows)
    selected: list[StudyRow] = []
    selected_ids: set[str] = set()
    remaining = budget
    order = sorted(
        groups,
        key=lambda group_id: stable_hash(f"scaled-mixture:{seed}:{source}:{split}:{group_id}"),
    )
    for group_id in order:
        values = groups[group_id]
        if len(values) > remaining:
            continue
        selected.extend(values)
        selected_ids.add(group_id)
        remaining -= len(values)
        if remaining == 0:
            break
    if remaining:
        sizes = Counter(len(values) for values in groups.values())
        raise ValueError(
            f"Cannot select exact budget source={source} split={split} budget={budget}; "
            f"remaining={remaining}, available={sum(map(len, groups.values()))}, "
            f"group_sizes={dict(sorted(sizes.items()))}"
        )
    return selected, selected_ids


def select_balanced_nemotron(
    rows: list[StudyRow], budget: int, *, seed: int, split: str, allow_tiny_shortfall: bool = False
) -> tuple[list[StudyRow], set[str]]:
    """Select complete upstream IDs against an explicit near-equal language quota."""
    if budget == 0:
        return [], set()
    base, remainder = divmod(budget, len(LANGUAGES))
    remainder_order = sorted(
        LANGUAGES, key=lambda language: stable_hash(f"scaled-language-remainder:{seed}:{split}:{language}")
    )
    targets = {language: base + int(language in remainder_order[:remainder]) for language in LANGUAGES}
    groups = group_rows(rows)

    def vector(values: list[StudyRow]) -> dict[str, int]:
        result = {language: 0 for language in LANGUAGES}
        for row in values:
            if row.language not in result:
                raise ValueError(f"Unexpected Nemotron language: {row.language}")
            result[row.language] += 1
        return result

    vectors = {group_id: vector(values) for group_id, values in groups.items()}
    stable = lambda group_id: stable_hash(f"scaled-balanced:{seed}:{split}:{group_id}")
    counts = {language: 0 for language in LANGUAGES}
    selected: list[StudyRow] = []
    selected_ids: set[str] = set()

    def fits(group_id: str) -> bool:
        values = groups[group_id]
        values_by_language = vectors[group_id]
        if len(selected) + len(values) > budget:
            return False
        return not any(
            counts[language] + values_by_language[language] > targets[language]
            for language in LANGUAGES
        )

    def consume(group_id: str) -> None:
        values = groups[group_id]
        values_by_language = vectors[group_id]
        selected.extend(values)
        selected_ids.add(group_id)
        for language in LANGUAGES:
            counts[language] += values_by_language[language]

    # First consume groups that are internally balanced across all languages.
    balanced_ids = [
        group_id
        for group_id, values in vectors.items()
        if len(set(values.values())) == 1 and next(iter(values.values())) > 0
    ]
    for group_id in sorted(balanced_ids, key=lambda value: (-len(groups[value]), stable(value))):
        if fits(group_id):
            consume(group_id)

    # Bucket the remaining groups by their nine-language count vector. At each
    # step choose the vector that most reduces squared language deficit per row.
    # This avoids the English-heavy tail produced by plain random group order.
    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for group_id in groups:
        if group_id not in selected_ids:
            signature = tuple(vectors[group_id][language] for language in LANGUAGES)
            buckets[signature].append(group_id)
    for values in buckets.values():
        values.sort(key=stable, reverse=True)

    while len(selected) < budget:
        deficits = {language: targets[language] - counts[language] for language in LANGUAGES}
        choices: list[tuple[float, int, str, tuple[int, ...]]] = []
        for signature, ids in buckets.items():
            if not ids:
                continue
            group_id = ids[-1]
            if not fits(group_id):
                continue
            size = sum(signature)
            improvement = sum(
                deficits[language] ** 2 - (deficits[language] - signature[index]) ** 2
                for index, language in enumerate(LANGUAGES)
            )
            choices.append((improvement / size, size, stable(group_id), signature))
        if not choices:
            break
        _, _, _, signature = max(choices)
        consume(buckets[signature].pop())
    shortfall = budget - len(selected)
    permitted_shortfall = max(200, math.ceil(0.001 * budget)) if allow_tiny_shortfall else 0
    if shortfall > permitted_shortfall:
        remaining = {language: targets[language] - counts[language] for language in LANGUAGES}
        raise ValueError(
            f"Cannot satisfy complete-group balanced Nemotron budget={budget}; "
            f"selected={len(selected)}, remaining_by_language={remaining}"
        )
    return selected, selected_ids


def language_balance(rows: list[StudyRow]) -> dict[str, Any]:
    counts = {language: 0 for language in LANGUAGES}
    for row in rows:
        if row.language in counts:
            counts[row.language] += 1
    if not rows:
        return {"counts": counts, "expected_per_language": 0.0, "maximum_relative_deviation": 0.0}
    expected = len(rows) / len(LANGUAGES)
    maximum = max(abs(value - expected) / expected for value in counts.values())
    return {
        "counts": counts,
        "expected_per_language": expected,
        "maximum_relative_deviation": maximum,
    }


def clean_cross_source(
    nemo: list[StudyRow], wild: list[StudyRow]
) -> tuple[list[StudyRow], list[StudyRow], dict[str, Any]]:
    nemo_hashes: dict[str, list[StudyRow]] = defaultdict(list)
    wild_hashes: dict[str, list[StudyRow]] = defaultdict(list)
    for row in nemo:
        nemo_hashes[row.exact_hash].append(row)
    for row in wild:
        wild_hashes[row.exact_hash].append(row)
    shared = set(nemo_hashes) & set(wild_hashes)
    conflicts = {
        value
        for value in shared
        if {row.label for row in nemo_hashes[value]} != {row.label for row in wild_hashes[value]}
    }
    nemo_groups = {row.group_id for value in shared for row in nemo_hashes[value]}
    wild_groups = {row.group_id for value in shared for row in wild_hashes[value]}
    clean_nemo = [row for row in nemo if row.group_id not in nemo_groups]
    clean_wild = [row for row in wild if row.group_id not in wild_groups]
    return clean_nemo, clean_wild, {
        "policy": "drop every complete semantic group touching a cross-source exact-content overlap",
        "cross_source_exact_hashes": len(shared),
        "cross_source_conflicting_hashes": len(conflicts),
        "excluded_semantic_groups": {"nemotron": len(nemo_groups), "wildguard": len(wild_groups)},
        "removed_rows": {"nemotron": len(nemo) - len(clean_nemo), "wildguard": len(wild) - len(clean_wild)},
    }


def manifest_is_compatible(
    output: Path,
    ratio: dict[str, int],
    selection_sha256: str,
    source_paths: dict[str, str],
) -> dict[str, Any] | None:
    manifest_path = output / "manifest.json"
    if not output.exists() or not any(output.iterdir()):
        return None
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing non-empty output without manifest: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selected_ratio") != ratio:
        raise FileExistsError(f"Existing output has another ratio: {output}")
    if manifest.get("selection_sha256") != selection_sha256:
        raise FileExistsError(f"Existing output was built from another selection report: {output}")
    if manifest.get("source_paths") != source_paths:
        raise FileExistsError(f"Existing output was built from other raw-source paths: {output}")
    for split in ("train", "validation"):
        path = output / f"{split}.jsonl"
        expected = str(((manifest.get("splits") or {}).get(split) or {}).get("sha256") or "")
        from core.stage2_data import file_hash

        if not path.is_file() or not expected or file_hash(path) != expected:
            raise FileExistsError(f"Existing output split is incomplete or changed: {path}")
    report = validate_natural_arm(output)
    if not report["valid"]:
        raise FileExistsError(f"Existing output failed validation: {report['errors'][:5]}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the selected scaled Nemotron/WildGuard training mixture fully offline.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("runs-source-study-mixtures/qwen3_4b/ratio_selection.json"),
    )
    parser.add_argument("--nemotron-root", type=Path, required=True)
    parser.add_argument("--wildguard-path", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=Path("work/benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("work/source-study-scaled"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--language-balance-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    ratio = {key: int(value) for key, value in selection["winner_ratio"].items()}
    requested = {key: int(value) for key, value in selection["requested_scaled_train_rows"].items()}
    arm = f"nemotron{ratio['nemotron']}_wildguard{ratio['wildguard']}_scaled"
    output = args.output_root / arm
    selection_sha256 = __import__("hashlib").sha256(args.selection.read_bytes()).hexdigest()
    source_paths = {
        "nemotron": str(args.nemotron_root.resolve()),
        "wildguard": str(args.wildguard_path.resolve()),
    }
    existing = manifest_is_compatible(output, ratio, selection_sha256, source_paths)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    if not args.nemotron_root.is_dir():
        raise FileNotFoundError(f"Nemotron raw root missing: {args.nemotron_root}")
    if not args.wildguard_path.exists():
        raise FileNotFoundError(f"WildGuard raw source missing: {args.wildguard_path}")
    leakage = benchmark_hashes(args.benchmark_root)
    nemo, nemo_canonical = canonicalize(_v3_natural_rows(args.nemotron_root), leakage)
    wild, wild_canonical = canonicalize(_wildguard_natural_rows(args.wildguard_path), leakage)

    # Preserve Nemotron's official split and remove any semantic ID crossing it.
    nemo_train_groups = set(row.group_id for row in nemo if row.pool == "train")
    nemo_validation_groups = set(row.group_id for row in nemo if row.pool == "validation")
    official_cross = nemo_train_groups & nemo_validation_groups
    if official_cross:
        nemo = [row for row in nemo if row.group_id not in official_cross]

    if ratio["nemotron"] and ratio["wildguard"]:
        nemo, wild, cross_audit = clean_cross_source(nemo, wild)
    else:
        cross_audit = {
            "policy": "not applied because the selected recipe contains only one source",
            "cross_source_exact_hashes": None,
            "cross_source_conflicting_hashes": None,
            "excluded_semantic_groups": {"nemotron": 0, "wildguard": 0},
            "removed_rows": {"nemotron": 0, "wildguard": 0},
        }
    nemo_train_pool = [row for row in nemo if row.pool == "train"]
    nemo_validation_pool = [row for row in nemo if row.pool == "validation"]

    validation_total = 1_000
    n_val = round(validation_total * ratio["nemotron"] / 100)
    w_val = validation_total - n_val
    selected_n_val, _ = select_balanced_nemotron(
        nemo_validation_pool, n_val, seed=args.seed, split="validation"
    )
    selected_w_val, wild_val_ids = select_complete_groups(
        wild, w_val, seed=args.seed, source="wildguard", split="validation"
    )
    wild_train_pool = [row for row in wild if row.group_id not in wild_val_ids]

    n_train = requested["nemotron"]
    w_train = requested["wildguard"]
    nemo_language_capacity = Counter(row.language for row in nemo_train_pool)
    balanced_nemo_cap = 9 * math.floor(
        0.99 * min(nemo_language_capacity.get(language, 0) for language in LANGUAGES)
    )
    # For a pure-Nemotron winner, 420k is a raw cap, not permission to restore
    # benchmark overlap, duplicates or invalid rows. Use every eligible complete
    # train group up to that cap. Mixed ratios are fitted after cleaning so an
    # incomplete WildGuard pool cannot make the whole build fail.
    if ratio["wildguard"] == 0:
        n_train = min(n_train, balanced_nemo_cap, len(nemo_train_pool))
    elif ratio["nemotron"] == 0:
        w_train = min(w_train, len(wild_train_pool))
    else:
        fitted = fit_mixed_budgets(
            requested_nemotron=n_train,
            requested_wildguard=w_train,
            nemotron_percent=ratio["nemotron"],
            wildguard_percent=ratio["wildguard"],
            nemo_capacity=len(nemo_train_pool),
            wild_capacity=len(wild_train_pool),
            balanced_nemo_capacity=balanced_nemo_cap,
        )
        n_train, w_train = fitted["nemotron"], fitted["wildguard"]
    planned_n_train = n_train
    selected_n_train, _ = select_balanced_nemotron(
        nemo_train_pool, n_train, seed=args.seed, split="train", allow_tiny_shortfall=True
    )
    n_train = len(selected_n_train)
    if n_train != planned_n_train and ratio["nemotron"] and ratio["wildguard"]:
        w_train = round(n_train * ratio["wildguard"] / ratio["nemotron"])
    selected_w_train, _ = select_complete_groups(
        wild_train_pool, w_train, seed=args.seed, source="wildguard", split="train"
    )

    n_balance = language_balance(selected_n_train)
    if selected_n_train and n_balance["maximum_relative_deviation"] > args.language_balance_tolerance:
        raise RuntimeError(
            "Nemotron language balance exceeds tolerance: "
            f"{n_balance['maximum_relative_deviation']:.4%} > {args.language_balance_tolerance:.4%}; "
            f"counts={n_balance['counts']}"
        )

    train = _render_binary(selected_n_train + selected_w_train, "train", args.seed)
    validation = _render_binary(selected_n_val + selected_w_val, "validation", args.seed)
    train.sort(key=lambda row: stable_hash(f"scaled-output:{args.seed}:train:{row['example_id']}"))
    validation.sort(key=lambda row: stable_hash(f"scaled-output:{args.seed}:validation:{row['example_id']}"))
    output.mkdir(parents=True, exist_ok=False)
    splits = {
        "train": _write_jsonl(output / "train.jsonl", train),
        "validation": _write_jsonl(output / "validation.jsonl", validation),
    }
    manifest = {
        "schema_version": 1,
        "study": "selected_scaled_nemotron_wildguard",
        "source": arm,
        "seed": args.seed,
        "selected_ratio": ratio,
        "requested_train_rows": requested,
        "actual_train_rows": {"nemotron": n_train, "wildguard": w_train},
        "train_examples": len(train),
        "validation_examples": len(validation),
        "source_paths": source_paths,
        "selection_sha256": selection_sha256,
        "rendering": {"label_contract": "nemotron_binary_json_taxonomy_off", "thinking_mode": "no_think"},
        "sampling": "stable SHA-256 sampling of complete semantic groups after benchmark and cross-source cleaning",
        "nemotron_language_balance": n_balance,
        "cross_source_cleaning": cross_audit,
        "official_split_group_conflicts_removed": len(official_cross),
        "canonicalization": {"nemotron": nemo_canonical, "wildguard": wild_canonical},
        "eligible_distribution": {
            "nemotron_train": _distribution(nemo_train_pool),
            "wildguard_train": _distribution(wild_train_pool),
        },
        "selected_distribution": {
            "nemotron_train": _distribution(selected_n_train),
            "wildguard_train": _distribution(selected_w_train),
        },
        "splits": splits,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = validate_natural_arm(output)
    if not report["valid"]:
        raise RuntimeError(f"Scaled mixture validation failed: {report['errors'][:10]}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
