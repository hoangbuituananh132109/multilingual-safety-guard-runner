from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .prompt import sea_guard_instruction
    from .source_study_data import (
        SOURCE_REVISIONS,
        StudyRow,
        _parquet_batches,
        benchmark_hashes,
        canonicalize,
        content_hash,
        default_source_paths,
        stable_hash,
    )
    from .stage2_data import Normalized, file_hash, read_jsonl, render
except ImportError:
    from prompt import sea_guard_instruction
    from source_study_data import (
        SOURCE_REVISIONS,
        StudyRow,
        _parquet_batches,
        benchmark_hashes,
        canonicalize,
        content_hash,
        default_source_paths,
        stable_hash,
    )
    from stage2_data import Normalized, file_hash, read_jsonl, render


THREE_WAY_LABELS = {"safe", "sensitive", "unsafe"}
TRAIN_BUDGET = 80_000
VALIDATION_BUDGET = 1_000
SMOKE_TRAIN_BUDGET = 64
SMOKE_VALIDATION_BUDGET = 16

SOURCES = (
    "nemotron_v3_9lang_natural",
    "wildguardtrain_en_natural",
    "sea_cultural_vi_natural",
    "sea_cultural_bilingual_id50_natural",
)


def _v3_natural_rows(root: Path) -> Iterator[StudyRow]:
    """Keep one task per released V3 row; do not synthesize a second P task."""
    languages = ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh")
    for pool, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        for language in languages:
            path = root / language / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            for raw in read_jsonl(path):
                prompt = str(raw.get("prompt") or "").strip()
                response_raw = raw.get("response")
                response = str(response_raw).strip() if response_raw is not None else None
                prompt_label = str(raw.get("prompt_label") or "").casefold()
                response_label = str(raw.get("response_label") or "").casefold() or None
                upstream_id = str(raw.get("id") or f"{language}-{raw['_line']}")
                if not prompt or prompt_label not in {"safe", "unsafe"}:
                    continue
                if response is not None and response_label not in {"safe", "unsafe"}:
                    continue
                yield StudyRow(
                    source="nemotron_v3_9lang",
                    source_id=f"{upstream_id}:{content_hash(prompt, response)[:12]}",
                    pool=pool,
                    group_id=f"nemotron-v3:{upstream_id}",
                    language=language,
                    prompt=prompt,
                    response=response,
                    prompt_label=prompt_label,
                    response_label=response_label if response is not None else None,
                    categories=[],
                    metadata={
                        "upstream_id": upstream_id,
                        "official_split": pool,
                        "violated_categories_raw": raw.get("violated_categories"),
                        "prompt_label_source": raw.get("prompt_label_source"),
                        "response_label_source": raw.get("response_label_source"),
                        "tag": raw.get("tag"),
                    },
                )


def _wildguard_natural_rows(path: Path) -> Iterator[StudyRow]:
    """Keep WildGuard's released P/PR record shape without task expansion."""
    if path.is_dir():
        candidates = sorted(path.rglob("wildguard_train.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No wildguard_train.parquet under {path}")
        raw_rows: Iterable[dict[str, Any]] = _parquet_batches(
            candidates,
            [
                "prompt",
                "response",
                "prompt_harm_label",
                "response_harm_label",
                "response_refusal_label",
                "adversarial",
                "subcategory",
            ],
        )
    elif path.suffix.casefold() == ".parquet":
        raw_rows = _parquet_batches(
            [path],
            [
                "prompt",
                "response",
                "prompt_harm_label",
                "response_harm_label",
                "response_refusal_label",
                "adversarial",
                "subcategory",
            ],
        )
    else:
        raw_rows = read_jsonl(path)
    label_map = {"harmful": "unsafe", "unharmful": "safe"}
    for raw in raw_rows:
        prompt = str(raw.get("prompt") or "").strip()
        response_raw = raw.get("response")
        response = str(response_raw).strip() if response_raw is not None else None
        prompt_label = label_map.get(str(raw.get("prompt_harm_label") or "").casefold())
        response_label = label_map.get(str(raw.get("response_harm_label") or "").casefold())
        if not prompt or prompt_label is None:
            continue
        if response is not None and response_label is None:
            continue
        prompt_group = content_hash(prompt, None)
        yield StudyRow(
            source="wildguardtrain_en",
            source_id=content_hash(prompt, response)[:24],
            pool="unsplit",
            group_id=f"wildguard-prompt:{prompt_group}",
            language="en",
            prompt=prompt,
            response=response,
            prompt_label=prompt_label,
            response_label=response_label if response is not None else None,
            categories=[],
            metadata={
                "response_refusal_label": raw.get("response_refusal_label"),
                "adversarial": raw.get("adversarial"),
                "subcategory": raw.get("subcategory"),
            },
        )


def _sea_three_way_rows(root: Path) -> Iterator[StudyRow]:
    paths = sorted(root.glob("train_refined_v1_qa_pass_only-*.parquet"))
    if not paths:
        paths = sorted(root.rglob("train_refined_v1_qa_pass_only-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No QA-pass-only SEA parquet shards under {root}")
    columns = [
        "id",
        "language",
        "topic",
        "quality_assurance",
        "prompt",
        "prompt_label",
        "prompt_risk_score",
        "response",
        "response_label",
        "response_risk_score",
        "response_model",
    ]
    language_map = {"english": "en", "vietnamese": "vi"}
    for raw in _parquet_batches(paths, columns):
        language_raw = str(raw.get("language") or "").casefold()
        language = language_map.get(language_raw)
        if language is None:
            continue
        if str(raw.get("quality_assurance") or "").casefold() != "pass":
            continue
        prompt = str(raw.get("prompt") or "").strip()
        response_raw = raw.get("response")
        response = str(response_raw).strip() if response_raw is not None else None
        prompt_label = str(raw.get("prompt_label") or "").casefold()
        response_label = str(raw.get("response_label") or "").casefold() or None
        upstream_id = str(raw.get("id") or content_hash(prompt, None)[:24])
        source_id = f"{upstream_id}:{language}:{content_hash(prompt, response)[:16]}"
        common = dict(
            source="sea_cultural",
            source_id=source_id,
            pool="unsplit",
            group_id=f"sea-cultural:{upstream_id}",
            language=language,
            prompt=prompt,
            prompt_label=prompt_label,
            categories=[],
            metadata={
                "upstream_id": upstream_id,
                "topic": raw.get("topic"),
                "prompt_risk_score": raw.get("prompt_risk_score"),
                "response_risk_score": raw.get("response_risk_score"),
                "response_model": raw.get("response_model"),
                "quality_assurance": raw.get("quality_assurance"),
            },
        )
        if prompt and response is None and prompt_label in THREE_WAY_LABELS:
            yield StudyRow(response=None, response_label=None, **common)
        if prompt and response and response_label in THREE_WAY_LABELS:
            yield StudyRow(response=response, response_label=response_label, **common)


def _group_rows(rows: Iterable[StudyRow]) -> dict[str, list[StudyRow]]:
    groups: dict[str, list[StudyRow]] = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    return dict(groups)


def _select_groups_exact(
    groups: dict[str, list[StudyRow]],
    budget: int,
    *,
    seed: int,
    source: str,
    split: str,
) -> tuple[list[StudyRow], set[str]]:
    """Select complete semantic groups with an exact row budget.

    Groups are traversed in a stable random order.  Near the boundary, groups
    that do not fit are skipped; size-one groups make all current 80k arms
    exactly attainable without splitting a prompt/upstream ID.
    """
    selected: list[StudyRow] = []
    selected_ids: set[str] = set()
    remaining = budget
    order = sorted(
        groups,
        key=lambda group_id: stable_hash(f"natural-group:{seed}:{source}:{split}:{group_id}"),
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
        available = sum(len(values) for values in groups.values())
        sizes = Counter(len(values) for values in groups.values())
        raise ValueError(
            f"Cannot reach exact {budget} rows for {source}/{split}; "
            f"remaining={remaining}, available={available}, group_sizes={dict(sorted(sizes.items()))}"
        )
    return selected, selected_ids


def _split_natural_groups(
    rows: list[StudyRow],
    source: str,
    seed: int,
    train_budget: int,
    validation_budget: int,
    *,
    official_split: bool,
) -> tuple[list[StudyRow], list[StudyRow], dict[str, Any]]:
    if official_split:
        train_groups = _group_rows(row for row in rows if row.pool == "train")
        validation_groups = _group_rows(row for row in rows if row.pool == "validation")
        overlap = set(train_groups) & set(validation_groups)
        for group_id in overlap:
            train_groups.pop(group_id, None)
            validation_groups.pop(group_id, None)
    else:
        all_groups = _group_rows(rows)
        validation, validation_ids = _select_groups_exact(
            all_groups,
            validation_budget,
            seed=seed,
            source=source,
            split="validation",
        )
        train_groups = {key: value for key, value in all_groups.items() if key not in validation_ids}
        train, train_ids = _select_groups_exact(
            train_groups,
            train_budget,
            seed=seed,
            source=source,
            split="train",
        )
        return train, validation, {
            "official_split_group_conflicts_removed": 0,
            "train_groups": len(train_ids),
            "validation_groups": len(validation_ids),
            "train_group_ids_sha256": stable_hash("\n".join(sorted(train_ids))),
            "validation_group_ids_sha256": stable_hash("\n".join(sorted(validation_ids))),
        }

    validation, validation_ids = _select_groups_exact(
        validation_groups,
        validation_budget,
        seed=seed,
        source=source,
        split="validation",
    )
    train, train_ids = _select_groups_exact(
        train_groups,
        train_budget,
        seed=seed,
        source=source,
        split="train",
    )
    if train_ids & validation_ids:
        raise AssertionError("Official train/validation semantic groups overlap after selection")
    return train, validation, {
        "official_split_group_conflicts_removed": len(overlap),
        "train_groups": len(train_ids),
        "validation_groups": len(validation_ids),
        "train_group_ids_sha256": stable_hash("\n".join(sorted(train_ids))),
        "validation_group_ids_sha256": stable_hash("\n".join(sorted(validation_ids))),
    }


def _sea_reference_split(
    rows: list[StudyRow],
    source: str,
    seed: int,
    train_budget: int,
    validation_budget: int,
) -> tuple[list[StudyRow], list[StudyRow], dict[str, Any]]:
    by_language = {language: _group_rows(row for row in rows if row.language == language) for language in ("en", "vi")}
    paired_ids = set(by_language["en"]) & set(by_language["vi"])
    vi_reference = [row for group_id in paired_ids for row in by_language["vi"][group_id]]
    train_vi, validation_vi, audit = _split_natural_groups(
        vi_reference,
        "sea_cultural_shared_vi_reference",
        seed,
        train_budget,
        validation_budget,
        official_split=False,
    )
    audit.update(
        {
            "requested_arm": source,
            "paired_candidate_groups": len(paired_ids),
            "english_only_groups_excluded": len(set(by_language["en"]) - paired_ids),
            "vietnamese_only_groups_excluded": len(set(by_language["vi"]) - paired_ids),
            "selection_reference_language": "vi",
        }
    )
    return train_vi, validation_vi, audit


def _switch_half_ids_to_english(
    vi_rows: list[StudyRow],
    all_rows: list[StudyRow],
    *,
    seed: int,
    split: str,
) -> tuple[list[StudyRow], dict[str, Any]]:
    selected_ids = sorted({row.group_id for row in vi_rows})
    random_order = sorted(
        selected_ids,
        key=lambda group_id: stable_hash(f"sea-id-language:{seed}:{split}:{group_id}"),
    )
    english_ids = set(random_order[: len(random_order) // 2])
    selected_id_set = set(selected_ids)
    candidates = [
        row
        for row in all_rows
        if row.group_id in selected_id_set
        and ((row.group_id in english_ids and row.language == "en") or (row.group_id not in english_ids and row.language == "vi"))
    ]
    present = {row.group_id for row in candidates}
    missing = selected_id_set - present
    if missing:
        raise ValueError(f"SEA bilingual switch lost {len(missing)} selected IDs")
    return candidates, {
        "semantic_ids": len(selected_ids),
        "english_ids": len(english_ids),
        "vietnamese_ids": len(selected_ids) - len(english_ids),
        "rows_after_language_switch": len(candidates),
    }


def _distribution(rows: list[StudyRow]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        counts["rows"] += 1
        counts[f"language:{row.language}"] += 1
        counts[f"view:{row.view}"] += 1
        counts[f"label:{row.label}"] += 1
        counts[f"cell:{row.view}:{row.label}"] += 1
        response_model = str(row.metadata.get("response_model") or "")
        if response_model:
            counts[f"response_model:{response_model}"] += 1
        groups[row.language].add(row.group_id)
    counts["semantic_groups"] = len({row.group_id for row in rows})
    for language, group_ids in groups.items():
        counts[f"semantic_groups:{language}"] = len(group_ids)
    return dict(sorted(counts.items()))


def _render_binary(rows: list[StudyRow], split: str, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        normalized = Normalized(
            source=row.source,
            source_id=row.source_id,
            split=split,
            language=row.language,
            prompt=row.prompt,
            response=row.response,
            prompt_label=row.prompt_label,
            response_label=row.response_label,
            categories=row.categories,
            force_taxonomy_off=True,
            metadata={**row.metadata, "study_group_id": row.group_id},
        )
        output.append(render(normalized, seed, taxonomy_mode="off", thinking_mode="no_think"))
    return sorted(output, key=lambda value: stable_hash(f"natural-output:{seed}:{value['example_id']}"))


def _render_sea(rows: list[StudyRow], split: str, seed: int, source_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        example_id = stable_hash(f"{source_name}:{split}:{row.identity}")[:24]
        output.append(
            {
                "example_id": example_id,
                "dataset_source": source_name,
                "source_id": row.source_id,
                "split": split,
                "language": row.language,
                "view": row.view,
                "prompt": row.prompt,
                "response": row.response,
                "prompt_safety_label": row.prompt_label,
                "safety_label": row.label,
                "categories": [],
                "taxonomy_mode": "not_applicable",
                "thinking_mode": "no_think",
                "instruction": sea_guard_instruction(row.prompt, row.response),
                "target": row.label,
                "target_json": None,
                "content_sha256": row.exact_hash,
                "metadata": {
                    **row.metadata,
                    "study_group_id": row.group_id,
                    "label_space": "safe/sensitive/unsafe",
                    "binary_evaluation_mapping": "P sensitive->safe; PR sensitive->unsafe",
                },
            }
        )
    return sorted(output, key=lambda value: stable_hash(f"natural-output:{seed}:{value['example_id']}"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            language = str(row["language"])
            counts["examples"] += 1
            counts[f"language:{language}"] += 1
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
            counts[f"cell:{row['view']}:{row['safety_label']}"] += 1
            groups[language].add(str((row.get("metadata") or {}).get("study_group_id")))
    counts["semantic_groups"] = len(set().union(*groups.values())) if groups else 0
    for language, group_ids in groups.items():
        counts[f"semantic_groups:{language}"] = len(group_ids)
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        **dict(sorted(counts.items())),
    }


def build_natural_arm(
    source: str,
    source_path: Path,
    benchmark_root: Path,
    output_dir: Path,
    *,
    seed: int = 3407,
    smoke: bool = False,
    prepared_candidates: list[StudyRow] | None = None,
    prepared_canonical_audit: dict[str, Any] | None = None,
    prepared_benchmark_hash_count: int | None = None,
) -> dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"Unknown natural study source: {source}")
    train_budget = SMOKE_TRAIN_BUDGET if smoke else TRAIN_BUDGET
    validation_budget = SMOKE_VALIDATION_BUDGET if smoke else VALIDATION_BUDGET
    leakage_hashes: set[str] | None = None
    benchmark_hash_count = prepared_benchmark_hash_count
    if prepared_candidates is not None:
        if prepared_canonical_audit is None or prepared_benchmark_hash_count is None:
            raise ValueError("Prepared candidates require their canonical audit and benchmark hash count")
        candidates = prepared_candidates
        canonical_audit = prepared_canonical_audit
    else:
        leakage_hashes = benchmark_hashes(benchmark_root)
        benchmark_hash_count = len(leakage_hashes)

    if source == "nemotron_v3_9lang_natural":
        if prepared_candidates is None:
            assert leakage_hashes is not None
            candidates, canonical_audit = canonicalize(_v3_natural_rows(source_path), leakage_hashes)
        train, validation, selection_audit = _split_natural_groups(
            candidates, source, seed, train_budget, validation_budget, official_split=True
        )
        label_contract = "nemotron_binary_json_taxonomy_off"
        rendered = {
            "train": _render_binary(train, "train", seed),
            "validation": _render_binary(validation, "validation", seed),
        }
    elif source == "wildguardtrain_en_natural":
        if prepared_candidates is None:
            assert leakage_hashes is not None
            candidates, canonical_audit = canonicalize(_wildguard_natural_rows(source_path), leakage_hashes)
        train, validation, selection_audit = _split_natural_groups(
            candidates, source, seed, train_budget, validation_budget, official_split=False
        )
        label_contract = "nemotron_binary_json_taxonomy_off"
        rendered = {
            "train": _render_binary(train, "train", seed),
            "validation": _render_binary(validation, "validation", seed),
        }
    else:
        if prepared_candidates is None:
            assert leakage_hashes is not None
            candidates, canonical_audit = canonicalize(_sea_three_way_rows(source_path), leakage_hashes)
        train_vi, validation_vi, selection_audit = _sea_reference_split(
            candidates, source, seed, train_budget, validation_budget
        )
        if source == "sea_cultural_bilingual_id50_natural":
            train, train_language_audit = _switch_half_ids_to_english(
                train_vi, candidates, seed=seed, split="train"
            )
            validation, validation_language_audit = _switch_half_ids_to_english(
                validation_vi, candidates, seed=seed, split="validation"
            )
            selection_audit["language_assignment"] = {
                "method": "stable SHA-256 random half of selected semantic IDs switched from Vietnamese to English",
                "train": train_language_audit,
                "validation": validation_language_audit,
            }
        else:
            train, validation = train_vi, validation_vi
        label_contract = "sea_guard_three_way_single_label"
        rendered = {
            "train": _render_sea(train, "train", seed, source),
            "validation": _render_sea(validation, "validation", seed, source),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in rendered.items()
    }
    manifest = {
        "schema_version": 2,
        "study": "natural_source_recipe_smoke" if smoke else "natural_source_recipe_80k_target",
        "source": source,
        "source_path": str(source_path.resolve()),
        "source_revisions": SOURCE_REVISIONS,
        "seed": seed,
        "train_target_rows": train_budget,
        "validation_target_rows": validation_budget,
        "train_examples": len(rendered["train"]),
        "validation_examples": len(rendered["validation"]),
        "sampling": {
            "method": "stable SHA-256 sampling of complete semantic groups; no view/label/model/topic rebalancing",
            "sea_policy": "select semantic IDs using Vietnamese QA-pass rows; bilingual arm switches exactly half the selected IDs to English and retains every surviving row for that ID/language",
        },
        "rendering": {
            "label_contract": label_contract,
            "thinking_mode": "no_think",
            "binary_benchmark_mapping_for_sea": "P sensitive->safe; PR sensitive->unsafe",
        },
        "benchmark_root": str(benchmark_root.resolve()),
        "benchmark_hashes": benchmark_hash_count,
        "canonicalization": canonical_audit,
        "candidate_distribution": _distribution(candidates),
        "selected_distribution": {
            "train": _distribution(train),
            "validation": _distribution(validation),
        },
        "selection": selection_audit,
        "splits": splits,
        "license_note": (
            "CC-BY-4.0; preserve attribution."
            if source.startswith("nemotron")
            else "ODC-BY plus AI2 gated Responsible Use acceptance; do not redistribute gated rows."
            if source.startswith("wildguard")
            else "Gated upstream release; do not publicly redistribute rows without explicit permission."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = validate_natural_arm(output_dir)
    if report["errors"]:
        raise ValueError("Built natural arm failed validation: " + "; ".join(report["errors"][:10]))
    return manifest


def build_sea_pair(
    source_path: Path,
    benchmark_root: Path,
    output_root: Path,
    *,
    seed: int = 3407,
    smoke: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build VI and ID-50 bilingual SEA arms from one canonical source scan."""
    leakage_hashes = benchmark_hashes(benchmark_root)
    candidates, canonical_audit = canonicalize(_sea_three_way_rows(source_path), leakage_hashes)
    manifests: dict[str, dict[str, Any]] = {}
    for source in ("sea_cultural_vi_natural", "sea_cultural_bilingual_id50_natural"):
        manifests[source] = build_natural_arm(
            source,
            source_path,
            benchmark_root,
            output_root / source,
            seed=seed,
            smoke=smoke,
            prepared_candidates=candidates,
            prepared_canonical_audit=canonical_audit,
            prepared_benchmark_hash_count=len(leakage_hashes),
        )
    return manifests


def _read_rendered_groups(directory: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(directory / f"{split}.jsonl"):
        row.pop("_line", None)
        group_id = str((row.get("metadata") or {}).get("study_group_id") or "")
        if not group_id:
            raise ValueError(f"Missing study_group_id in {directory}/{split}.jsonl")
        groups[group_id].append(row)
    return dict(groups)


def _small_group_sample(
    groups: dict[str, list[dict[str, Any]]], budget: int, *, seed: int, name: str
) -> tuple[list[dict[str, Any]], set[str]]:
    output: list[dict[str, Any]] = []
    selected: set[str] = set()
    for group_id in sorted(groups, key=lambda value: stable_hash(f"derived-smoke:{seed}:{name}:{value}")):
        output.extend(groups[group_id])
        selected.add(group_id)
        if len(output) >= budget:
            break
    if len(output) < budget:
        raise ValueError(f"Not enough rows to derive smoke for {name}: {len(output)} < {budget}")
    return output, selected


def _balanced_bilingual_smoke_ids(
    groups: dict[str, list[dict[str, Any]]], budget: int, *, seed: int, split: str
) -> set[str]:
    by_language: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in groups.items():
        languages = {str(row.get("language") or "") for row in rows}
        if len(languages) != 1:
            raise ValueError(f"Bilingual full arm mixes languages inside {split}/{group_id}")
        by_language[next(iter(languages))].append(group_id)
    for language in ("en", "vi"):
        by_language[language].sort(
            key=lambda value: stable_hash(f"derived-smoke-sea:{seed}:{split}:{language}:{value}")
        )
    selected: set[str] = set()
    rows = 0
    for en_group, vi_group in zip(by_language["en"], by_language["vi"]):
        selected.update((en_group, vi_group))
        rows += len(groups[en_group]) + len(groups[vi_group])
        if rows >= budget:
            return selected
    raise ValueError(f"Not enough balanced bilingual rows for {split} smoke")


def derive_smoke_tree(source_root: Path, output_root: Path, *, seed: int = 3407) -> dict[str, Any]:
    """Derive tiny GPU smoke inputs from already validated full arms."""
    for arm in SOURCES:
        report = validate_natural_arm(source_root / arm)
        if not report["valid"]:
            raise ValueError(f"Cannot derive smoke from invalid arm {arm}: {report['errors'][:5]}")

    selected_by_arm: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for arm in ("nemotron_v3_9lang_natural", "wildguardtrain_en_natural"):
        for split, budget in (("train", SMOKE_TRAIN_BUDGET), ("validation", SMOKE_VALIDATION_BUDGET)):
            groups = _read_rendered_groups(source_root / arm, split)
            rows, _ = _small_group_sample(groups, budget, seed=seed, name=f"{arm}:{split}")
            selected_by_arm[arm][split] = rows

    vi_arm = "sea_cultural_vi_natural"
    bilingual_arm = "sea_cultural_bilingual_id50_natural"
    for split, budget in (("train", SMOKE_TRAIN_BUDGET), ("validation", SMOKE_VALIDATION_BUDGET)):
        bilingual_groups = _read_rendered_groups(source_root / bilingual_arm, split)
        selected_ids = _balanced_bilingual_smoke_ids(bilingual_groups, budget, seed=seed, split=split)
        selected_by_arm[bilingual_arm][split] = [
            row for group_id in selected_ids for row in bilingual_groups[group_id]
        ]
        vi_groups = _read_rendered_groups(source_root / vi_arm, split)
        missing = selected_ids - set(vi_groups)
        if missing:
            raise ValueError(f"VI full arm is missing {len(missing)} bilingual smoke IDs in {split}")
        selected_by_arm[vi_arm][split] = [row for group_id in selected_ids for row in vi_groups[group_id]]

    manifests: dict[str, Any] = {}
    for arm in SOURCES:
        directory = output_root / arm
        directory.mkdir(parents=True, exist_ok=True)
        splits = {
            split: _write_jsonl(directory / f"{split}.jsonl", selected_by_arm[arm][split])
            for split in ("train", "validation")
        }
        parent_manifest = json.loads((source_root / arm / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            **parent_manifest,
            "study": "natural_source_recipe_derived_gpu_smoke",
            "derived_from": str((source_root / arm).resolve()),
            "train_target_rows": SMOKE_TRAIN_BUDGET,
            "validation_target_rows": SMOKE_VALIDATION_BUDGET,
            "train_examples": len(selected_by_arm[arm]["train"]),
            "validation_examples": len(selected_by_arm[arm]["validation"]),
            "selection": {
                "method": "complete semantic groups deterministically selected from the validated full arm",
                "seed": seed,
            },
            "selected_distribution": {
                split: {
                    key: value
                    for key, value in splits[split].items()
                    if key not in {"path", "sha256", "bytes"}
                }
                for split in ("train", "validation")
            },
            "splits": splits,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = validate_natural_arm(directory)
        if not report["valid"]:
            raise ValueError(f"Derived smoke arm {arm} is invalid: {report['errors'][:5]}")
        manifests[arm] = {
            "train_examples": manifest["train_examples"],
            "validation_examples": manifest["validation_examples"],
            "train_sha256": splits["train"]["sha256"],
            "validation_sha256": splits["validation"]["sha256"],
        }
    return manifests


def validate_natural_arm(directory: Path) -> dict[str, Any]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    group_languages: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_examples: set[str] = set()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = str(manifest.get("source") or "")
    sea = source.startswith("sea_cultural")
    expected = {
        "train": int(manifest.get("train_examples", -1)),
        "validation": int(manifest.get("validation_examples", -1)),
    }
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        for row in read_jsonl(path):
            line = row["_line"]
            counts[f"split:{split}"] += 1
            counts[f"split:{split}:language:{row.get('language')}"] += 1
            counts[f"split:{split}:cell:{row.get('view')}:{row.get('safety_label')}"] += 1
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in seen_examples:
                errors.append(f"{path}:{line}: missing or duplicate example_id")
            seen_examples.add(example_id)
            if row.get("thinking_mode") != "no_think":
                errors.append(f"{path}:{line}: expected no_think")
            view = str(row.get("view") or "")
            if view not in {"P", "PR"}:
                errors.append(f"{path}:{line}: invalid view={view}")
            label = str(row.get("safety_label") or "")
            target = str(row.get("target") or "")
            if sea:
                if row.get("taxonomy_mode") != "not_applicable":
                    errors.append(f"{path}:{line}: SEA taxonomy_mode must be not_applicable")
                if label not in THREE_WAY_LABELS or target != label:
                    errors.append(f"{path}:{line}: invalid SEA three-way target")
                instruction = str(row.get("instruction") or "")
                if not all(value in instruction for value in ("safe", "sensitive", "unsafe")):
                    errors.append(f"{path}:{line}: SEA instruction lacks the three-way label contract")
            else:
                if row.get("taxonomy_mode") != "off" or label not in {"safe", "unsafe"}:
                    errors.append(f"{path}:{line}: invalid binary taxonomy-off contract")
                try:
                    payload = json.loads(target)
                    key = "Response Safety" if view == "PR" else "User Safety"
                    if str(payload.get(key) or "") != label:
                        errors.append(f"{path}:{line}: binary JSON target does not match label")
                except (json.JSONDecodeError, AttributeError):
                    errors.append(f"{path}:{line}: binary target is not a JSON object")
            metadata = row.get("metadata") or {}
            group_id = str(metadata.get("study_group_id") or "")
            if not group_id:
                errors.append(f"{path}:{line}: missing study_group_id")
            split_groups[split].add(group_id)
            group_languages[(split, group_id)].add(str(row.get("language") or ""))
            split_content[split].add(str(row.get("content_sha256") or ""))
        if counts[f"split:{split}"] != expected[split]:
            errors.append(f"{split} expected {expected[split]}, got {counts[f'split:{split}']}")
        expected_sha256 = str(((manifest.get("splits") or {}).get(split) or {}).get("sha256") or "")
        if expected_sha256 and file_hash(path) != expected_sha256:
            errors.append(f"{split} SHA-256 does not match manifest")

    group_overlap = split_groups["train"] & split_groups["validation"]
    content_overlap = split_content["train"] & split_content["validation"]
    if group_overlap:
        errors.append(f"{len(group_overlap)} semantic groups cross train/validation")
    if content_overlap:
        errors.append(f"{len(content_overlap)} exact content hashes cross train/validation")
    if source == "sea_cultural_bilingual_id50_natural":
        for (split, group_id), languages in group_languages.items():
            if len(languages) != 1:
                errors.append(f"{split}/{group_id}: bilingual semantic ID crosses languages")
        for split in ("train", "validation"):
            en_groups = {
                group_id for (key_split, group_id), languages in group_languages.items()
                if key_split == split and languages == {"en"}
            }
            vi_groups = {
                group_id for (key_split, group_id), languages in group_languages.items()
                if key_split == split and languages == {"vi"}
            }
            if abs(len(en_groups) - len(vi_groups)) > 1:
                errors.append(f"{split}: ID language assignment is not 50/50 ({len(en_groups)} en, {len(vi_groups)} vi)")
    report = {
        "directory": str(directory),
        "source": source,
        "counts": dict(sorted(counts.items())),
        "group_overlap": len(group_overlap),
        "content_overlap": len(content_overlap),
        "errors": errors,
        "valid": not errors,
    }
    (directory / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def natural_source_path(source: str, repo_root: Path) -> Path:
    defaults = default_source_paths(repo_root)
    if source.startswith("nemotron"):
        return defaults["nemotron_v3_9lang"]
    if source.startswith("wildguard"):
        return defaults["wildguardtrain_en"]
    return defaults["sea_cultural_vi"]
