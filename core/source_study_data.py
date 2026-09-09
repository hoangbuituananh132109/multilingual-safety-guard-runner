from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .stage2_data import Normalized, file_hash, read_jsonl, render, split_categories
except ImportError:
    from stage2_data import Normalized, file_hash, read_jsonl, render, split_categories


LANGUAGES = ("en", "ar", "de", "es", "fr", "hi", "ja", "th", "zh")
VALID_LABELS = {"safe", "unsafe"}
TRAIN_TARGETS = {
    ("P", "safe"): 21_501,
    ("P", "unsafe"): 23_102,
    ("PR", "safe"): 27_594,
    ("PR", "unsafe"): 7_803,
}
VALIDATION_TARGETS = {
    ("P", "safe"): 269,
    ("P", "unsafe"): 289,
    ("PR", "safe"): 345,
    ("PR", "unsafe"): 97,
}
SMOKE_TRAIN_TARGETS = {
    ("P", "safe"): 8,
    ("P", "unsafe"): 8,
    ("PR", "safe"): 8,
    ("PR", "unsafe"): 8,
}
SMOKE_VALIDATION_TARGETS = {
    ("P", "safe"): 2,
    ("P", "unsafe"): 2,
    ("PR", "safe"): 2,
    ("PR", "unsafe"): 2,
}
SOURCE_REVISIONS = {
    "nemotron_v3": "a3f7ecb3433d1933701a83f18de16c36934a7f51",
    "wildguardmix": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
    "sea_safeguard_train_cultural_v3": "2a51a837855ff7ae7d383fa4af6fa11c2cee8a92",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def content_hash(prompt: Any, response: Any = None) -> str:
    return stable_hash(normalized_text(prompt) + "\n<response>\n" + normalized_text(response))


def loose_content_hash(prompt: Any, response: Any = None) -> str:
    def loose(value: Any) -> str:
        return re.sub(r"[^\w]+", " ", normalized_text(value), flags=re.UNICODE).strip()

    return stable_hash(loose(prompt) + "\n<response>\n" + loose(response))


@dataclass
class StudyRow:
    source: str
    source_id: str
    pool: str
    group_id: str
    language: str
    prompt: str
    response: str | None
    prompt_label: str
    response_label: str | None
    categories: list[str]
    metadata: dict[str, Any]

    @property
    def view(self) -> str:
        return "PR" if self.response is not None else "P"

    @property
    def label(self) -> str:
        return self.response_label if self.response is not None else self.prompt_label

    @property
    def exact_hash(self) -> str:
        return content_hash(self.prompt, self.response)

    @property
    def loose_hash(self) -> str:
        return loose_content_hash(self.prompt, self.response)

    @property
    def prompt_hash(self) -> str:
        return content_hash(self.prompt, None)

    @property
    def prompt_loose_hash(self) -> str:
        return loose_content_hash(self.prompt, None)

    @property
    def identity(self) -> str:
        return f"{self.source}:{self.source_id}:{self.language}:{self.view}:{self.exact_hash}"


def _v3_rows(root: Path) -> Iterator[StudyRow]:
    for pool, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        for language in LANGUAGES:
            path = root / language / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            for raw in read_jsonl(path):
                prompt = str(raw.get("prompt") or "").strip()
                response = raw.get("response")
                response = str(response).strip() if response is not None else None
                prompt_label = str(raw.get("prompt_label") or "").casefold()
                response_label = str(raw.get("response_label") or "").casefold() or None
                upstream_id = str(raw.get("id") or f"{language}-{raw['_line']}")
                pair_hash = content_hash(prompt, response)
                source_id = f"{upstream_id}:{pair_hash[:12]}"
                common = dict(
                    source="nemotron_v3_9lang",
                    source_id=source_id,
                    pool=pool,
                    group_id=f"nemotron-v3:{upstream_id}",
                    language=language,
                    prompt=prompt,
                    prompt_label=prompt_label,
                    categories=split_categories(raw.get("violated_categories")),
                    metadata={"upstream_id": upstream_id, "official_split": pool},
                )
                if prompt and prompt_label in VALID_LABELS:
                    yield StudyRow(response=None, response_label=None, **common)
                if prompt and response and response_label in VALID_LABELS:
                    yield StudyRow(response=response, response_label=response_label, **common)


def _wildguard_rows(path: Path) -> Iterator[StudyRow]:
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
    for raw in raw_rows:
        prompt = str(raw.get("prompt") or "").strip()
        response = raw.get("response")
        response = str(response).strip() if response is not None else None
        prompt_label = {
            "harmful": "unsafe",
            "unharmful": "safe",
        }.get(str(raw.get("prompt_harm_label") or "").casefold())
        response_label = {
            "harmful": "unsafe",
            "unharmful": "safe",
        }.get(str(raw.get("response_harm_label") or "").casefold())
        prompt_group = content_hash(prompt, None)
        source_id = content_hash(prompt, response)[:24]
        common = dict(
            source="wildguardtrain_en",
            source_id=source_id,
            pool="unsplit",
            group_id=f"wildguard-prompt:{prompt_group}",
            language="en",
            prompt=prompt,
            prompt_label=prompt_label or "",
            categories=[],
            metadata={
                "adversarial": raw.get("adversarial"),
                "subcategory": raw.get("subcategory"),
                "response_refusal_label": raw.get("response_refusal_label"),
            },
        )
        if prompt and prompt_label in VALID_LABELS:
            yield StudyRow(response=None, response_label=None, **common)
        if prompt and response and response_label in VALID_LABELS:
            yield StudyRow(response=response, response_label=response_label, **common)


def _parquet_batches(paths: Iterable[Path], columns: list[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
            yield from batch.to_pylist()


def _sea_rows(root: Path) -> Iterator[StudyRow]:
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
    prompt_map = {"safe": "safe", "sensitive": "safe", "unsafe": "unsafe"}
    response_map = {"safe": "safe", "sensitive": "unsafe", "unsafe": "unsafe"}
    for raw in _parquet_batches(paths, columns):
        if str(raw.get("language") or "").casefold() != "vietnamese":
            continue
        if str(raw.get("quality_assurance") or "").casefold() != "pass":
            continue
        prompt = str(raw.get("prompt") or "").strip()
        response = raw.get("response")
        response = str(response).strip() if response is not None else None
        prompt_label_raw = str(raw.get("prompt_label") or "").casefold()
        response_label_raw = str(raw.get("response_label") or "").casefold()
        prompt_label = prompt_map.get(prompt_label_raw)
        response_label = response_map.get(response_label_raw)
        upstream_id = str(raw.get("id") or content_hash(prompt, None)[:24])
        source_id = f"{upstream_id}:{content_hash(prompt, response)[:12]}"
        common = dict(
            source="sea_cultural_vi",
            source_id=source_id,
            pool="unsplit",
            group_id=f"sea-cultural:{upstream_id}",
            language="vi",
            prompt=prompt,
            prompt_label=prompt_label or "",
            categories=[],
            metadata={
                "upstream_id": upstream_id,
                "topic": raw.get("topic"),
                "prompt_label_three_way": raw.get("prompt_label"),
                "response_label_three_way": raw.get("response_label"),
                "prompt_risk_score": raw.get("prompt_risk_score"),
                "response_risk_score": raw.get("response_risk_score"),
                "response_model": raw.get("response_model"),
                "binary_mapping": "Sensitive prompt->safe; Sensitive response->unsafe",
            },
        )
        if prompt and prompt_label in VALID_LABELS:
            yield StudyRow(response=None, response_label=None, **common)
        if prompt and response and response_label in VALID_LABELS:
            yield StudyRow(response=response, response_label=response_label, **common)


def benchmark_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    for path in paths:
        for raw in read_jsonl(path):
            prompt = raw.get("prompt") or raw.get("query")
            response = raw.get("response")
            if not prompt:
                continue
            for exact, loose in (
                (content_hash(prompt, None), loose_content_hash(prompt, None)),
                (content_hash(prompt, response), loose_content_hash(prompt, response)),
            ):
                hashes.add("exact:" + exact)
                hashes.add("loose:" + loose)
    return hashes


def canonicalize(rows: Iterable[StudyRow], leakage_hashes: set[str]) -> tuple[list[StudyRow], dict[str, Any]]:
    canonical: dict[tuple[str, str, str], StudyRow] = {}
    conflicts: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        counts["raw_normalized_rows"] += 1
        if (
            "exact:" + row.exact_hash in leakage_hashes
            or "loose:" + row.loose_hash in leakage_hashes
            or "exact:" + row.prompt_hash in leakage_hashes
            or "loose:" + row.prompt_loose_hash in leakage_hashes
        ):
            counts["benchmark_overlap_rows"] += 1
            continue
        key = (row.language, row.view, row.exact_hash)
        if key in conflicts:
            counts["rows_in_conflict_groups"] += 1
            continue
        old = canonical.get(key)
        if old is None:
            canonical[key] = row
            continue
        old_labels = (old.prompt_label, old.response_label)
        new_labels = (row.prompt_label, row.response_label)
        if old_labels != new_labels:
            del canonical[key]
            conflicts.add(key)
            counts["conflicting_content_groups"] += 1
            counts["rows_in_conflict_groups"] += 2
        else:
            counts["duplicate_rows"] += 1
    values = list(canonical.values())
    counts["canonical_candidates"] = len(values)
    return values, dict(sorted(counts.items()))


def _distribute(total: int, keys: tuple[str, ...], seed: int, cell: tuple[str, str]) -> dict[str, int]:
    base, remainder = divmod(total, len(keys))
    order = sorted(keys, key=lambda key: stable_hash(f"language-remainder:{seed}:{cell}:{key}"))
    result = {key: base for key in keys}
    for key in order[:remainder]:
        result[key] += 1
    return result


def _targets_by_stratum(
    targets: dict[tuple[str, str], int],
    *,
    multilingual: bool,
    seed: int,
) -> dict[tuple[str, ...], int]:
    if not multilingual:
        return {(view, label): count for (view, label), count in targets.items()}
    output: dict[tuple[str, ...], int] = {}
    for cell, total in targets.items():
        for language, count in _distribute(total, LANGUAGES, seed, cell).items():
            output[(language, *cell)] = count
    return output


def _row_stratum(row: StudyRow, multilingual: bool) -> tuple[str, ...]:
    cell = (row.view, row.label)
    return (row.language, *cell) if multilingual else cell


def select_rows(
    candidates: list[StudyRow],
    source: str,
    seed: int,
    *,
    multilingual: bool,
    train_cell_targets: dict[tuple[str, str], int] | None = None,
    validation_cell_targets: dict[tuple[str, str], int] | None = None,
) -> tuple[list[StudyRow], list[StudyRow], dict[str, Any]]:
    train_cell_targets = train_cell_targets or TRAIN_TARGETS
    validation_cell_targets = validation_cell_targets or VALIDATION_TARGETS
    if source == "nemotron_v3_9lang":
        train_pool = [row for row in candidates if row.pool == "train"]
        validation_pool = [row for row in candidates if row.pool == "validation"]
        train_groups = {row.group_id for row in train_pool}
        validation_groups = {row.group_id for row in validation_pool}
        overlap = train_groups & validation_groups
        if overlap:
            train_pool = [row for row in train_pool if row.group_id not in overlap]
            validation_pool = [row for row in validation_pool if row.group_id not in overlap]
    else:
        train_pool = list(candidates)
        validation_pool = list(candidates)
        overlap = set()

    validation_targets = _targets_by_stratum(validation_cell_targets, multilingual=multilingual, seed=seed)
    train_targets = _targets_by_stratum(train_cell_targets, multilingual=multilingual, seed=seed)

    by_validation_stratum: dict[tuple[str, ...], list[StudyRow]] = defaultdict(list)
    for row in validation_pool:
        by_validation_stratum[_row_stratum(row, multilingual)].append(row)
    selected_validation: list[StudyRow] = []
    validation_groups_selected: set[str] = set()
    for stratum, target in sorted(validation_targets.items()):
        if target == 0:
            continue
        values = sorted(
            by_validation_stratum.get(stratum, []),
            key=lambda row: stable_hash(f"sample:{seed}:{source}:validation:{row.identity}"),
        )
        actual = 0
        for row in values:
            if row.group_id in validation_groups_selected:
                continue
            validation_groups_selected.add(row.group_id)
            selected_validation.append(row)
            actual += 1
            if actual == target:
                break
        if actual != target:
            raise ValueError(f"{source} validation stratum {stratum} needs {target}, found {actual}")

    by_train_stratum: dict[tuple[str, ...], list[StudyRow]] = defaultdict(list)
    for row in train_pool:
        if row.group_id not in validation_groups_selected:
            by_train_stratum[_row_stratum(row, multilingual)].append(row)
    selected_train: list[StudyRow] = []
    for stratum, target in sorted(train_targets.items()):
        if target == 0:
            continue
        values = sorted(
            by_train_stratum.get(stratum, []),
            key=lambda row: stable_hash(f"sample:{seed}:{source}:train:{row.identity}"),
        )
        if len(values) < target:
            raise ValueError(f"{source} train stratum {stratum} needs {target}, found {len(values)}")
        selected_train.extend(values[:target])

    expected_train = sum(train_cell_targets.values())
    expected_validation = sum(validation_cell_targets.values())
    if len(selected_train) != expected_train:
        raise AssertionError(f"Expected {expected_train:,} train rows, got {len(selected_train)}")
    if len(selected_validation) != expected_validation:
        raise AssertionError(f"Expected {expected_validation:,} validation rows, got {len(selected_validation)}")
    if {row.group_id for row in selected_train} & {row.group_id for row in selected_validation}:
        raise AssertionError("Group leakage after selection")
    audit = {
        "official_split_group_conflicts_removed": len(overlap),
        "validation_groups_selected": len(validation_groups_selected),
        "train_targets": {"/".join(key): value for key, value in sorted(train_targets.items())},
        "validation_targets": {"/".join(key): value for key, value in sorted(validation_targets.items())},
    }
    return selected_train, selected_validation, audit


def _render_rows(rows: list[StudyRow], split: str, seed: int) -> list[dict[str, Any]]:
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
    return sorted(output, key=lambda value: stable_hash(f"output:{seed}:{value['example_id']}"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["examples"] += 1
            counts[f"language:{row['language']}"] += 1
            counts[f"view:{row['view']}"] += 1
            counts[f"label:{row['safety_label']}"] += 1
            counts[f"cell:{row['view']}:{row['safety_label']}"] += 1
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        **dict(sorted(counts.items())),
    }


def build_arm(
    source: str,
    source_path: Path,
    benchmark_root: Path,
    output_dir: Path,
    *,
    seed: int = 3407,
    smoke: bool = False,
) -> dict[str, Any]:
    loaders = {
        "nemotron_v3_9lang": lambda: _v3_rows(source_path),
        "wildguardtrain_en": lambda: _wildguard_rows(source_path),
        "sea_cultural_vi": lambda: _sea_rows(source_path),
    }
    if source not in loaders:
        raise ValueError(f"Unknown study source: {source}")
    leakage_hashes = benchmark_hashes(benchmark_root)
    candidates, canonical_audit = canonicalize(loaders[source](), leakage_hashes)
    multilingual = source == "nemotron_v3_9lang"
    train_cell_targets = SMOKE_TRAIN_TARGETS if smoke else TRAIN_TARGETS
    validation_cell_targets = SMOKE_VALIDATION_TARGETS if smoke else VALIDATION_TARGETS
    train, validation, selection_audit = select_rows(
        candidates,
        source,
        seed,
        multilingual=multilingual,
        train_cell_targets=train_cell_targets,
        validation_cell_targets=validation_cell_targets,
    )
    rendered = {
        "train": _render_rows(train, "train", seed),
        "validation": _render_rows(validation, "validation", seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in rendered.items()
    }
    manifest = {
        "schema_version": 1,
        "study": "qwen3_4b_data_source_screen_smoke" if smoke else "qwen3_4b_data_source_screen_80k",
        "source": source,
        "source_path": str(source_path.resolve()),
        "source_revisions": SOURCE_REVISIONS,
        "seed": seed,
        "train_examples": len(rendered["train"]),
        "validation_examples": len(rendered["validation"]),
        "sampling": {
            "method": "stable SHA-256 random sampling within matched view/binary-label cells",
            "v3_language_policy": "equal per-language quota within every cell",
            "train_cell_targets": {f"{view}/{label}": count for (view, label), count in train_cell_targets.items()},
            "validation_cell_targets": {f"{view}/{label}": count for (view, label), count in validation_cell_targets.items()},
        },
        "rendering": {
            "taxonomy_mode": "off",
            "thinking_mode": "no_think",
            "target": "Nemotron JSON binary contract",
        },
        "benchmark_root": str(benchmark_root.resolve()),
        "benchmark_hashes": len(leakage_hashes),
        "canonicalization": canonical_audit,
        "selection": selection_audit,
        "splits": splits,
        "license_note": {
            "nemotron_v3_9lang": "CC-BY-4.0; derivative redistribution allowed with attribution.",
            "wildguardtrain_en": "ODC-BY plus AI2 gated Responsible Use acceptance; do not publish a bundle that bypasses the gate.",
            "sea_cultural_vi": "Manually gated upstream with no license declared in the pinned dataset card; do not publicly redistribute rows without permission.",
        }[source],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_report = validate_arm(output_dir)
    if validation_report["errors"]:
        raise ValueError("Built arm failed validation: " + "; ".join(validation_report["errors"][:10]))
    return manifest


def validate_arm(directory: Path) -> dict[str, Any]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    seen_examples: set[str] = set()
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid manifest.json: {exc}")
    expected_by_split = {
        "train": int(manifest.get("train_examples", 80_000)),
        "validation": int(manifest.get("validation_examples", 1_000)),
    }
    for split, expected in expected_by_split.items():
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        for raw in read_jsonl(path):
            counts[f"split:{split}"] += 1
            counts[f"split:{split}:cell:{raw.get('view')}:{raw.get('safety_label')}"] += 1
            example_id = str(raw.get("example_id") or "")
            if not example_id or example_id in seen_examples:
                errors.append(f"{path}:{raw['_line']}: missing or duplicate example_id")
            seen_examples.add(example_id)
            if raw.get("taxonomy_mode") != "off" or raw.get("thinking_mode") != "no_think":
                errors.append(f"{path}:{raw['_line']}: expected taxonomy-off/no-think")
            try:
                target = json.loads(str(raw.get("target") or ""))
            except json.JSONDecodeError:
                errors.append(f"{path}:{raw['_line']}: target is not JSON")
                target = {}
            if "Safety Categories" in target:
                errors.append(f"{path}:{raw['_line']}: taxonomy-off target contains categories")
            metadata = raw.get("metadata") or {}
            group_id = str(metadata.get("study_group_id") or "")
            if not group_id:
                errors.append(f"{path}:{raw['_line']}: missing study_group_id")
            split_groups[split].add(group_id)
            split_content[split].add(str(raw.get("content_sha256") or ""))
        if counts[f"split:{split}"] != expected:
            errors.append(f"{split} expected {expected}, got {counts[f'split:{split}']}")
    group_overlap = split_groups["train"] & split_groups["validation"]
    content_overlap = split_content["train"] & split_content["validation"]
    if group_overlap:
        errors.append(f"{len(group_overlap)} study groups cross train/validation")
    if content_overlap:
        errors.append(f"{len(content_overlap)} exact content hashes cross train/validation")
    report = {
        "directory": str(directory),
        "counts": dict(sorted(counts.items())),
        "group_overlap": len(group_overlap),
        "content_overlap": len(content_overlap),
        "errors": errors,
        "valid": not errors,
    }
    (directory / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def default_source_paths(repo_root: Path) -> dict[str, Path]:
    sea_env = os.environ.get("SOURCE_STUDY_SEA_ROOT")
    sea_default = Path(
        r"D:\AI\HuggingFace\hub\datasets--aisingapore--SEA-Safeguard-Train-Cultural-v3"
        r"\snapshots\2a51a837855ff7ae7d383fa4af6fa11c2cee8a92\Vietnam"
    )
    return {
        "nemotron_v3_9lang": (repo_root / ".." / "offline-bundle-work" / "snapshots" / "nemotron-9lang").resolve(),
        "wildguardtrain_en": (repo_root / "input" / "stage2" / "wildguard" / "wildguardtrain.jsonl").resolve(),
        "sea_cultural_vi": Path(sea_env).resolve() if sea_env else sea_default,
    }
