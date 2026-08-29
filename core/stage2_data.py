from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import yaml

try:
    from .prompt import N23, nemotron_instruction
except ImportError:
    from prompt import N23, nemotron_instruction


VALID_LABELS = {"safe", "unsafe"}
SPLIT_NAMES = {"train", "valid", "validation"}
REASONING_FIELDS = (
    "efficient_reasoning_gpt_oss_120b",
    "efficient_reasoning_qwen3_32b",
    "efficient_reasoning_deepseek_r1_0528",
)

# These are the only Nemotron 3.5 train sources that add new text-only safety
# examples to this blend.  ``aegis_v3_human`` is a multilingual replay of V3,
# and ``nemotron_content_safety_reasoning_dataset_aegis`` is already consumed
# from the richer reasoning JSONL.  The remaining sources are image-grounded
# or topic-following.  Keep this allowlist explicit so a future dataset update
# cannot silently change the research mixture.
N35_SELECTED_TEXT_SOURCES = {"text_synthetic", "multimodal_synthetic_2"}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def content_hash(prompt: Any, response: Any = None) -> str:
    return stable_hash(normalized_text(prompt) + "\n<response>\n" + normalized_text(response))


def loose_content_hash(prompt: Any, response: Any = None) -> str:
    def loose(value: Any) -> str:
        return re.sub(r"[^\w]+", " ", normalized_text(value), flags=re.UNICODE).strip()

    return stable_hash(loose(prompt) + "\n<response>\n" + loose(response))


def split_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;|]", str(value))
    return list(dict.fromkeys(item.strip() for item in items if str(item).strip()))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            value["_line"] = line_number
            yield value


def resolve_path(root: Path, value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded)
    return path if path.is_absolute() else (root / path).resolve()


def load_stage2_config(path: Path) -> tuple[dict[str, Any], Path]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML object: {path}")
    return cfg, path.resolve().parent


def output_payload(prompt_label: str, response_label: str | None, categories: list[str], taxonomy_mode: str) -> str:
    payload: dict[str, str] = {"User Safety": prompt_label}
    if response_label is not None:
        payload["Response Safety"] = response_label
    # NVIDIA's V3/3.5 contract says to omit Safety Categories when all
    # classified turns are safe.  Some upstream rows retain an auxiliary
    # category such as "Needs Caution" despite a safe binary verdict; that
    # annotation must not produce a contradictory safe+category target.
    any_unsafe = prompt_label == "unsafe" or response_label == "unsafe"
    if taxonomy_mode == "on" and any_unsafe and categories:
        payload["Safety Categories"] = ", ".join(category for category in N23 if category in set(categories))
        if not payload["Safety Categories"]:
            payload.pop("Safety Categories")
    return json.dumps(payload, ensure_ascii=False)


def clean_reasoning(value: str) -> str:
    value = value.strip()
    match = re.search(r"<think>(.*?)</think>", value, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else value).strip()


def reasoning_has_label_leakage(value: str) -> bool:
    normalized = normalized_text(value)
    return any(
        phrase in normalized
        for phrase in (
            "ground truth says",
            "ground-truth says",
            "provided label is",
            "given label is",
            "gold label is",
        )
    )


@dataclass
class Normalized:
    source: str
    source_id: str
    split: str
    language: str
    prompt: str
    response: str | None
    prompt_label: str
    response_label: str | None
    categories: list[str]
    reasoning: str | None = None
    force_taxonomy_off: bool = False
    metadata: dict[str, Any] | None = None

    @property
    def view(self) -> str:
        return "PR" if self.response is not None else "P"

    @property
    def semantic_id(self) -> str:
        return f"{self.source}:{self.source_id}:{self.view}:{self.language}"


def assigned_split(source: str, source_id: str, validation_fraction: float) -> str:
    bucket = int(stable_hash(f"stage2-split:{source}:{source_id}")[:12], 16) / float(16**12)
    return "validation" if bucket < validation_fraction else "train"


def taxonomy_on_allowed(row: Normalized) -> bool:
    """Return whether a row can be rendered with the N23 taxonomy visible.

    A safe row needs no category target and is therefore valid in taxonomy-ON
    mode.  Unsafe rows require at least one usable N23 category; unknown-only
    unsafe rows stay taxonomy-OFF.  Mixed unsafe rows may remain ON, but only
    their N23 labels are emitted in the target.
    """
    known = [value for value in row.categories if value in N23]
    any_unsafe = row.prompt_label == "unsafe" or row.response_label == "unsafe"
    return not row.force_taxonomy_off and (not any_unsafe or bool(known))


def render(
    row: Normalized,
    seed: int,
    *,
    taxonomy_mode: str | None = None,
    thinking_mode: str | None = None,
) -> dict[str, Any]:
    unknown_categories = [value for value in row.categories if value not in N23]
    taxonomy_mode = taxonomy_mode or ("on" if taxonomy_on_allowed(row) else "off")
    thinking_mode = thinking_mode or "no_think"
    if taxonomy_mode not in {"on", "off"}:
        raise ValueError(f"Unknown taxonomy mode: {taxonomy_mode}")
    if taxonomy_mode == "on" and not taxonomy_on_allowed(row):
        raise ValueError(f"Cannot render taxonomy-on without compatible N23 labels: {row.semantic_id}")
    if thinking_mode not in {"think", "no_think"}:
        raise ValueError(f"Unknown thinking mode: {thinking_mode}")
    if thinking_mode == "think" and not row.reasoning:
        raise ValueError(f"Cannot render THINK without reasoning: {row.semantic_id}")
    target_json = output_payload(row.prompt_label, row.response_label, row.categories, taxonomy_mode)
    target = target_json
    if thinking_mode == "think":
        # Qwen's native thinking chat template already ends the prompt with
        # ``<think>\n``. The supervised completion therefore starts with the
        # rationale and closes the native block exactly once.
        target = f"{clean_reasoning(row.reasoning or '')}\n</think>\n\n{target_json}"
    render_key = f"{row.semantic_id}:taxonomy={taxonomy_mode}:thinking={thinking_mode}"
    return {
        "example_id": f"stage2:{stable_hash(render_key)[:24]}",
        "semantic_id": row.semantic_id,
        "source_id": row.source_id,
        "dataset_source": row.source,
        "source_split": row.split,
        "language": row.language,
        "view": row.view,
        "prompt": row.prompt,
        "response": row.response,
        "safety_label": row.response_label if row.response is not None else row.prompt_label,
        "prompt_safety_label": row.prompt_label,
        "categories": [value for value in row.categories if value in N23],
        "unknown_categories": unknown_categories,
        "taxonomy_mode": taxonomy_mode,
        "thinking_mode": thinking_mode,
        "reasoning": clean_reasoning(row.reasoning or "") if thinking_mode == "think" else None,
        "instruction": nemotron_instruction(
            row.prompt,
            row.response,
            taxonomy_mode=taxonomy_mode,
            thinking_mode=thinking_mode,
        ),
        "target": target,
        "target_json": target_json,
        "content_sha256": content_hash(row.prompt, row.response),
        "content_loose_sha256": loose_content_hash(row.prompt, row.response),
        "metadata": row.metadata or {},
    }


def render_views(row: Normalized, seed: int) -> list[dict[str, Any]]:
    """Render one deterministic view per semantic record.

    Taxonomy ON/OFF is sampled by semantic ID (75/25 for V3 and VI).  The
    reasoning source uses this project's stable 50/50 THINK/NO-THINK policy,
    never a paired duplicate.  The public NVIDIA artifacts expose both modes
    but do not document this exact sampling ratio.
    """
    if row.source == "wildguardtrain":
        taxonomy_modes = ["off"]
        thinking_modes = ["no_think"]
    elif row.source in {"nemotron_v3_replay"} or row.source.startswith("vi_"):
        bucket = int(stable_hash(f"taxonomy:{seed}:{row.semantic_id}")[:8], 16) / float(16**8)
        taxonomy_modes = ["on" if bucket < 0.75 and taxonomy_on_allowed(row) else "off"]
        thinking_modes = ["no_think"]
    elif row.source == "nemotron_reasoning_28k":
        taxonomy_modes = ["on" if taxonomy_on_allowed(row) else "off"]
        if row.reasoning:
            bucket = int(stable_hash(f"thinking:{seed}:{row.semantic_id}")[:8], 16) / float(16**8)
            thinking_modes = ["think" if bucket < 0.5 else "no_think"]
        else:
            thinking_modes = ["no_think"]
    elif row.source == "nemotron35_selected":
        taxonomy_modes = ["on" if taxonomy_on_allowed(row) else "off"]
        thinking_modes = ["no_think"]
    else:
        taxonomy_modes = ["on"] if taxonomy_on_allowed(row) else ["off"]
        thinking_modes = ["no_think"]
    return [
        render(row, seed, taxonomy_mode=taxonomy_mode, thinking_mode=thinking_mode)
        for taxonomy_mode in taxonomy_modes
        for thinking_mode in thinking_modes
    ]


def vi_records(directory: Path, source_name: str) -> Iterator[Normalized]:
    for split, filename in (("train", "nemotron_train_en_vi_v10_final.jsonl"), ("validation", "nemotron_valid_en_vi_v10_final.jsonl")):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        for raw in read_jsonl(path):
            source_id = str(raw.get("record_uid") or raw.get("id") or f"line-{raw['_line']}")
            prompt = str(raw.get("prompt_vi") or "").strip()
            response = raw.get("response_vi")
            response = str(response).strip() if response is not None else None
            prompt_label = str(raw.get("prompt_label") or "").lower()
            response_label = str(raw.get("response_label") or "").lower() or None
            categories = split_categories(raw.get("violated_categories"))
            metadata = {"prompt_semantic_hash": stable_hash(normalized_text(raw.get("prompt_en")))}
            prompt_usable = bool(prompt) and prompt != "REDACTED"
            if prompt_usable and prompt_label in VALID_LABELS:
                yield Normalized(source_name, source_id, split, "vi", prompt, None, prompt_label, None, categories, metadata=metadata)
            if prompt_usable and response and response_label in VALID_LABELS:
                yield Normalized(source_name, source_id, split, "vi", prompt, response, prompt_label, response_label, categories, metadata=metadata)


def v3_records(root: Path, languages: list[str], seed: int) -> Iterator[Normalized]:
    for split, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for language in languages:
            path = root / language / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            for raw in read_jsonl(path):
                upstream_id = str(raw.get("id") or f"{language}-{raw['_line']}")
                grouped[upstream_id][language].append(raw)
        for upstream_id in sorted(grouped):
            available = sorted(grouped[upstream_id])
            chosen = available[int(stable_hash(f"v3-language:{seed}:{upstream_id}")[:12], 16) % len(available)]
            for raw in grouped[upstream_id][chosen]:
                prompt = str(raw.get("prompt") or "").strip()
                response = raw.get("response")
                response = str(response).strip() if response is not None else None
                source_id = f"{upstream_id}:{content_hash(prompt, response)[:12]}"
                prompt_label = str(raw.get("prompt_label") or "").lower()
                response_label = str(raw.get("response_label") or "").lower() or None
                categories = split_categories(raw.get("violated_categories"))
                if prompt and prompt_label in VALID_LABELS:
                    yield Normalized("nemotron_v3_replay", source_id, split, chosen, prompt, None, prompt_label, None, categories)
                if prompt and response and response_label in VALID_LABELS:
                    yield Normalized("nemotron_v3_replay", source_id, split, chosen, prompt, response, prompt_label, response_label, categories)


def reasoning_records(path: Path, validation_fraction: float, seed: int) -> Iterator[Normalized]:
    for raw in read_jsonl(path):
        upstream_id = str(raw.get("id") or f"line-{raw['_line']}")
        prompt = str(raw.get("prompt") or "").strip()
        response = raw.get("response")
        response = str(response).strip() if response is not None else None
        source_id = f"{upstream_id}:{content_hash(prompt, response)[:12]}"
        split = assigned_split("nemotron_reasoning_28k", source_id, validation_fraction)
        prompt_label = str(raw.get("prompt_label") or "").lower()
        response_label = str(raw.get("response_label") or "").lower() or None
        categories = split_categories(raw.get("violated_categories"))
        eligible: list[tuple[str, str]] = []
        leaked_fields: list[str] = []
        for field in REASONING_FIELDS:
            teacher = field.removeprefix("efficient_reasoning_")
            value = str(raw.get(field) or "").strip()
            if not value or raw.get(f"prediction_gt_mismatch_{teacher}") is not False:
                continue
            if reasoning_has_label_leakage(value):
                leaked_fields.append(field)
                continue
            eligible.append((field, value))
        if eligible:
            selected = int(stable_hash(f"reasoner:{seed}:{source_id}")[:12], 16) % len(eligible)
            field, reasoning = eligible[selected]
        else:
            field, reasoning = None, None
        metadata = {
            "reasoning_field": field,
            "eligible_reasoning_fields": len(eligible),
            "reasoning_fields_rejected_for_label_leakage": leaked_fields,
        }
        if prompt and prompt_label in VALID_LABELS:
            # A PR teacher trace commonly discusses the response. Do not attach
            # it to the prompt-only view, where that would reveal unseen text.
            prompt_reasoning = reasoning if response is None else None
            yield Normalized("nemotron_reasoning_28k", source_id, split, "en", prompt, None, prompt_label, None, categories, prompt_reasoning, metadata=metadata)
        if prompt and response and response_label in VALID_LABELS:
            yield Normalized("nemotron_reasoning_28k", source_id, split, "en", prompt, response, prompt_label, response_label, categories, reasoning, metadata=metadata)


def wildguard_records(path: Path, validation_fraction: float) -> Iterator[Normalized]:
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(path)
    for file in files:
        for raw in read_jsonl(file):
            source_id = str(raw.get("id") or stable_hash(f"{raw.get('prompt')}\n{raw.get('response')}")[:24])
            split = assigned_split("wildguardtrain", source_id, validation_fraction)
            prompt = str(raw.get("prompt") or "").strip()
            response = raw.get("response")
            response = str(response).strip() if response is not None else None
            prompt_label = {"harmful": "unsafe", "unharmful": "safe"}.get(str(raw.get("prompt_harm_label") or "").lower(), str(raw.get("prompt_harm_label") or "").lower())
            response_label = {"harmful": "unsafe", "unharmful": "safe"}.get(str(raw.get("response_harm_label") or "").lower(), str(raw.get("response_harm_label") or "").lower()) or None
            metadata = {
                "response_refusal_label": raw.get("response_refusal_label"),
                "adversarial": raw.get("adversarial"),
                "subcategory": raw.get("subcategory"),
            }
            if prompt and prompt_label in VALID_LABELS:
                yield Normalized("wildguardtrain", source_id, split, "en", prompt, None, prompt_label, None, [], force_taxonomy_off=True, metadata=metadata)
            if prompt and response and response_label in VALID_LABELS:
                yield Normalized("wildguardtrain", source_id, split, "en", prompt, response, prompt_label, response_label, [], force_taxonomy_off=True, metadata=metadata)


def nemotron35_records(path: Path, validation_fraction: float, excluded_hashes: set[str]) -> Iterator[Normalized]:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    seen: set[str] = set()
    for raw in table.to_pylist():
        if raw.get("image_path") or str(raw.get("task_type")) != "safety":
            continue
        if str(raw.get("dataset_source") or "") not in N35_SELECTED_TEXT_SOURCES:
            continue
        prompt = str(raw.get("prompt") or "").strip()
        response = raw.get("response")
        response = str(response).strip() if response is not None else None
        row_hash = content_hash(prompt, response)
        if not prompt or row_hash in seen or row_hash in excluded_hashes:
            continue
        seen.add(row_hash)
        source_id = str(raw.get("row_id") or row_hash[:24])
        split = assigned_split("nemotron35_selected", source_id, validation_fraction)
        prompt_label = str(raw.get("input_label") or "").lower()
        response_label = str(raw.get("response_label") or "").lower() or None
        categories = split_categories(raw.get("violated_categories"))
        reasoning = str(raw.get("reasoning_trace") or "").strip() or None
        if reasoning and reasoning_has_label_leakage(reasoning):
            reasoning = None
        metadata = {
            "dataset_source": raw.get("dataset_source"),
            "provenance": raw.get("provenance"),
            "categories_raw": categories,
            "categories_n23": [category for category in categories if category in N23],
        }
        if prompt_label in VALID_LABELS:
            prompt_reasoning = reasoning if response is None else None
            yield Normalized("nemotron35_selected", source_id, split, str(raw.get("language") or "en"), prompt, None, prompt_label, None, categories, prompt_reasoning, metadata=metadata)
        if response and response_label in VALID_LABELS:
            yield Normalized("nemotron35_selected", source_id, split, str(raw.get("language") or "en"), prompt, response, prompt_label, response_label, categories, reasoning, metadata=metadata)


def benchmark_hashes(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for root in paths:
        files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file():
                continue
            for raw in read_jsonl(path):
                prompt = raw.get("prompt") or raw.get("query") or raw.get("text")
                response = raw.get("response")
                if prompt:
                    result.add("exact:" + content_hash(prompt, response))
                    result.add("exact:" + content_hash(prompt, None))
                    result.add("loose:" + loose_content_hash(prompt, response))
                    result.add("loose:" + loose_content_hash(prompt, None))
    return result


def audit_translation_directory(directory: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    all_uids: dict[str, set[str]] = {}
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    hard_errors: list[str] = []
    audit_warnings: list[str] = []
    for split, filename in (("train", "nemotron_train_en_vi_v10_final.jsonl"), ("valid", "nemotron_valid_en_vi_v10_final.jsonl"), ("test", "nemotron_test_en_vi_v10_final.jsonl")):
        path = directory / filename
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        contract_digest = hashlib.sha256()
        for raw in read_jsonl(path):
            uid = str(raw.get("record_uid") or "")
            if not uid or uid in seen:
                hard_errors.append(f"{split}:{raw['_line']}: missing or duplicate record_uid={uid!r}")
            seen.add(uid)
            prompt_vi = str(raw.get("prompt_vi") or "").strip()
            if not prompt_vi:
                if str(raw.get("prompt_en") or "").strip():
                    hard_errors.append(f"{split}:{uid}: source prompt exists but prompt_vi is blank")
                else:
                    audit_warnings.append(f"{split}:{uid}: blank source prompt; row is non-renderable and will be skipped")
            if str(raw.get("response_en") or "").strip() and not str(raw.get("response_vi") or "").strip():
                hard_errors.append(f"{split}:{uid}: response_en exists but response_vi is blank")
            if str(raw.get("prompt_label") or "").lower() not in VALID_LABELS:
                hard_errors.append(f"{split}:{uid}: invalid prompt_label")
            response_label = str(raw.get("response_label") or "").strip().lower()
            if response_label and response_label not in VALID_LABELS:
                hard_errors.append(f"{split}:{uid}: invalid response_label")
            providers[str(raw.get("translation_provider") or "unknown")] += 1
            models[str(raw.get("translation_model") or "unknown")] += 1
            dispositions[str(raw.get("translation_disposition") or "unknown")] += 1
            counts["records"] += 1
            counts[f"prompt_label:{str(raw.get('prompt_label')).lower()}"] += 1
            counts[f"response_present:{bool(str(raw.get('response_vi') or '').strip())}"] += 1
            source_contract = {
                key: raw.get(key)
                for key in ("record_uid", "source_split", "source_id", "prompt_en", "response_en", "prompt_label", "response_label", "violated_categories")
            }
            contract_digest.update(json.dumps(source_contract, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            contract_digest.update(b"\n")
        all_uids[split] = seen
        splits[split] = {
            "path": str(path),
            "sha256": file_hash(path),
            "uid_set_sha256": stable_hash("\n".join(sorted(seen))),
            "source_contract_sha256": contract_digest.hexdigest(),
            **dict(counts),
        }
    overlaps = {
        "train_valid": len(all_uids["train"] & all_uids["valid"]),
        "train_test": len(all_uids["train"] & all_uids["test"]),
        "valid_test": len(all_uids["valid"] & all_uids["test"]),
    }
    if any(overlaps.values()):
        hard_errors.append(f"cross-split UID overlap: {overlaps}")
    return {
        "directory": str(directory),
        "records": sum(len(value) for value in all_uids.values()),
        "splits": splits,
        "providers": dict(providers),
        "models": dict(models),
        "dispositions": dict(dispositions),
        "split_uid_overlap": overlaps,
        "hard_error_count": len(hard_errors),
        "hard_errors": hard_errors[:100],
        "warning_count": len(audit_warnings),
        "warnings": audit_warnings[:100],
        "ready": not hard_errors,
    }


def compare_translation_sources(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"same_record_count": left["records"] == right["records"], "splits": {}}
    for split in ("train", "valid", "test"):
        result["splits"][split] = {
            "left_records": left["splits"][split]["records"],
            "right_records": right["splits"][split]["records"],
            "same_uid_set": left["splits"][split]["uid_set_sha256"] == right["splits"][split]["uid_set_sha256"],
            "same_source_contract": left["splits"][split]["source_contract_sha256"] == right["splits"][split]["source_contract_sha256"],
        }
    return result


def audit_translations(cfg: dict[str, Any], config_root: Path) -> dict[str, Any]:
    sources = cfg["sources"]
    luna = audit_translation_directory(resolve_path(config_root, sources["vi_luna_sol"]))
    gemini = audit_translation_directory(resolve_path(config_root, sources["vi_gemini"]))
    return {"luna_sol": luna, "gemini": gemini, "comparison": compare_translation_sources(luna, gemini)}


def inventory_sources(cfg: dict[str, Any], config_root: Path, translation_source: str) -> dict[str, Any]:
    """Count the full normalized registry without rendering large instruction strings."""
    if translation_source not in {"gemini", "luna_sol"}:
        raise ValueError("translation_source must be gemini or luna_sol")
    stage = cfg["stage2"]
    sources = cfg["sources"]
    seed = int(stage["seed"])
    validation_fraction = float(stage["validation_fraction"])
    paths = {
        "vi": resolve_path(config_root, sources[f"vi_{translation_source}"]),
        "v3": resolve_path(config_root, sources["v3_root"]),
        "reasoning": resolve_path(config_root, sources["reasoning_jsonl"]),
        "nemotron35": resolve_path(config_root, sources["nemotron35_parquet"]),
        "wildguard": resolve_path(config_root, sources["wildguardtrain_jsonl"]),
    }
    counts: Counter[str] = Counter()
    content: set[str] = set()
    semantics: set[str] = set()
    prompt_views: set[tuple[str, str, str, str]] = set()
    blockers = [f"missing source {name}: {path}" for name, path in paths.items() if not path.exists()]
    for path in (resolve_path(config_root, value) for value in sources.get("benchmark_test_paths", [])):
        if not path.exists():
            blockers.append(f"missing benchmark test path for leakage audit: {path}")

    def add(name: str, values: Iterable[Normalized]) -> None:
        for row in values:
            if row.semantic_id in semantics:
                counts[f"source:{name}:duplicate_semantic"] += 1
                continue
            if row.view == "P":
                prompt_hash = str((row.metadata or {}).get("prompt_semantic_hash") or content_hash(row.prompt, None))
                key = (name, row.split, row.language, prompt_hash)
                if key in prompt_views:
                    counts[f"source:{name}:duplicate_prompt_view"] += 1
                    continue
                prompt_views.add(key)
            semantics.add(row.semantic_id)
            content.add(content_hash(row.prompt, row.response))
            counts[f"source:{name}:examples"] += 1
            counts[f"source:{name}:split:{row.split}"] += 1
            counts[f"source:{name}:view:{row.view}"] += 1
            counts[f"source:{name}:label:{row.response_label if row.response is not None else row.prompt_label}"] += 1
            counts[f"source:{name}:unknown_category:{any(category not in N23 for category in row.categories)}"] += 1

    if paths["v3"].exists():
        add("nemotron_v3_replay", v3_records(paths["v3"], list(stage["v3_languages"]), seed))
    if paths["vi"].exists():
        add(f"vi_{translation_source}", vi_records(paths["vi"], f"vi_{translation_source}"))
    if paths["wildguard"].exists():
        add("wildguardtrain", wildguard_records(paths["wildguard"], validation_fraction))
    if paths["reasoning"].exists():
        add("nemotron_reasoning_28k", reasoning_records(paths["reasoning"], validation_fraction, seed))
    if paths["nemotron35"].exists():
        add("nemotron35_selected", nemotron35_records(paths["nemotron35"], validation_fraction, content))
    return {
        "translation_source": translation_source,
        "source_paths": {key: str(value) for key, value in paths.items()},
        "counts": dict(sorted(counts.items())),
        "unique_semantic_examples": len(semantics),
        "unique_content_hashes": len(content),
        "blockers": blockers,
    }


def build_dataset(
    cfg: dict[str, Any],
    config_root: Path,
    translation_source: str,
    output_dir: Path,
    smoke_per_source: int = 0,
    allow_incomplete: bool = False,
    excluded_sources: set[str] | None = None,
) -> dict[str, Any]:
    stage = cfg["stage2"]
    sources = cfg["sources"]
    seed = int(stage["seed"])
    validation_fraction = float(stage["validation_fraction"])
    blockers: list[str] = []
    source_blockers: list[str] = []
    warnings: list[str] = []
    excluded_sources = set(excluded_sources or ())
    valid_source_keys = {"v3", "vi", "wildguard", "reasoning", "nemotron35"}
    if unknown_exclusions := excluded_sources - valid_source_keys:
        raise ValueError("Unknown excluded sources: " + ", ".join(sorted(unknown_exclusions)))
    if translation_source not in {"gemini", "luna_sol"}:
        raise ValueError("translation_source must be gemini or luna_sol")
    source_paths = {
        "vi": resolve_path(config_root, sources[f"vi_{translation_source}"]),
        "v3": resolve_path(config_root, sources["v3_root"]),
        "reasoning": resolve_path(config_root, sources["reasoning_jsonl"]),
        "nemotron35": resolve_path(config_root, sources["nemotron35_parquet"]),
        "wildguard": resolve_path(config_root, sources["wildguardtrain_jsonl"]),
    }
    for name, path in source_paths.items():
        if name in excluded_sources:
            continue
        if not path.exists():
            message = f"missing source {name}: {path}"
            blockers.append(message)
            source_blockers.append(message)
    if blockers and not allow_incomplete:
        raise FileNotFoundError("; ".join(blockers))

    benchmark_paths = [resolve_path(config_root, value) for value in sources.get("benchmark_test_paths", [])]
    for path in benchmark_paths:
        if not path.exists():
            blockers.append(f"missing benchmark test path for leakage audit: {path}")
    leakage_hashes = benchmark_hashes(benchmark_paths)
    counters: Counter[str] = Counter()
    semantic_seen: set[str] = set()
    split_semantics: dict[str, set[str]] = defaultdict(set)
    selected_content_hashes: set[str] = set()
    prompt_view_seen: set[tuple[str, str, str, str]] = set()
    rendered: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def consume(name: str, values: Iterable[Normalized]) -> None:
        taken: Counter[str] = Counter()
        for row in values:
            if smoke_per_source and taken[row.split] >= smoke_per_source:
                if taken["train"] >= smoke_per_source and taken["validation"] >= smoke_per_source:
                    break
                continue
            if row.semantic_id in semantic_seen:
                counters[f"{name}:duplicate_semantic"] += 1
                continue
            if row.view == "P":
                prompt_hash = str((row.metadata or {}).get("prompt_semantic_hash") or content_hash(row.prompt, None))
                prompt_key = (name, row.split, row.language, prompt_hash)
                if prompt_key in prompt_view_seen:
                    counters[f"{name}:duplicate_prompt_view"] += 1
                    continue
                prompt_view_seen.add(prompt_key)
            values = render_views(row, seed)
            representative = values[0]
            if "exact:" + representative["content_sha256"] in leakage_hashes or "loose:" + representative["content_loose_sha256"] in leakage_hashes:
                counters[f"{name}:benchmark_overlap"] += 1
                continue
            semantic_seen.add(row.semantic_id)
            split_semantics[row.split].add(row.semantic_id)
            counters[f"semantic_source:{name}"] += 1
            counters[f"semantic_source:{name}:split:{row.split}"] += 1
            for value in values:
                selected_content_hashes.add(value["content_sha256"])
                rendered[row.split].append(value)
                counters[f"source:{name}"] += 1
                counters[f"source:{name}:split:{row.split}"] += 1
                counters[f"taxonomy:{value['taxonomy_mode']}"] += 1
                counters[f"thinking:{value['thinking_mode']}"] += 1
                counters[f"source:{name}:taxonomy:{value['taxonomy_mode']}"] += 1
                counters[f"source:{name}:thinking:{value['thinking_mode']}"] += 1
                counters[f"view:{value['view']}"] += 1
                counters[f"label:{value['safety_label']}"] += 1
                any_unsafe = value["prompt_safety_label"] == "unsafe" or (
                    value["view"] == "PR" and value["safety_label"] == "unsafe"
                )
                counters[f"source:{name}:any_unsafe:{any_unsafe}"] += 1
                counters[f"unknown_categories:{len(value['unknown_categories']) > 0}"] += 1
            taken[row.split] += 1

    if "v3" not in excluded_sources and source_paths["v3"].exists():
        consume("nemotron_v3_replay", v3_records(source_paths["v3"], list(stage["v3_languages"]), seed))
    if "vi" not in excluded_sources and source_paths["vi"].exists():
        consume(f"vi_{translation_source}", vi_records(source_paths["vi"], f"vi_{translation_source}"))
    if "wildguard" not in excluded_sources and source_paths["wildguard"].exists():
        consume("wildguardtrain", wildguard_records(source_paths["wildguard"], validation_fraction))
    if "reasoning" not in excluded_sources and source_paths["reasoning"].exists():
        consume("nemotron_reasoning_28k", reasoning_records(source_paths["reasoning"], validation_fraction, seed))
    if "nemotron35" not in excluded_sources and source_paths["nemotron35"].exists():
        consume("nemotron35_selected", nemotron35_records(source_paths["nemotron35"], validation_fraction, selected_content_hashes))

    integrity_blockers: list[str] = []
    content_groups: dict[str, dict[str, set[Any]]] = defaultdict(lambda: {"splits": set(), "labels": set()})
    for split, values in rendered.items():
        for value in values:
            content_id = str(value["content_sha256"])
            content_groups[content_id]["splits"].add(split)
            content_groups[content_id]["labels"].add(
                (
                    value["prompt_safety_label"],
                    value["safety_label"] if value["view"] == "PR" else None,
                )
            )
    cross_split_content = sum(len(group["splits"]) > 1 for group in content_groups.values())
    conflicting_labels = sum(len(group["labels"]) > 1 for group in content_groups.values())
    counters["integrity:cross_split_content_hashes"] = cross_split_content
    counters["integrity:conflicting_label_hashes"] = conflicting_labels
    if cross_split_content:
        integrity_blockers.append(f"{cross_split_content} content hashes occur in both train and validation")
    if conflicting_labels:
        integrity_blockers.append(f"{conflicting_labels} content hashes have conflicting safety labels")
    blockers.extend(integrity_blockers)

    cross_split = split_semantics["train"] & split_semantics["validation"]
    if cross_split:
        raise ValueError(f"Cross-split semantic leakage: {len(cross_split)}")
    benchmark_overlap = sum(value for key, value in counters.items() if key.endswith(":benchmark_overlap"))
    if benchmark_overlap:
        warnings.append(f"removed {benchmark_overlap} examples overlapping configured benchmark tests")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    for split in ("train", "validation"):
        path = output_dir / f"{split}.jsonl"
        rows = sorted(rendered.get(split, []), key=lambda value: stable_hash(f"shuffle:{seed}:{value['example_id']}"))
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for value in rows:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        split_reports[split] = {"path": str(path), "examples": len(rows), "sha256": file_hash(path)}

    required_by_key = {
        "v3": "nemotron_v3_replay",
        "vi": f"vi_{translation_source}",
        "wildguard": "wildguardtrain",
        "reasoning": "nemotron_reasoning_28k",
        "nemotron35": "nemotron35_selected",
    }
    required_sources = {value for key, value in required_by_key.items() if key not in excluded_sources}
    present_sources = {key.removeprefix("semantic_source:") for key in counters if key.startswith("semantic_source:") and ":split:" not in key and counters[key]}
    missing_rendered = sorted(required_sources - present_sources)
    if missing_rendered:
        message = "no rendered examples for: " + ", ".join(missing_rendered)
        blockers.append(message)
        source_blockers.append(message)
    training_ready = not source_blockers and not integrity_blockers and not smoke_per_source
    status = "ready" if training_ready and not blockers else "blocked"
    if training_ready and blockers:
        status = "ready_for_training_eval_pending"
    if smoke_per_source:
        status = "smoke_ready" if not blockers else "smoke_ready_incomplete"
    manifest = {
        "schema_version": 3,
        "status": status,
        "training_ready": training_ready,
        "full_ready": not blockers and not smoke_per_source,
        "translation_source": translation_source,
        "seed": seed,
        "smoke_per_source": smoke_per_source,
        "excluded_sources": sorted(excluded_sources),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "source_revisions": cfg.get("source_revisions", {}),
        "splits": split_reports,
        "counts": dict(sorted(counters.items())),
        "benchmark_hash_count": len(leakage_hashes),
        "blockers": blockers,
        "source_blockers": source_blockers,
        "integrity_blockers": integrity_blockers,
        "warnings": warnings,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_dataset(directory: Path) -> dict[str, Any]:
    seen_examples: dict[str, str] = {}
    semantic_splits: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    content_labels: dict[str, tuple[str, str | None]] = {}
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        for row in read_jsonl(path):
            example_id = str(row.get("example_id") or "")
            semantic_id = str(row.get("semantic_id") or "")
            if not example_id:
                errors.append(f"{path}:{row['_line']}: missing example_id")
            elif example_id in seen_examples:
                errors.append(f"{path}:{row['_line']}: duplicate example_id first seen in {seen_examples[example_id]}")
            else:
                seen_examples[example_id] = f"{split}:{row['_line']}"
            if not semantic_id:
                errors.append(f"{path}:{row['_line']}: missing semantic_id")
            elif semantic_id in semantic_splits and semantic_splits[semantic_id] != split:
                errors.append(f"{path}:{row['_line']}: cross-split semantic_id first seen in {semantic_splits[semantic_id]}")
            else:
                semantic_splits[semantic_id] = split
            content_id = str(row.get("content_sha256") or "")
            label_pair = (
                str(row.get("prompt_safety_label") or ""),
                str(row.get("safety_label") or "") if row.get("view") == "PR" else None,
            )
            if content_id:
                if content_id in content_splits and content_splits[content_id] != split:
                    errors.append(f"{path}:{row['_line']}: cross-split content_sha256 first seen in {content_splits[content_id]}")
                else:
                    content_splits[content_id] = split
                if content_id in content_labels and content_labels[content_id] != label_pair:
                    errors.append(
                        f"{path}:{row['_line']}: conflicting labels for content_sha256 "
                        f"{content_labels[content_id]} vs {label_pair}"
                    )
                else:
                    content_labels[content_id] = label_pair
            for field in ("instruction", "target", "prompt", "safety_label", "view", "taxonomy_mode", "thinking_mode"):
                if row.get(field) in (None, ""):
                    errors.append(f"{path}:{row['_line']}: missing {field}")
            if row.get("view") not in {"P", "PR"}:
                errors.append(f"{path}:{row['_line']}: invalid view")
            if row.get("taxonomy_mode") == "off" and "Safety Categories" in str(row.get("target")):
                errors.append(f"{path}:{row['_line']}: taxonomy-off target contains categories")
            target = str(row.get("target") or "")
            if row.get("thinking_mode") == "think":
                if target.startswith("<think>") or "</think>" not in target:
                    errors.append(f"{path}:{row['_line']}: THINK target violates native Qwen completion contract")
                final = target.rsplit("</think>", 1)[-1].strip()
            else:
                if target.startswith("<think>") or "</think>" in target:
                    errors.append(f"{path}:{row['_line']}: NO-THINK target contains think tags")
                final = target.strip()
            try:
                parsed = json.loads(final)
                if not isinstance(parsed, dict):
                    raise ValueError("target JSON is not an object")
                any_unsafe = row.get("prompt_safety_label") == "unsafe" or (
                    row.get("view") == "PR" and row.get("safety_label") == "unsafe"
                )
                if not any_unsafe and "Safety Categories" in parsed:
                    errors.append(f"{path}:{row['_line']}: all-safe target must omit Safety Categories")
                if row.get("taxonomy_mode") == "on" and any_unsafe and "Safety Categories" not in parsed:
                    errors.append(f"{path}:{row['_line']}: taxonomy-on unsafe target is missing Safety Categories")
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path}:{row['_line']}: invalid final target JSON: {exc}")
            counts[f"split:{split}"] += 1
            counts[f"source:{row.get('dataset_source')}"] += 1
            counts[f"taxonomy:{row.get('taxonomy_mode')}"] += 1
            counts[f"thinking:{row.get('thinking_mode')}"] += 1
    counts["unique_semantic_examples"] = len(semantic_splits)
    counts["rendered_examples"] = len(seen_examples)
    return {"valid": not errors, "counts": dict(sorted(counts.items())), "error_count": len(errors), "errors": errors[:100]}
