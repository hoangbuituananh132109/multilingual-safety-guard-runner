#!/usr/bin/env python3
"""Deterministically sample SEA-Instruct-2602 Vietnamese without downloading it all.

The Hugging Face dataset viewer accepts authenticated row-range requests.  We
take one random contiguous window from each equal-sized stratum so that sources
stored in distant parts of the split are represented better than by ``head()``.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from huggingface_hub import get_token


DATASET = "aisingapore/SEA-Instruct-2602"
CONFIG = "Vietnamese"
SPLIT = "train"
TOTAL_ROWS = 1_153_050
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_rows(headers: dict[str, str], offset: int, length: int) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=90)
            response.raise_for_status()
            return response.json()["rows"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch offset={offset}, length={length}") from last_error


def parse_conversations(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
            return value if isinstance(value, list) else []
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return []


def counter_dict(counter: Counter[Any], limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): value for key, value in items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/research/sea_instruct_2602_vi"),
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.window_size <= 0 or args.rows % args.window_size:
        parser.error("--rows must be positive and divisible by --window-size")

    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found; run `hf auth login` first.")

    windows = args.rows // args.window_size
    if windows > TOTAL_ROWS // args.window_size:
        parser.error("Requested sample is too large")

    rng = random.Random(args.seed)
    stratum_width = TOTAL_ROWS / windows
    offsets: list[int] = []
    for index in range(windows):
        start = int(index * stratum_width)
        end = min(int((index + 1) * stratum_width), TOTAL_ROWS)
        max_offset = max(start, end - args.window_size)
        offsets.append(rng.randint(start, max_offset))

    headers = {"Authorization": f"Bearer {token}"}
    rows_by_offset: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_rows, headers, offset, args.window_size): offset
            for offset in offsets
        }
        for number, future in enumerate(as_completed(futures), start=1):
            offset = futures[future]
            rows_by_offset[offset] = future.result()
            if number % 10 == 0 or number == windows:
                print(
                    f"Fetched {number * args.window_size}/{args.rows} rows "
                    f"({number}/{windows} windows)",
                    flush=True,
                )
    sampled = [item for offset in offsets for item in rows_by_offset[offset]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / f"sample_{args.rows}_seed{args.seed}.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for item in sampled:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    fields = {
        "sensitivity": "prompt_sensitivity",
        "source": "source",
        "primary_language": "prompt_primary_language",
        "language_category": "prompt_language_category",
        "primary_task": "prompt_primary_task",
        "primary_domain": "prompt_primary_domain",
        "region_scope": "prompt_region_scope",
        "local_cultural_knowledge": "prompt_requires_local_cultural_knowledge",
        "translated_from_english": "prompt_translated_from_english",
        "system_present": "system_present",
    }
    counters = {name: Counter() for name in fields}
    sensitivity_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    role_patterns: Counter[str] = Counter()
    parse_failures = 0
    ids: list[str] = []
    prompt_lengths: list[int] = []
    assistant_lengths: list[int] = []

    for item in sampled:
        row = item["row"]
        for name, field in fields.items():
            counters[name][row.get(field)] += 1
        sensitivity = str(row.get("prompt_sensitivity"))
        sensitivity_by_source[str(row.get("source"))][sensitivity] += 1
        sensitivity_by_language[str(row.get("prompt_primary_language"))][sensitivity] += 1
        ids.append(str(row.get("conversations_id")))

        conversations = parse_conversations(row.get("conversations"))
        if not conversations:
            parse_failures += 1
            continue
        role_patterns["->".join(str(turn.get("role")) for turn in conversations)] += 1
        for turn in conversations:
            content = turn.get("content")
            if not isinstance(content, str):
                continue
            if turn.get("role") == "user":
                prompt_lengths.append(len(content))
            elif turn.get("role") == "assistant":
                assistant_lengths.append(len(content))

    def length_summary(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean_chars": 0.0, "p50_chars": 0, "p95_chars": 0}
        ordered = sorted(values)
        return {
            "count": len(values),
            "mean_chars": round(sum(values) / len(values), 2),
            "p50_chars": ordered[int(0.50 * (len(ordered) - 1))],
            "p95_chars": ordered[int(0.95 * (len(ordered) - 1))],
        }

    summary = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "population_rows": TOTAL_ROWS,
        "sample_rows": len(sampled),
        "sampling": {
            "method": "one deterministic random window per equal-sized stratum",
            "seed": args.seed,
            "window_size": args.window_size,
            "windows": windows,
            "offsets": offsets,
        },
        "duplicate_conversation_ids": len(ids) - len(set(ids)),
        "conversation_parse_failures": parse_failures,
        "distributions": {
            name: counter_dict(counter, 30 if name in {"source", "primary_task", "primary_domain"} else None)
            for name, counter in counters.items()
        },
        "sensitivity_by_source": {
            source: counter_dict(counter)
            for source, counter in sorted(
                sensitivity_by_source.items(), key=lambda pair: -sum(pair[1].values())
            )[:30]
        },
        "sensitivity_by_primary_language": {
            language: counter_dict(counter)
            for language, counter in sorted(
                sensitivity_by_language.items(), key=lambda pair: -sum(pair[1].values())
            )
        },
        "conversation_role_patterns": counter_dict(role_patterns),
        "user_turn_lengths": length_summary(prompt_lengths),
        "assistant_turn_lengths": length_summary(assistant_lengths),
    }
    summary_path = args.output_dir / f"summary_{args.rows}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sample:  {sample_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
