"""Convert the gated WildGuardTrain parquet into the JSONL contract used by stage2.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pq.read_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in table.to_pylist():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": table.num_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
