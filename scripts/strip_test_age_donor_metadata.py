#!/usr/bin/env python3
"""
Write a copy of donor_metadata.csv with ``age`` cleared for rows where ``split`` is
``test`` (public / competition release). Train and val ages unchanged.

Uses only the standard library (no pandas).

Usage:
  python scripts/strip_test_age_donor_metadata.py INPUT.csv OUTPUT.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    src = args.input_csv
    dst = args.output_csv
    if not src.is_file():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 1

    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("ERROR: empty CSV", file=sys.stderr)
            return 1
        if "split" not in fieldnames:
            print("ERROR: expected a 'split' column", file=sys.stderr)
            return 1
        has_age = "age" in fieldnames
        rows = []
        for row in reader:
            if has_age and str(row.get("split", "")).strip().lower() == "test":
                row["age"] = ""
            rows.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {dst} ({len(rows)} donors; test split age cleared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
