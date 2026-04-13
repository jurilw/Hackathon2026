#!/usr/bin/env python3
"""
Verify that public competition .h5ad files do not leak ground-truth age for the test split.

Checks these objects relative to the chosen data root (see --data-dir). For each
logical file, uses ``*_public.h5ad`` if present (output of ``strip_test_age_h5ad.py``),
otherwise the canonical name (e.g. ``train.h5ad``).

If ``pseudobulk/train_pseudobulk_donor_aggregated*.h5ad`` (and val/test) exist, only
those three pseudobulk files are checked, plus any of ``train.h5ad`` / ``val.h5ad`` /
``test.h5ad`` present at the root (no ``combined`` or combined long-form pseudobulk
required). Otherwise the legacy trio is used: combined cell-level + combined
pseudobulk + single donor-aggregated file.

Default data root: prefers ``data/``, then ``data_prep/output_public/`` (stripped
``*_public.h5ad``), then ``data_prep/output/``, else ``data/``.

For every row with obs['_split'] == 'test', column 'age' must be entirely missing/NaN
(if the column exists). Any finite age value on test rows is treated as a leak.

Requires: scanpy, pandas, numpy (same as teaching notebooks).

Usage:
  python scripts/check_test_age_withheld.py
  python scripts/check_test_age_withheld.py --data-dir /path/to/data

Exit codes:
  0 — all present files pass (no test-age leak)
  1 — missing file, missing _split, or leak detected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc


def _default_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    prep_out = root / "data_prep" / "output"
    out_pub = root / "data_prep" / "output_public"
    if (data / "combined.h5ad").is_file() or (data / "combined_public.h5ad").is_file():
        return data
    if (out_pub / "combined_public.h5ad").is_file() or (out_pub / "combined.h5ad").is_file():
        return out_pub
    if (prep_out / "combined.h5ad").is_file() or (prep_out / "combined_public.h5ad").is_file():
        return prep_out
    return data


def _h5ad_path(data_dir: Path, rel: Path, name_suffix: str) -> Path:
    """Prefer ``{stem}{suffix}.h5ad`` if that file exists, else ``rel``."""
    tagged = rel.with_name(f"{rel.stem}{name_suffix}{rel.suffix}")
    p_tag = data_dir / tagged
    p_std = data_dir / rel
    if p_tag.is_file():
        return p_tag
    return p_std


def _test_mask(obs: pd.DataFrame) -> pd.Series:
    if "_split" not in obs.columns:
        raise ValueError("obs has no '_split' column")
    return obs["_split"].astype(str).str.strip().str.lower() == "test"


def _ages_for_test(obs: pd.DataFrame, test_mask: pd.Series) -> pd.Series | None:
    for col in ("age", "Age", "AGE"):
        if col in obs.columns:
            return pd.to_numeric(obs.loc[test_mask, col], errors="coerce")
    return None


def check_h5ad(path: Path, label: str, *, train_or_val_only: bool = False) -> tuple[bool, list[str]]:
    """Return (ok, lines of report)."""
    lines: list[str] = []
    if not path.is_file():
        lines.append(f"{label}: ERROR — file not found: {path}")
        return False, lines

    adata = sc.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        n_rows = obs.shape[0]
        try:
            tm = _test_mask(obs)
        except ValueError as e:
            lines.append(f"{label}: ERROR — {e}")
            return False, lines
        n_test = int(tm.sum())
        lines.append(f"{label}: {path.name} — {n_rows} rows, {n_test} with _split=='test'")

        if n_test == 0:
            if train_or_val_only:
                lines.append("  OK — train-only or val-only file (no test donors).")
                return True, lines
            lines.append("  WARNING: no test split rows; check file version.")
            return True, lines

        ages = _ages_for_test(obs, tm)
        if ages is None:
            lines.append("  OK — no 'age' column (nothing to leak on test).")
            return True, lines

        vals = ages.to_numpy(dtype=float)
        finite = np.isfinite(vals)
        n_leak = int(finite.sum())
        if n_leak > 0:
            lines.append(f"  FAIL — {n_leak} test rows have finite age values (leak).")
            return False, lines

        lines.append("  OK — 'age' column exists but all test values are NaN.")
        return True, lines
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder with combined*.h5ad and pseudobulk/ (default: auto data/, output_public, or data_prep/output/)",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="_public",
        help="Prefer files named {stem}{suffix}.h5ad when present (default: _public). Use '' to only check canonical names.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir()
    sfx = args.name_suffix

    files: Optional[list[tuple[str, Path, bool]]] = None
    for pb in (Path("pseudobulk"), Path("scRNA-seq_pseudobulk")):
        split_agg = [
            pb / "train_pseudobulk_donor_aggregated.h5ad",
            pb / "val_pseudobulk_donor_aggregated.h5ad",
            pb / "test_pseudobulk_donor_aggregated.h5ad",
        ]
        split_paths = [_h5ad_path(data_dir, r, sfx) for r in split_agg]
        if not all(p.is_file() for p in split_paths):
            continue
        files = [
            ("Pseudobulk donor-aggregated (train)", split_paths[0], True),
            ("Pseudobulk donor-aggregated (val)", split_paths[1], True),
            ("Pseudobulk donor-aggregated (test)", split_paths[2], False),
        ]
        for stem, tvo in (("train", True), ("val", True), ("test", False)):
            for rel in (Path("scRNA-seq_raw") / ("%s.h5ad" % stem), Path("%s.h5ad" % stem)):
                p = _h5ad_path(data_dir, rel, sfx)
                if p.is_file():
                    files.append(("Cell-level %s" % stem, p, tvo))
                    break
        print("Checking split donor-aggregated pseudobulk under %s/ (+ cell-level splits if present).\n" % pb)
        break

    if files is None:
        _pb = Path("pseudobulk")
        rels = [
            Path("combined.h5ad"),
            _pb / "combined_pseudobulk_combined.h5ad",
            _pb / "combined_pseudobulk_donor_aggregated.h5ad",
        ]
        files = [
            ("Cell-level combined", _h5ad_path(data_dir, rels[0], sfx), False),
            ("Pseudobulk combined (donor × cell type)", _h5ad_path(data_dir, rels[1], sfx), False),
            ("Pseudobulk donor-aggregated", _h5ad_path(data_dir, rels[2], sfx), False),
        ]

    assert files is not None

    print(f"Data directory: {data_dir.resolve()}\n")
    all_ok = True
    for label, p, tvo in files:
        ok, lines = check_h5ad(p, label, train_or_val_only=tvo)
        for line in lines:
            print(line)
        print()
        if not ok:
            all_ok = False

    if all_ok:
        print("Summary: PASS — no test-split age leak detected in checked files.")
        return 0
    print("Summary: FAIL — fix data or paths before publishing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
