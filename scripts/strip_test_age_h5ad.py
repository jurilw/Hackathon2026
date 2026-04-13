#!/usr/bin/env python3
"""
Copy the three competition AnnData objects and set ``age`` to NaN for rows with
``obs['_split'] == 'test'``. Train/val ages are unchanged.

Input layout (under ``--input-dir``):
  - combined.h5ad
  - pseudobulk/combined_pseudobulk_combined.h5ad
  - pseudobulk/combined_pseudobulk_donor_aggregated.h5ad

Writes under ``--output-dir`` with a **public** name tag (default: ``*_public.h5ad``), e.g.
``combined_public.h5ad``, ``pseudobulk/combined_pseudobulk_combined_public.h5ad``. Inputs are not modified.
Use ``--output-name-suffix ''`` to keep the original base names in the output folder.

Large ``combined.h5ad`` is opened in backed mode so the expression matrix stays
on disk; only ``obs`` is updated in memory before writing a new file.

Usage:
  python scripts/strip_test_age_h5ad.py
  python scripts/strip_test_age_h5ad.py --input-dir data_prep/output --output-dir data_prep/output_public
  python scripts/strip_test_age_h5ad.py --output-dir data   # after checking sizes / backups

Requires: scanpy, anndata, pandas, numpy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_input_dir() -> Path:
    root = _repo_root()
    data = root / "data"
    prep_out = root / "data_prep" / "output"
    if (data / "combined.h5ad").is_file():
        return data
    if (prep_out / "combined.h5ad").is_file():
        return prep_out
    return prep_out


def _default_output_dir(root: Path) -> Path:
    """Prefer ``data_prep/output_public`` next to pipeline output; else ``data_stripped/`` at repo root."""
    prep_out = root / "data_prep" / "output"
    if prep_out.is_dir():
        return prep_out.parent / "output_public"
    return root / "data_stripped"


def _output_rel(src_rel: Path, name_suffix: str) -> Path:
    """Map input relative path to output filename (e.g. ``combined.h5ad`` -> ``combined_public.h5ad``)."""
    if not name_suffix:
        return src_rel
    return src_rel.with_name(f"{src_rel.stem}{name_suffix}{src_rel.suffix}")


def _test_mask(obs: pd.DataFrame) -> pd.Series:
    if "_split" not in obs.columns:
        raise ValueError("obs has no '_split' column")
    return obs["_split"].astype(str).str.strip().str.lower() == "test"


def _strip_age_test_rows(adata: AnnData) -> tuple[int, int]:
    """
    Set ``age`` to NaN on test rows. Returns (n_test_rows, n_cleared_finite_age).
    """
    obs = adata.obs.copy()
    tm = _test_mask(obs)
    n_test = int(tm.sum())
    if n_test == 0 or "age" not in obs.columns:
        return n_test, 0

    age = pd.to_numeric(obs["age"], errors="coerce")
    n_clear = int(age.loc[tm].notna().sum())
    age.loc[tm] = np.nan
    obs["age"] = age
    adata.obs = obs
    return n_test, n_clear


def _process_one(in_path: Path, out_path: Path, *, large: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if large:
        adata = sc.read_h5ad(in_path, backed="r")
    else:
        adata = sc.read_h5ad(in_path)
    try:
        n_test, n_clear = _strip_age_test_rows(adata)
        print(
            f"  {in_path.name}: test rows={n_test}, cleared finite age on test={n_clear} -> {out_path.name}"
        )
        adata.write_h5ad(out_path)
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing combined.h5ad and pseudobulk/ (default: auto data/ or data_prep/output/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination root (default: data_prep/output_public if data_prep/output exists, else data_stripped/)",
    )
    parser.add_argument(
        "--combined-mb-threshold",
        type=float,
        default=500.0,
        help="Use backed mode for inputs larger than this many MB (default: 500)",
    )
    parser.add_argument(
        "--output-name-suffix",
        type=str,
        default="_public",
        help="Insert before .h5ad on outputs (default: _public). Use empty string for same names as inputs.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else _default_input_dir()
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(root)
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    rel_files = [
        Path("combined.h5ad"),
        Path("pseudobulk") / "combined_pseudobulk_combined.h5ad",
        Path("pseudobulk") / "combined_pseudobulk_donor_aggregated.h5ad",
    ]

    suffix = args.output_name_suffix
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Output filename suffix before .h5ad: {repr(suffix) if suffix else '(none — same names as inputs)'}\n")
    for rel in rel_files:
        src = input_dir / rel
        out_rel = _output_rel(rel, suffix)
        dst = output_dir / out_rel
        if not src.is_file():
            print(f"ERROR: missing input: {src}", file=sys.stderr)
            return 1
        mb = src.stat().st_size / (1024 * 1024)
        large = mb >= args.combined_mb_threshold
        _process_one(src, dst, large=large)

    print("\nDone. Run scripts/check_test_age_withheld.py --data-dir {output_dir} to verify.".format(output_dir=output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
