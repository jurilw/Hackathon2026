#!/usr/bin/env python3
"""
Convert h5ad (cell-level) to pseudobulk: sum counts per (donor, celltype).

Output: one AnnData per cell type (donors x genes) + one combined (donor,celltype) x genes.
Preserves age and other donor-level metadata when available.

Usage:
  python data_prep/h5ad_to_pseudobulk.py data_prep/output/train.h5ad
  python data_prep/h5ad_to_pseudobulk.py data_prep/output/combined.h5ad -o data_prep/output/pseudobulk
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Built-in cell-type mappings
# ---------------------------------------------------------------------------

# Maps AIDA author_cell_type labels → the 5 broad Onek1K cell types.
# Cells whose type is not listed here are DROPPED when this map is applied.
BUILTIN_CELLTYPE_MAPS = {
    "onek1k": {
        "CD4_T": [
            "CD4+_T_naive", "CD4+_T_cm", "CD4+_T_em",
            "CD4+_T_cyt", "CD4+_T_unknown", "Treg",
        ],
        "CD8_T": [
            "CD8+_T_naive", "CD8+_T_GZMBhi", "CD8+_T_GZMKhi",
            "CD8+_T_unknown", "MAIT", "gdT_GZMBhi", "gdT_GZMKhi",
            "dnT",
        ],
        "NK": ["CD16+_NK", "CD56+_NK", "NK_unknown", "ILC"],
        "B_cells": [
            "naive_B", "memory_B_IGHMhi", "memory_B_IGHMlo",
            "memory_B_unknown", "atypical_B", "B_unknown", "Plasma_Cell",
        ],
        "monocytes": [
            "CD14+_Monocyte", "CD16+_Monocyte",
            "cDC1", "cDC2", "DC_SIGLEC6hi", "pDC", "Myeloid_unknown",
        ],
    }
}


def load_celltype_map(spec: str) -> dict:
    """
    Load a cell-type mapping from:
      - a built-in name (e.g. 'onek1k')
      - a JSON file path   { "target_type": ["source_type1", ...], ... }
    Returns inverted dict: source_label -> target_label.
    """
    if spec in BUILTIN_CELLTYPE_MAPS:
        raw = BUILTIN_CELLTYPE_MAPS[spec]
    else:
        p = Path(spec)
        if not p.exists():
            raise FileNotFoundError("Cell-type map not found: %s" % spec)
        with open(p) as fh:
            raw = json.load(fh)

    # Invert: source_label -> target_label
    inverted = {}
    for target, sources in raw.items():
        for src in sources:
            if src in inverted:
                raise ValueError("Source cell type '%s' mapped to both '%s' and '%s'" % (
                    src, inverted[src], target))
            inverted[src] = target
    return inverted


def apply_celltype_map(obs: pd.DataFrame, celltype_col: str, inverted_map: dict) -> pd.DataFrame:
    """
    Remap celltype_col values using inverted_map (source -> target).
    Rows whose source label is NOT in the map are DROPPED.
    Prints a summary of cells kept/dropped per original type.
    """
    original_col = obs[celltype_col].astype(str)
    mapped = original_col.map(inverted_map)           # NaN where not in map
    n_total  = len(obs)
    n_kept   = mapped.notna().sum()
    n_dropped = n_total - n_kept

    dropped_types = original_col[mapped.isna()].value_counts()
    if n_dropped:
        print("  Cell-type map: dropping %d / %d cells (%.1f%%) — unmapped types:" % (
            n_dropped, n_total, 100 * n_dropped / n_total))
        for ct, cnt in dropped_types.items():
            print("    %-30s  %d" % (ct, cnt))

    obs = obs[mapped.notna()].copy()
    obs[celltype_col] = mapped[mapped.notna()].values
    print("  Cell-type map: kept %d cells across %d target types" % (
        n_kept, obs[celltype_col].nunique()))
    return obs


def _sparse_groupsum(X, group_labels):
    """
    Sum rows of X (sparse or dense) by group label using sparse matrix multiply.
    Never densifies the full X; only the small result (n_groups x n_genes) is dense.
    Returns (X_agg_dense, unique_labels_in_order).
    """
    labels = np.asarray(group_labels)
    unique = list(dict.fromkeys(labels))          # ordered unique, first-occurrence order
    label_to_idx = {g: i for i, g in enumerate(unique)}
    row_idx = np.array([label_to_idx[g] for g in labels])
    col_idx = np.arange(len(labels))
    agg_mat = csr_matrix(
        (np.ones(len(labels), dtype=np.float32), (row_idx, col_idx)),
        shape=(len(unique), len(labels)),
    )
    X_sp = csr_matrix(X) if not issparse(X) else X
    result = agg_mat.dot(X_sp)                    # (n_groups x n_genes), sparse
    return np.asarray(result.todense(), dtype=np.float64), unique


def h5ad_to_pseudobulk(adata, donor_col="donor_id", celltype_col="celltype", layer=None):
    """
    Aggregate cell-level counts to pseudobulk per (donor, celltype).
    Returns dict: celltype -> AnnData (donors x genes, sum of counts).
    Never densifies the full X matrix.
    """
    if donor_col not in adata.obs.columns:
        raise ValueError("obs must have '%s'" % donor_col)
    if celltype_col not in adata.obs.columns:
        raise ValueError("obs must have '%s'" % celltype_col)

    X = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if not issparse(X):
        X = csr_matrix(X)

    obs = adata.obs
    donors    = obs[donor_col].astype(str).values
    celltypes = obs[celltype_col].astype(str).values

    donor_meta_cols = [c for c in obs.columns if c not in [donor_col, celltype_col]]
    donor_meta = obs.groupby(obs[donor_col].astype(str), observed=True).first()

    result = {}
    for ct in np.unique(celltypes):
        mask = celltypes == ct
        donors_ct = donors[mask]
        X_ct = X[mask]                            # sparse slice (n_ct_cells x n_genes)

        # Sparse group-sum: donors x genes (dense, small)
        X_pb, donors_u = _sparse_groupsum(X_ct, donors_ct)

        obs_pb = pd.DataFrame({donor_col: donors_u})
        for c in donor_meta_cols:
            if c in donor_meta.columns:
                obs_pb[c] = obs_pb[donor_col].map(donor_meta[c].to_dict())
        obs_pb["celltype"] = ct
        obs_pb["n_cells"]  = np.array([(donors_ct == d).sum() for d in donors_u])

        ad_pb = sc.AnnData(X=X_pb, obs=obs_pb, var=adata.var.copy())
        ad_pb.obs_names    = ad_pb.obs[donor_col].astype(str) + "_" + ct
        ad_pb.obs.index.name = None
        result[ct] = ad_pb

    return result


def aggregate_to_donor_level(adata, donor_col="donor_id", celltype_col="celltype"):
    """
    Reshape (donor, celltype) x genes -> donors x (genes x n_cell_types).
    One row per donor; features are named gene__celltype.
    Works with dense or sparse X without densifying the full matrix.
    """
    obs = adata.obs
    donors    = obs[donor_col].astype(str).values
    celltypes = obs[celltype_col].astype(str).values
    genes     = adata.var_names.tolist()
    ct_order  = sorted(set(celltypes))

    donor_ids    = sorted(set(donors))
    n_donors     = len(donor_ids)
    n_genes      = len(genes)
    n_ct         = len(ct_order)

    X_agg = np.zeros((n_donors, n_genes * n_ct), dtype=np.float64)
    donor_to_idx = {d: i for i, d in enumerate(donor_ids)}
    ct_to_idx    = {ct: i for i, ct in enumerate(ct_order)}

    X = adata.X
    for row_i in range(adata.n_obs):
        d  = donors[row_i]
        ct = celltypes[row_i]
        if d not in donor_to_idx or ct not in ct_to_idx:
            continue
        d_idx  = donor_to_idx[d]
        ct_idx = ct_to_idx[ct]
        x_row  = X[row_i]
        if issparse(x_row):
            x_row = np.asarray(x_row.todense()).flatten()
        else:
            x_row = np.asarray(x_row).flatten()
        X_agg[d_idx, ct_idx * n_genes : (ct_idx + 1) * n_genes] = x_row

    ct_safe       = [c.replace(" ", "_").replace("/", "_") for c in ct_order]
    feature_names = ["%s__%s" % (g, ct) for g in genes for ct in ct_safe]

    meta    = obs.groupby(obs[donor_col].astype(str), observed=True).first()
    obs_agg = meta.reindex(donor_ids).copy()
    if donor_col not in obs_agg.columns:
        obs_agg[donor_col] = obs_agg.index.astype(str)

    var_agg = pd.DataFrame(index=feature_names)
    ad_agg  = sc.AnnData(X=X_agg, obs=obs_agg, var=var_agg)
    ad_agg.obs_names    = obs_agg[donor_col].astype(str)
    ad_agg.obs.index.name = None
    return ad_agg


def main():
    parser = argparse.ArgumentParser(description="Convert h5ad to pseudobulk per cell type")
    parser.add_argument("input", type=str, help="Input h5ad path")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Output directory (default: same dir as input)")
    parser.add_argument("--donor-col",    type=str, default="donor_id")
    parser.add_argument("--celltype-col", type=str, default="celltype")
    parser.add_argument("--layer",        type=str, default=None,
                        help="Use this layer instead of X (e.g. 'counts' for raw)")
    parser.add_argument("--combined-only", action="store_true",
                        help="Save only combined file, not per-celltype")
    parser.add_argument("--celltype-map", type=str, default=None,
                        help="Remap cell types before aggregation. "
                             "Use 'onek1k' for the built-in AIDA→Onek1K 5-type mapping, "
                             "or provide a path to a JSON file {target: [source, ...]}. "
                             "Cells not covered by the map are dropped. "
                             "Output files get a '_mapped_NAME' suffix.")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_dir = Path(args.output_dir) if args.output_dir else inp.parent / "pseudobulk"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading %s..." % inp)
    adata = sc.read_h5ad(inp)

    # Apply cell-type remapping if requested
    map_suffix = ""
    if args.celltype_map:
        map_name = Path(args.celltype_map).stem if Path(args.celltype_map).exists() else args.celltype_map
        print("Applying cell-type map '%s' ..." % args.celltype_map)
        inverted_map = load_celltype_map(args.celltype_map)
        # Remap obs; this may drop rows, so we must also subset adata.X accordingly
        new_obs = apply_celltype_map(adata.obs.copy(), args.celltype_col, inverted_map)
        keep_idx = adata.obs.index.isin(new_obs.index)
        adata = adata[keep_idx].copy()
        adata.obs[args.celltype_col] = new_obs.loc[adata.obs.index, args.celltype_col].values
        map_suffix = "_mapped_%s" % map_name
        # Save the mapping JSON alongside outputs for reference
        map_raw = BUILTIN_CELLTYPE_MAPS.get(args.celltype_map, inverted_map)
        with open(out_dir / ("celltype_map_%s.json" % map_name), "w") as fh:
            json.dump(map_raw, fh, indent=2)
        print("  Mapping saved to celltype_map_%s.json" % map_name)

    print("Converting to pseudobulk (sum per donor, celltype)...")
    pb_by_ct = h5ad_to_pseudobulk(
        adata,
        donor_col=args.donor_col,
        celltype_col=args.celltype_col,
        layer=args.layer,
    )

    stem = inp.stem + map_suffix
    if not args.combined_only:
        for ct, ad_pb in pb_by_ct.items():
            ct_safe = ct.replace(" ", "_").replace("/", "_")
            p = out_dir / ("%s_%s.h5ad" % (stem, ct_safe))
            ad_pb.write_h5ad(p)
            print("  %s: %d donors, %d genes -> %s" % (ct, ad_pb.n_obs, ad_pb.n_vars, p.name))

    # Combined: (donor, celltype) x genes
    ad_combined = sc.concat(list(pb_by_ct.values()), axis=0, join="inner")
    ad_combined.obs.index.name = None

    # Add n_cells_{celltype} columns
    donor_col = args.donor_col
    for ct, ad_pb in pb_by_ct.items():
        ct_safe  = ct.replace(" ", "_").replace("/", "_")
        col_name = "n_cells_" + ct_safe
        donor_to_n = {str(k): v for k, v in ad_pb.obs.set_index(donor_col)["n_cells"].to_dict().items()}
        ad_combined.obs[col_name] = (
            ad_combined.obs[donor_col].astype(str).map(donor_to_n).fillna(0).astype(int)
        )

    p_comb = out_dir / ("%s_pseudobulk_combined.h5ad" % stem)
    ad_combined.write_h5ad(p_comb)
    print("  Combined: %d samples (donor x celltype), %d genes -> %s" % (
        ad_combined.n_obs, ad_combined.n_vars, p_comb.name))

    # Donor-aggregated: one row per donor, one value per (gene, celltype) pair
    ad_agg = aggregate_to_donor_level(
        ad_combined, donor_col=args.donor_col, celltype_col=args.celltype_col
    )
    p_agg = out_dir / ("%s_pseudobulk_donor_aggregated.h5ad" % stem)
    ad_agg.write_h5ad(p_agg)
    print("  Donor-aggregated: %d donors x %d features -> %s" % (
        ad_agg.n_obs, ad_agg.n_vars, p_agg.name))

    print("Done. Output dir: %s" % out_dir)


if __name__ == "__main__":
    main()
