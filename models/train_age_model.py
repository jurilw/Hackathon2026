#!/usr/bin/env python3
"""
Train age model using separate train/val pseudobulk files (no test required).

This version is for competitor-facing workflows where:
- pseudobulk h5ad files do NOT contain age
- labels are provided via CSV files
- evaluation is done on validation only

Usage:
  python models/train_age_model.py \
    --train-h5ad data/scRNA-seq_pseudobulk/train_pseudobulk_donor_aggregated_public.h5ad \
    --val-h5ad   data/scRNA-seq_pseudobulk/val_pseudobulk_donor_aggregated_public.h5ad \
    --train-labels data/metadata/train_age.csv \
    --val-labels   data/metadata/val_age.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def select_genes(X: np.ndarray, feature_names: list[str], n_genes: int = 2000, n_celltypes: int = 5):
    """Select top genes by variance, keeping all cell-type slots per selected gene."""
    n_features = X.shape[1]
    n_genes_total = n_features // n_celltypes
    if n_genes >= n_genes_total:
        return X, feature_names

    var = np.var(X, axis=0)
    gene_vars = []
    for g in range(n_genes_total):
        start = g * n_celltypes
        end = start + n_celltypes
        gene_vars.append(np.mean(var[start:end]))
    gene_vars = np.asarray(gene_vars)
    top_genes = np.argsort(gene_vars)[::-1][:n_genes]

    keep_cols = []
    for g in top_genes:
        keep_cols.extend(range(g * n_celltypes, (g + 1) * n_celltypes))
    return X[:, keep_cols], [feature_names[i] for i in keep_cols]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pr, "Pearson_p": pp, "Spearman": sr, "Spearman_p": sp}


def read_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if not {"donor_id", "age"}.issubset(df.columns):
        raise ValueError(f"{path} must contain donor_id, age columns")
    df["donor_id"] = df["donor_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if df["age"].isna().any():
        raise ValueError(f"{path} contains non-numeric/empty age values")
    return df.drop_duplicates("donor_id").set_index("donor_id")["age"]


def load_matrix_and_labels(h5ad_path: Path, label_path: Path):
    ad = sc.read_h5ad(h5ad_path)
    X = ad.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    feat_names = ad.var_names.tolist()
    donor_ids = ad.obs["donor_id"].astype(str).values

    age_map = read_labels(label_path)
    y = pd.Series(donor_ids).map(age_map).to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        missing = pd.Series(donor_ids)[np.isnan(y)].head(5).tolist()
        raise ValueError(f"Missing labels for donor_id(s), e.g. {missing}, from {label_path}")
    return X, y, donor_ids, feat_names


def main() -> int:
    proj_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train RF age model on separate train/val files (no test).")
    parser.add_argument("--train-h5ad", type=Path, default=proj_root / "data" / "scRNA-seq_pseudobulk" / "train_pseudobulk_donor_aggregated_public.h5ad")
    parser.add_argument("--val-h5ad", type=Path, default=proj_root / "data" / "scRNA-seq_pseudobulk" / "val_pseudobulk_donor_aggregated_public.h5ad")
    parser.add_argument("--train-labels", type=Path, default=proj_root / "data" / "metadata" / "train_age.csv")
    parser.add_argument("--val-labels", type=Path, default=proj_root / "data" / "metadata" / "val_age.csv")
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--show-val-metrics",
        action="store_true",
        help="Compute/print validation metrics during training and save val_metrics.csv. "
             "Default: off (evaluation is expected via models/evaluate_val.py).",
    )
    args = parser.parse_args()

    for p in [args.train_h5ad, args.val_h5ad, args.train_labels, args.val_labels]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    output_base = proj_root / "results"
    run_dir = args.output_dir if args.output_dir else output_base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    print(f"Loading train: {args.train_h5ad}")
    X_train, y_train, train_donors, feat_names = load_matrix_and_labels(args.train_h5ad, args.train_labels)
    print(f"Loading val:   {args.val_h5ad}")
    X_val, y_val, val_donors, _ = load_matrix_and_labels(args.val_h5ad, args.val_labels)
    print(f"Train donors: {len(train_donors)}  Val donors: {len(val_donors)}")

    if not args.all_features:
        print(f"Selecting top {args.n_genes} genes by variance...")
        X_train, feat_names = select_genes(X_train, feat_names, n_genes=args.n_genes)
        # same selected columns for val
        feat_index = {f: i for i, f in enumerate(sc.read_h5ad(args.train_h5ad).var_names.tolist())}
        keep_idx = [feat_index[f] for f in feat_names]
        X_val = X_val[:, keep_idx]
        print(f"Using {X_train.shape[1]} features")
    else:
        print(f"Using all {X_train.shape[1]} features")

    X_train = np.log1p(X_train)
    X_val = np.log1p(X_val)

    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)
    val_pred_df = pd.DataFrame({"donor_id": val_donors, "predicted_age": y_val_pred})
    val_pred_df.to_csv(run_dir / "val_predictions.csv", index=False)
    print(f"\nSaved: {run_dir / 'val_predictions.csv'}")

    if args.show_val_metrics:
        m = compute_metrics(y_val, y_val_pred)
        print("\nValidation metrics:")
        print(f"  MAE:      {m['MAE']:.3f}")
        print(f"  RMSE:     {m['RMSE']:.3f}")
        print(f"  R²:       {m['R2']:.3f}")
        print(f"  Pearson:  {m['Pearson']:.3f} (p={m['Pearson_p']:.2e})")
        print(f"  Spearman: {m['Spearman']:.3f} (p={m['Spearman_p']:.2e})")
        pd.DataFrame([{"metric": k, "value": v} for k, v in [
            ("MAE", m["MAE"]), ("RMSE", m["RMSE"]), ("R2", m["R2"]),
            ("Pearson", m["Pearson"]), ("Spearman", m["Spearman"]),
        ]]).to_csv(run_dir / "val_metrics.csv", index=False)
        print(f"Saved: {run_dir / 'val_metrics.csv'}")

    # Top feature importance aggregated by base gene
    imp = rf.feature_importances_
    feat_imp = {}
    for i, name in enumerate(feat_names):
        gene = name.rsplit("__", 1)[0] if "__" in name else name
        feat_imp[gene] = feat_imp.get(gene, 0.0) + float(imp[i])
    top20 = sorted(feat_imp.items(), key=lambda x: -x[1])[:20]
    top20_df = pd.DataFrame(top20, columns=["feature", "importance"])
    top20_df.insert(0, "rank", range(1, len(top20_df) + 1))
    top20_df.to_csv(run_dir / "top_genes.csv", index=False)
    print(f"Saved: {run_dir / 'top_genes.csv'}")

    import joblib

    joblib.dump(rf, run_dir / "rf_age_model.joblib")
    pd.DataFrame({"feature": feat_names}).to_csv(run_dir / "feature_names.csv", index=False)
    print(f"Saved: {run_dir / 'rf_age_model.joblib'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
