#!/usr/bin/env python3
"""
Evaluate your model predictions against the validation set ground truth.

Competitors self-evaluate on the VALIDATION set using this script.
The test set is scored by the organisers after submission.

Usage:
  # Evaluate the latest training run's val predictions automatically:
  python models/evaluate_val.py

  # Specify a predictions file:
  python models/evaluate_val.py --predictions models/output/TIMESTAMP/val_predictions.csv

  # Save a scatter plot:
  python models/evaluate_val.py --plot

Output: MAE, RMSE, R², Pearson r, Spearman rho on the validation set.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

PROJ_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_BASE = PROJ_ROOT / "models" / "output"


def _latest_val_predictions():
    """Find val_predictions.csv in the most recent timestamped run directory."""
    if OUTPUT_BASE.exists():
        for d in sorted(OUTPUT_BASE.iterdir(), reverse=True):
            if d.is_dir():
                for name in ["val_predictions.csv", "pseudobulk_val_predictions.csv",
                             "pseudobulk_sex_val_predictions.csv"]:
                    p = d / name
                    if p.exists():
                        return str(p)
    return None


def _load_val_truth(h5ad_path: str | None) -> pd.DataFrame:
    """
    Load val ground truth from val.h5ad (donor_id, age) or from a manually specified CSV.
    """
    if h5ad_path:
        adata = sc.read_h5ad(h5ad_path)
        df = adata.obs[["donor_id", "age"]].drop_duplicates("donor_id").copy()
        df["donor_id"] = df["donor_id"].astype(str)
        return df

    # Auto-discover val.h5ad relative to this script
    for candidate in [
        PROJ_ROOT / "data_prep" / "output" / "val.h5ad",
        PROJ_ROOT / "data" / "val.h5ad",
    ]:
        if candidate.exists():
            adata = sc.read_h5ad(str(candidate))
            df = adata.obs[["donor_id", "age"]].drop_duplicates("donor_id").copy()
            df["donor_id"] = df["donor_id"].astype(str)
            return df

    raise FileNotFoundError(
        "Could not find val.h5ad. Specify --val-h5ad or --truth-csv."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate validation predictions (competitors use this, not the test scorer)"
    )
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to val_predictions.csv. Default: latest run in models/output/")
    parser.add_argument("--val-h5ad", type=str, default=None,
                        help="Path to val.h5ad to extract ground-truth ages. "
                             "Default: auto-discovered from data_prep/output/ or data/.")
    parser.add_argument("--truth-csv", type=str, default=None,
                        help="Alternative: provide ground-truth as CSV (donor_id, age).")
    parser.add_argument("--plot", action="store_true",
                        help="Save a predicted-vs-true scatter plot next to the predictions file.")
    args = parser.parse_args()

    # Locate predictions
    pred_path = args.predictions or _latest_val_predictions()
    if pred_path is None:
        raise FileNotFoundError(
            "No val_predictions.csv found in models/output/. "
            "Run train_age_model.py first, or pass --predictions PATH."
        )
    print("Predictions : %s" % pred_path)
    pred_df = pd.read_csv(pred_path)
    id_col  = "donor_id" if "donor_id" in pred_df.columns else "sample_id"
    pred_df[id_col] = pred_df[id_col].astype(str)

    # Load ground truth
    if args.truth_csv:
        truth_df = pd.read_csv(args.truth_csv)
        truth_df["donor_id"] = truth_df["donor_id"].astype(str)
    else:
        truth_df = _load_val_truth(args.val_h5ad)

    # Merge
    merged = pred_df.merge(truth_df, left_on=id_col, right_on="donor_id", how="inner")
    if merged.empty:
        raise ValueError(
            "No matching donor_ids between predictions (%s) and val ground truth. "
            "Check column names." % pred_path
        )

    y_true = merged["age"].values
    y_pred = merged["predicted_age"].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)

    print("\nValidation set metrics  (n=%d donors)" % len(y_true))
    print("  MAE:      %.3f years" % mae)
    print("  RMSE:     %.3f years" % rmse)
    print("  R\u00b2:       %.3f" % r2)
    print("  Pearson:  %.3f  (p=%.2e)" % (pr, pp))
    print("  Spearman: %.3f  (p=%.2e)" % (sr, sp))

    # Save metrics CSV alongside predictions
    out_dir = Path(pred_path).parent
    metrics_df = pd.DataFrame([{
        "metric": m, "value": v
    } for m, v in [("MAE", mae), ("RMSE", rmse), ("R2", r2),
                   ("Pearson", pr), ("Spearman", sr)]])
    metrics_path = out_dir / "val_eval_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMetrics saved to %s" % metrics_path)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
            lim = [min(y_true.min(), y_pred.min()) - 2,
                   max(y_true.max(), y_pred.max()) + 2]
            ax.plot(lim, lim, "r--", label="y=x (perfect)")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("True age (years)"); ax.set_ylabel("Predicted age (years)")
            ax.set_title("Validation: predicted vs true  (MAE=%.2f yr)" % mae)
            ax.legend(); ax.set_aspect("equal"); plt.tight_layout()
            plot_path = out_dir / "val_eval_plot.png"
            plt.savefig(plot_path, dpi=150); plt.close()
            print("Plot saved to %s" % plot_path)
        except ImportError:
            print("Install matplotlib for --plot: pip install matplotlib")


if __name__ == "__main__":
    main()
