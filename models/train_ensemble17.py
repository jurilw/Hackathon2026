import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

# Suppress the ElasticNet Convergence Warning to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Setup Paths
# ==========================================
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Expanded Ensemble Pipeline (Max Features, Saving Models)...")

# ==========================================
# 2. Load Targets & Metadata
# ==========================================
print("Loading labels and biological metadata...")
y_train = pd.read_csv(DATA_DIR / "metadata/train_age.csv").set_index("donor_id")["age"]
y_val = pd.read_csv(DATA_DIR / "metadata/val_age.csv").set_index("donor_id")["age"]

meta = pd.read_csv(DATA_DIR / "metadata/donor_metadata.csv").set_index("donor_id")
sex_train = meta.loc[y_train.index, ['sex_binary']].fillna(0)
sex_val = meta.loc[y_val.index, ['sex_binary']].fillna(0)

# ==========================================
# 3. Load & Align Feature Matrices
# ==========================================
def load_pseudobulk(split):
    adata = sc.read_h5ad(DATA_DIR / f"scRNA-seq_pseudobulk/{split}_pseudobulk_donor_aggregated_public.h5ad")
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    # Added columns=adata.var_names to preserve biological labels
    return pd.DataFrame(np.log1p(X), index=adata.obs["donor_id"].astype(int), columns=adata.var_names)

def load_geneformer(split):
    df = pd.read_csv(DATA_DIR / f"scRNA-seq_geneformer_pseudobulk/geneformer_pseudobulk_{split}.tsv.gz", sep='\t')
    return df.set_index("donor_id").filter(regex='__emb')

def load_genotype_pcs(split):
    df = pd.read_csv(DATA_DIR / f"genotypes/pca_{split}.tsv", sep='\t').set_index("donor_id")
    return df.select_dtypes(include=['number'])

pb_train = load_pseudobulk("train").loc[y_train.index].fillna(0)
gf_train = load_geneformer("train").loc[y_train.index].fillna(0)
pc_train = load_genotype_pcs("train").loc[y_train.index].fillna(0)

pb_val = load_pseudobulk("val").loc[y_val.index].fillna(0)
gf_val = load_geneformer("val").loc[y_val.index].fillna(0)
pc_val = load_genotype_pcs("val").loc[y_val.index].fillna(0)

test_pb_path = DATA_DIR / "scRNA-seq_pseudobulk/test_pseudobulk_donor_aggregated_public.h5ad"
test_data_available = test_pb_path.exists()

if test_data_available:
    pb_test = load_pseudobulk("test").fillna(0)
    test_donors = pb_test.index
    gf_test = load_geneformer("test").loc[test_donors].fillna(0)
    pc_test = load_genotype_pcs("test").loc[test_donors].fillna(0)
    sex_test = meta.loc[test_donors, ['sex_binary']].fillna(0)

# ==========================================
# 4. Feature Selection
# ==========================================
print("Running feature selection...")
pb_selector = Pipeline([
    ('variance', VarianceThreshold(threshold=0.01)),
    # Maximize features: k='all' bypasses the f_regression cutoff
    ('kbest', SelectKBest(score_func=f_regression, k='all')) 
])

pb_train_genes = pb_selector.fit_transform(pb_train, y_train)
pb_val_genes = pb_selector.transform(pb_val)

pb_train_sel = np.column_stack((pb_train_genes, sex_train.values))
pb_val_sel = np.column_stack((pb_val_genes, sex_val.values))

gf_train_sel = np.column_stack((gf_train.values, sex_train.values))
gf_val_sel = np.column_stack((gf_val.values, sex_val.values))

pc_train_sel = np.column_stack((pc_train.values, sex_train.values))
pc_val_sel = np.column_stack((pc_val.values, sex_val.values))

if test_data_available:
    pb_test_sel = np.column_stack((pb_selector.transform(pb_test), sex_test.values))
    gf_test_sel = np.column_stack((gf_test.values, sex_test.values))
    pc_test_sel = np.column_stack((pc_test.values, sex_test.values))

# ==========================================
# 5. Train Base Models
# ==========================================

# --- GROUP A: Pseudobulk Experts ---
print("\n--- Training Pseudobulk Models (XGB, LGBM, RF, ElasticNet) ---")
model_xgb = XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.05, early_stopping_rounds=20, random_state=42, n_jobs=-1)
model_xgb.fit(pb_train_sel, y_train, eval_set=[(pb_val_sel, y_val)], verbose=False)
pred_val_xgb = model_xgb.predict(pb_val_sel)

model_lgb = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
model_lgb.fit(pb_train_sel, y_train)
pred_val_lgb = model_lgb.predict(pb_val_sel)

model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
model_rf.fit(pb_train_sel, y_train)
pred_val_rf = model_rf.predict(pb_val_sel)

model_en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model_en.fit(pb_train_sel, y_train)
pred_val_en = model_en.predict(pb_val_sel)

# --- GROUP B: Geneformer Experts ---
print("--- Training Geneformer Models (MLP, KNN) ---")
model_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), alpha=0.1, activation='relu', early_stopping=True, max_iter=400, random_state=42))
])
model_mlp.fit(gf_train_sel, y_train)
pred_val_mlp = model_mlp.predict(gf_val_sel)

model_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1))
])
model_knn.fit(gf_train_sel, y_train)
pred_val_knn = model_knn.predict(gf_val_sel)

# --- GROUP C: Genotype Experts ---
print("--- Training Genotype Models (Ridge, SVR) ---")
model_ridge = Ridge(alpha=1.0, random_state=42)
model_ridge.fit(pc_train_sel, y_train)
pred_val_ridge = model_ridge.predict(pc_val_sel)

model_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1))
])
model_svr.fit(pc_train_sel, y_train)
pred_val_svr = model_svr.predict(pc_val_sel)

# ==========================================
# 6. Train Meta-Model & Evaluate
# ==========================================
print("\nTraining Meta-Model on 8 base learners...")
X_meta_train = np.column_stack((
    pred_val_xgb, pred_val_lgb, pred_val_rf, pred_val_en,
    pred_val_mlp, pred_val_knn,
    pred_val_ridge, pred_val_svr
))

# Using Ridge with slightly higher alpha to handle the increased collinearity of 8 models
meta_model = Ridge(alpha=10.0, random_state=42)
meta_model.fit(X_meta_train, y_val)

# Generate Predictions
ensemble_val_preds = meta_model.predict(X_meta_train)

# --- CALCULATE METRICS ---
mae = mean_absolute_error(y_val, ensemble_val_preds)
rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
r2 = r2_score(y_val, ensemble_val_preds)
p_corr, _ = pearsonr(y_val, ensemble_val_preds)
s_corr, _ = spearmanr(y_val, ensemble_val_preds)

print(f"=====================================")
print(f"       ENSEMBLE VALIDATION RESULTS   ")
print(f"=====================================")
print(f" MAE:         {mae:.4f}")
print(f" RMSE:        {rmse:.4f}")
print(f" Pearson r:   {p_corr:.4f}")
print(f" Spearman ρ:  {s_corr:.4f}")
print(f" R² Score:    {r2:.4f}")
print(f"=====================================")

# Display contribution weights
labels = ["XGB", "LGBM", "RF", "EN", "MLP", "KNN", "Ridge", "SVR"]
weights = meta_model.coef_
weight_str = "\n".join([f"{l:10} : {w:.4f}" for l, w in zip(labels, weights)])
print(f"Meta-Model Weights:\n{weight_str}")

# ==========================================
# 6.5 Feature Importance & Interpretability
# ==========================================
print("\n=====================================")
print("    TOP FEATURES DRIVING PREDICTIONS   ")
print("=====================================")

try:
    # 1. Recover Original Feature Names for Pseudobulk
    gene_cols = pb_train.columns
    var_mask = pb_selector.named_steps['variance'].get_support()
    genes_after_var = gene_cols[var_mask]
    kbest_mask = pb_selector.named_steps['kbest'].get_support()
    final_genes = genes_after_var[kbest_mask]
    
    pb_feature_names = list(final_genes) + ['sex_binary']
    pc_feature_names = list(pc_train.columns) + ['sex_binary']

    # 2. Extract and Print Importances for Top Pseudobulk Models
    print("\n--- Top 10 Pseudobulk Genes (XGBoost) ---")
    xgb_imp = pd.Series(model_xgb.feature_importances_, index=pb_feature_names)
    print(xgb_imp.sort_values(ascending=False).head(10).to_string(float_format="%.4f"))

    print("\n--- Top 10 Pseudobulk Genes (Random Forest) ---")
    rf_imp = pd.Series(model_rf.feature_importances_, index=pb_feature_names)
    print(rf_imp.sort_values(ascending=False).head(10).to_string(float_format="%.4f"))

    # 3. Extract and Print Importances for Top Genotype Model
    print("\n--- Top 10 Genotype PCs (Ridge) ---")
    ridge_imp = pd.Series(np.abs(model_ridge.coef_), index=pc_feature_names)
    print(ridge_imp.sort_values(ascending=False).head(10).to_string(float_format="%.4f"))

    # 4. Extract ElasticNet Coefficients (showing directionality)
    print("\n--- Top 10 Pseudobulk Genes (ElasticNet - Largest Magnitude) ---")
    en_imp = pd.Series(model_en.coef_, index=pb_feature_names)
    en_top = en_imp.iloc[en_imp.abs().argsort()[::-1]].head(10)
    print(en_top.to_string(float_format="%.4f"))

except Exception as e:
    print(f"\nCould not extract feature names due to an error: {e}")

# ==========================================
# 7. Generate Test Submission
# ==========================================
if test_data_available:
    print("\nGenerating final test predictions...")
    p_xgb = model_xgb.predict(pb_test_sel)
    p_lgb = model_lgb.predict(pb_test_sel)
    p_rf  = model_rf.predict(pb_test_sel)
    p_en  = model_en.predict(pb_test_sel)
    
    p_mlp = model_mlp.predict(gf_test_sel)
    p_knn = model_knn.predict(gf_test_sel)
    
    p_rid = model_ridge.predict(pc_test_sel)
    p_svr = model_svr.predict(pc_test_sel)

    X_meta_test = np.column_stack((p_xgb, p_lgb, p_rf, p_en, p_mlp, p_knn, p_rid, p_svr))
    final_test_preds = meta_model.predict(X_meta_test)

    submission = pd.DataFrame({'donor_id': test_donors, 'age': final_test_preds})
    out_path = RESULTS_DIR / "expanded_ensemble_submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Success! Saved final submission to {out_path}")

# ==========================================
# 8. Save Models for Final Test Release
# ==========================================
MODEL_DIR = PROJ_ROOT / "models" / "saved_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSaving ensemble pipeline to {MODEL_DIR}...")

# Save the feature selector
joblib.dump(pb_selector, MODEL_DIR / 'pb_selector.joblib')

# Save the 8 Base Learners
joblib.dump(model_xgb, MODEL_DIR / 'model_xgb.joblib')
joblib.dump(model_lgb, MODEL_DIR / 'model_lgb.joblib')
joblib.dump(model_rf, MODEL_DIR / 'model_rf.joblib')
joblib.dump(model_en, MODEL_DIR / 'model_en.joblib')
joblib.dump(model_mlp, MODEL_DIR / 'model_mlp.joblib')
joblib.dump(model_knn, MODEL_DIR / 'model_knn.joblib')
joblib.dump(model_ridge, MODEL_DIR / 'model_ridge.joblib')
joblib.dump(model_svr, MODEL_DIR / 'model_svr.joblib')

# Save the Meta-Model
joblib.dump(meta_model, MODEL_DIR / 'meta_model.joblib')

print("All models successfully saved! You are ready for the test data drop.")