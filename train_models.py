"""
train_models.py
Trains XGBoost, MLP, and KNN on the SFRC dataset.
Exports trained parameters as JSON for use in the JavaScript web app.
Also generates evaluation plots and the SHAP beeswarm figure.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import shap

# ── Font settings (Times New Roman, 12 pt) ────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 12

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel("SFRC dataset 818 data.xlsx")

# Keep only the 8 relevant columns (drop any extra unnamed cols)
FEATURES = [
    'Cementitious material',
    'Coarse aggregate',
    'Fine aggregate',
    'Water dosage',
    'Superplastizer',
    'Fiber volume',
    'Aspect ratio',
]
TARGET = 'Flexural strength'

df = df[FEATURES + [TARGET]].dropna()
X  = df[FEATURES]
y  = df[TARGET]

print(f"  Dataset shape: {df.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {}

# ══════════════════════════════════════════════════════════════════════════════
#   1.  XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining XGBoost …")

xgb_params = dict(
    objective        = 'reg:squarederror',
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.3,
    reg_lambda       = 1.0,
    early_stopping_rounds = 20,
    random_state     = 42,
    verbosity        = 0,
)

xgb_true, xgb_pred = [], []
xgb_model_final = None

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_true.extend(y_te.tolist())
    xgb_pred.extend(m.predict(X_te).tolist())
    xgb_model_final = m   # keep last fold model for SHAP + export

xgb_true = np.array(xgb_true)
xgb_pred = np.array(xgb_pred)

metrics['xgb'] = dict(
    rmse = float(np.sqrt(mean_squared_error(xgb_true, xgb_pred))),
    r2   = float(r2_score(xgb_true, xgb_pred)),
    mae  = float(mean_absolute_error(xgb_true, xgb_pred)),
)
print(f"  XGBoost → R²={metrics['xgb']['r2']:.4f}  RMSE={metrics['xgb']['rmse']:.4f}  MAE={metrics['xgb']['mae']:.4f}")

# ── Scatter plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(xgb_true, xgb_pred, alpha=0.6, color='teal', edgecolor='k', s=40)
lim = [min(xgb_true.min(), xgb_pred.min()) - 0.5,
       max(xgb_true.max(), xgb_pred.max()) + 0.5]
ax.plot(lim, lim, 'r--', lw=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual Flexural Strength (MPa)")
ax.set_ylabel("Predicted Flexural Strength (MPa)")
ax.set_title(f"XGBoost — R²={metrics['xgb']['r2']:.3f}  RMSE={metrics['xgb']['rmse']:.3f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("XGBoost_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved XGBoost_scatter.png")

# ── Export XGBoost model as JSON for JS traversal ─────────────────────────────
# Re-train on full dataset so JS can do real predictions
print("  Training XGBoost on full dataset for JS export …")
xgb_full = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
xgb_full.fit(X, y)
xgb_full.save_model("xgb_model.json")
print("  Saved xgb_model.json")

# ── SHAP beeswarm ─────────────────────────────────────────────────────────────
print("  Computing SHAP values …")
explainer   = shap.Explainer(xgb_full, X)
shap_values = explainer(X)

plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved shap_plot.png (SHAP beeswarm)")


# ══════════════════════════════════════════════════════════════════════════════
#   2.  MLP
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining MLP …")

scaler_mlp    = StandardScaler()
X_scaled_mlp  = scaler_mlp.fit_transform(X)

mlp_true, mlp_pred = [], []
mlp_model_final = None

for train_idx, test_idx in kf.split(X_scaled_mlp):
    X_tr, X_te = X_scaled_mlp[train_idx], X_scaled_mlp[test_idx]
    y_tr, y_te = y.iloc[train_idx],       y.iloc[test_idx]

    m = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation  = 'relu',
        solver      = 'adam',
        max_iter    = 1000,
        random_state= 42,
    )
    m.fit(X_tr, y_tr)
    mlp_true.extend(y_te.tolist())
    mlp_pred.extend(m.predict(X_te).tolist())
    mlp_model_final = m

mlp_true = np.array(mlp_true)
mlp_pred = np.array(mlp_pred)

metrics['mlp'] = dict(
    rmse = float(np.sqrt(mean_squared_error(mlp_true, mlp_pred))),
    r2   = float(r2_score(mlp_true, mlp_pred)),
    mae  = float(mean_absolute_error(mlp_true, mlp_pred)),
)
print(f"  MLP     → R²={metrics['mlp']['r2']:.4f}  RMSE={metrics['mlp']['rmse']:.4f}  MAE={metrics['mlp']['mae']:.4f}")

# ── Scatter plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(mlp_true, mlp_pred, alpha=0.6, color='teal', edgecolor='k', s=40)
lim = [min(mlp_true.min(), mlp_pred.min()) - 0.5,
       max(mlp_true.max(), mlp_pred.max()) + 0.5]
ax.plot(lim, lim, 'r--', lw=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual Flexural Strength (MPa)")
ax.set_ylabel("Predicted Flexural Strength (MPa)")
ax.set_title(f"MLP — R²={metrics['mlp']['r2']:.3f}  RMSE={metrics['mlp']['rmse']:.3f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("MLP_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved MLP_scatter.png")

# ── Re-train MLP on full dataset and export weights ────────────────────────────
print("  Training MLP on full dataset for JS export …")
mlp_full = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation  = 'relu',
    solver      = 'adam',
    max_iter    = 2000,
    random_state= 42,
)
mlp_full.fit(X_scaled_mlp, y)

mlp_export = {
    "scaler_mean" : scaler_mlp.mean_.tolist(),
    "scaler_scale": scaler_mlp.scale_.tolist(),
    "coefs"       : [c.tolist() for c in mlp_full.coefs_],
    "intercepts"  : [b.tolist() for b in mlp_full.intercepts_],
    "activation"  : mlp_full.activation,
    "n_layers"    : mlp_full.n_layers_,
}
with open("mlp_model.json", "w") as f:
    json.dump(mlp_export, f)
print("  Saved mlp_model.json")


# ══════════════════════════════════════════════════════════════════════════════
#   3.  KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining KNN …")

scaler_knn   = StandardScaler()
X_scaled_knn = scaler_knn.fit_transform(X)

# Grid search for best hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights'    : ['uniform', 'distance'],
    'metric'     : ['euclidean', 'manhattan'],
}
gs = GridSearchCV(
    KNeighborsRegressor(), param_grid,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
gs.fit(X_scaled_knn, y)
best = gs.best_params_
print(f"  Best KNN params: {best}")

knn_true, knn_pred = [], []

for train_idx, test_idx in kf.split(X_scaled_knn):
    X_tr, X_te = X_scaled_knn[train_idx], X_scaled_knn[test_idx]
    y_tr, y_te = y.iloc[train_idx],       y.iloc[test_idx]

    m = KNeighborsRegressor(**best)
    m.fit(X_tr, y_tr)
    knn_true.extend(y_te.tolist())
    knn_pred.extend(m.predict(X_te).tolist())

knn_true = np.array(knn_true)
knn_pred = np.array(knn_pred)

metrics['knn'] = dict(
    rmse        = float(np.sqrt(mean_squared_error(knn_true, knn_pred))),
    r2          = float(r2_score(knn_true, knn_pred)),
    mae         = float(mean_absolute_error(knn_true, knn_pred)),
    best_params = best,
)
print(f"  KNN     → R²={metrics['knn']['r2']:.4f}  RMSE={metrics['knn']['rmse']:.4f}  MAE={metrics['knn']['mae']:.4f}")

# ── Scatter plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(knn_true, knn_pred, alpha=0.6, color='teal', edgecolor='k', s=40)
lim = [min(knn_true.min(), knn_pred.min()) - 0.5,
       max(knn_true.max(), knn_pred.max()) + 0.5]
ax.plot(lim, lim, 'r--', lw=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual Flexural Strength (MPa)")
ax.set_ylabel("Predicted Flexural Strength (MPa)")
ax.set_title(f"KNN — R²={metrics['knn']['r2']:.3f}  RMSE={metrics['knn']['rmse']:.3f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("KNN_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved KNN_scatter.png")

# ── Export KNN scaler + best params for JS ────────────────────────────────────
knn_export = {
    "scaler_mean" : scaler_knn.mean_.tolist(),
    "scaler_scale": scaler_knn.scale_.tolist(),
    "n_neighbors" : int(best['n_neighbors']),
    "weights"     : best['weights'],
    "metric"      : best['metric'],
}
with open("knn_model.json", "w") as f:
    json.dump(knn_export, f)
print("  Saved knn_model.json")


# ══════════════════════════════════════════════════════════════════════════════
#   4.  Save all metrics + dataset stats
# ══════════════════════════════════════════════════════════════════════════════
output = {
    "metrics"  : metrics,
    "features" : FEATURES,
    "target"   : TARGET,
    "n_samples": int(len(df)),
    "feat_min" : X.min().tolist(),
    "feat_max" : X.max().tolist(),
    "feat_mean": X.mean().tolist(),
}
with open("model_metrics.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved model_metrics.json")

print("\n✅  All done! Files written:")
for fn in ["xgb_model.json", "mlp_model.json", "knn_model.json",
           "model_metrics.json", "XGBoost_scatter.png",
           "MLP_scatter.png", "KNN_scatter.png", "shap_plot.png"]:
    import os
    sz = os.path.getsize(fn) if os.path.exists(fn) else 0
    print(f"   {fn:30s}  {sz:>10,} bytes")
