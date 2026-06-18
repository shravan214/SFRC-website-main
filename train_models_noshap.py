"""
train_models_noshap.py
Trains XGBoost, MLP, and KNN. Exports weights as JSON for the web app.
SHAP plot already exists — skipped here to avoid numba dependency.
"""

import json, os, warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 12

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_excel("SFRC dataset 818 data.xlsx")

# Drop any unnamed/extra columns
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
print(f"  Columns: {list(df.columns)}")

# Identify feature and target columns
# Expected: 7 features + 1 target
FEATURES = [c for c in df.columns if 'flexural' not in c.lower()]
TARGET   = [c for c in df.columns if 'flexural'     in c.lower()][0]

print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Target: {TARGET}")

df = df[FEATURES + [TARGET]].dropna()
X  = df[FEATURES].astype(float)
y  = df[TARGET].astype(float)
print(f"  Dataset: {len(df)} rows")

kf      = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {}


# ═══════════════════════════════════════════════════════════════════
#   1. XGBoost
# ═══════════════════════════════════════════════════════════════════
print("\n── XGBoost ──")
xgb_params = dict(
    objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0, early_stopping_rounds=20,
    random_state=42, verbosity=0,
)
xgb_true, xgb_pred = [], []
for tr, te in kf.split(X):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X.iloc[tr], y.iloc[tr],
          eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
    xgb_true += y.iloc[te].tolist()
    xgb_pred += m.predict(X.iloc[te]).tolist()

xgb_true, xgb_pred = np.array(xgb_true), np.array(xgb_pred)
metrics['xgb'] = dict(
    rmse=float(np.sqrt(mean_squared_error(xgb_true, xgb_pred))),
    r2  =float(r2_score(xgb_true, xgb_pred)),
    mae =float(mean_absolute_error(xgb_true, xgb_pred)),
)
print(f"  R²={metrics['xgb']['r2']:.4f}  RMSE={metrics['xgb']['rmse']:.4f}  MAE={metrics['xgb']['mae']:.4f}")

# Train on full dataset and export
xgb_full = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbosity=0,
)
xgb_full.fit(X, y)
xgb_full.save_model("xgb_model.json")
print("  Saved xgb_model.json")

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(xgb_true, xgb_pred, alpha=0.55, color='teal', edgecolor='k', s=30)
lo = min(xgb_true.min(), xgb_pred.min()) - 0.5
hi = max(xgb_true.max(), xgb_pred.max()) + 0.5
ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
ax.set(xlim=[lo,hi], ylim=[lo,hi],
       xlabel="Actual Flexural Strength (MPa)",
       ylabel="Predicted Flexural Strength (MPa)",
       title=f"XGBoost  |  R²={metrics['xgb']['r2']:.3f}  RMSE={metrics['xgb']['rmse']:.3f} MPa")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("XGBoost_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved XGBoost_scatter.png")


# ═══════════════════════════════════════════════════════════════════
#   2. MLP
# ═══════════════════════════════════════════════════════════════════
print("\n── MLP ──")
scaler_mlp   = StandardScaler()
Xs_mlp       = scaler_mlp.fit_transform(X)
mlp_true, mlp_pred = [], []

for tr, te in kf.split(Xs_mlp):
    m = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                     solver='adam', max_iter=1000, random_state=42)
    m.fit(Xs_mlp[tr], y.iloc[tr])
    mlp_true += y.iloc[te].tolist()
    mlp_pred += m.predict(Xs_mlp[te]).tolist()

mlp_true, mlp_pred = np.array(mlp_true), np.array(mlp_pred)
metrics['mlp'] = dict(
    rmse=float(np.sqrt(mean_squared_error(mlp_true, mlp_pred))),
    r2  =float(r2_score(mlp_true, mlp_pred)),
    mae =float(mean_absolute_error(mlp_true, mlp_pred)),
)
print(f"  R²={metrics['mlp']['r2']:.4f}  RMSE={metrics['mlp']['rmse']:.4f}  MAE={metrics['mlp']['mae']:.4f}")

# Full-dataset MLP for JS export
mlp_full = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                        solver='adam', max_iter=2000, random_state=42)
mlp_full.fit(Xs_mlp, y)

mlp_export = {
    "scaler_mean" : scaler_mlp.mean_.tolist(),
    "scaler_scale": scaler_mlp.scale_.tolist(),
    "coefs"       : [c.tolist() for c in mlp_full.coefs_],
    "intercepts"  : [b.tolist() for b in mlp_full.intercepts_],
    "activation"  : mlp_full.activation,
}
with open("mlp_model.json", "w") as f:
    json.dump(mlp_export, f)
print("  Saved mlp_model.json")

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(mlp_true, mlp_pred, alpha=0.55, color='teal', edgecolor='k', s=30)
lo = min(mlp_true.min(), mlp_pred.min()) - 0.5
hi = max(mlp_true.max(), mlp_pred.max()) + 0.5
ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
ax.set(xlim=[lo,hi], ylim=[lo,hi],
       xlabel="Actual Flexural Strength (MPa)",
       ylabel="Predicted Flexural Strength (MPa)",
       title=f"MLP  |  R²={metrics['mlp']['r2']:.3f}  RMSE={metrics['mlp']['rmse']:.3f} MPa")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("MLP_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved MLP_scatter.png")


# ═══════════════════════════════════════════════════════════════════
#   3. KNN
# ═══════════════════════════════════════════════════════════════════
print("\n── KNN ──")
scaler_knn   = StandardScaler()
Xs_knn       = scaler_knn.fit_transform(X)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights'    : ['uniform', 'distance'],
    'metric'     : ['euclidean', 'manhattan'],
}
gs = GridSearchCV(KNeighborsRegressor(), param_grid,
                  cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(Xs_knn, y)
best = gs.best_params_
print(f"  Best params: {best}")

knn_true, knn_pred = [], []
for tr, te in kf.split(Xs_knn):
    m = KNeighborsRegressor(**best)
    m.fit(Xs_knn[tr], y.iloc[tr])
    knn_true += y.iloc[te].tolist()
    knn_pred += m.predict(Xs_knn[te]).tolist()

knn_true, knn_pred = np.array(knn_true), np.array(knn_pred)
metrics['knn'] = dict(
    rmse       =float(np.sqrt(mean_squared_error(knn_true, knn_pred))),
    r2         =float(r2_score(knn_true, knn_pred)),
    mae        =float(mean_absolute_error(knn_true, knn_pred)),
    best_params=best,
)
print(f"  R²={metrics['knn']['r2']:.4f}  RMSE={metrics['knn']['rmse']:.4f}  MAE={metrics['knn']['mae']:.4f}")

knn_export = {
    "scaler_mean" : scaler_knn.mean_.tolist(),
    "scaler_scale": scaler_knn.scale_.tolist(),
    "n_neighbors" : int(best['n_neighbors']),
    "weights"     : best['weights'],
    "metric"      : best['metric'],
    # Export full training set (scaled) for JS kNN
    "X_train"     : Xs_knn.tolist(),
    "y_train"     : y.tolist(),
}
with open("knn_model.json", "w") as f:
    json.dump(knn_export, f)
print("  Saved knn_model.json")

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(knn_true, knn_pred, alpha=0.55, color='teal', edgecolor='k', s=30)
lo = min(knn_true.min(), knn_pred.min()) - 0.5
hi = max(knn_true.max(), knn_pred.max()) + 0.5
ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
ax.set(xlim=[lo,hi], ylim=[lo,hi],
       xlabel="Actual Flexural Strength (MPa)",
       ylabel="Predicted Flexural Strength (MPa)",
       title=f"KNN (k={best['n_neighbors']})  |  R²={metrics['knn']['r2']:.3f}  RMSE={metrics['knn']['rmse']:.3f} MPa")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("KNN_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved KNN_scatter.png")


# ═══════════════════════════════════════════════════════════════════
#   4. Save metrics + dataset info
# ═══════════════════════════════════════════════════════════════════
output = {
    "metrics"  : metrics,
    "features" : FEATURES,
    "target"   : TARGET,
    "n_samples": int(len(df)),
    "feat_min" : X.min().tolist(),
    "feat_max" : X.max().tolist(),
}
with open("model_metrics.json", "w") as f:
    json.dump(output, f, indent=2)
print("\n  Saved model_metrics.json")

print("\n✅  Training complete!\n")
for fn in ["xgb_model.json","mlp_model.json","knn_model.json",
           "model_metrics.json","XGBoost_scatter.png","MLP_scatter.png","KNN_scatter.png"]:
    sz = os.path.getsize(fn) if os.path.exists(fn) else 0
    print(f"   {fn:30s}  {sz:>10,} bytes")
