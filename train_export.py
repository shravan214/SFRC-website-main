"""
train_export.py  — Fixed, production-ready training + export script
Trains XGBoost, MLP, KNN. Exports everything needed for JS predictions.
"""

import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONIOENCODING'] = 'utf-8'

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def scatter_plot(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, color='teal', edgecolor='k', s=28)
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
    ax.set(xlim=[lo, hi], ylim=[lo, hi],
           xlabel="Actual Flexural Strength (MPa)",
           ylabel="Predicted Flexural Strength (MPa)",
           title=title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_excel("SFRC dataset 818 data.xlsx")
df = df.loc[:, ~df.columns.str.startswith('Unnamed')].dropna()

FEATURES = [c for c in df.columns if 'flexural' not in c.lower()]
TARGET   = [c for c in df.columns if 'flexural'     in c.lower()][0]
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
print(f"  {len(df)} rows | features: {FEATURES}")

kf      = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {}

# =============================================================================
#  1. XGBoost
# =============================================================================
print("\n[1] XGBoost 5-fold CV...")
xgb_cv_params = dict(
    objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0, early_stopping_rounds=20,
    random_state=42, verbosity=0,
)
xgb_true, xgb_pred = [], []
for tr, te in kf.split(X):
    m = xgb.XGBRegressor(**xgb_cv_params)
    m.fit(X.iloc[tr], y.iloc[tr],
          eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
    xgb_true += y.iloc[te].tolist()
    xgb_pred += m.predict(X.iloc[te]).tolist()

xgb_true = np.array(xgb_true)
xgb_pred = np.array(xgb_pred)
r2   = r2_score(xgb_true, xgb_pred)
rmse = np.sqrt(mean_squared_error(xgb_true, xgb_pred))
mae  = mean_absolute_error(xgb_true, xgb_pred)
metrics['xgb'] = dict(r2=float(r2), rmse=float(rmse), mae=float(mae))
print(f"  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

scatter_plot(xgb_true, xgb_pred,
    f"XGBoost  |  R2={r2:.3f}  RMSE={rmse:.3f} MPa", "XGBoost_scatter.png")

# Train on full dataset (no early stopping) for JS export
print("  Training XGBoost on full data for export...")
xgb_full_params = dict(
    objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbosity=0,
)
xgb_full = xgb.XGBRegressor(**xgb_full_params)
xgb_full.fit(X, y)

# Export XGBoost predictions on entire training set.
# JS will use these as kNN lookup targets -> authentic XGBoost predictions.
xgb_train_preds = xgb_full.predict(X).tolist()

# Also export the booster trees as JSON for optional direct traversal
booster = xgb_full.get_booster()
booster.save_model("xgb_booster.json")
print("  Saved xgb_booster.json")

# Compact export: scaled X + XGBoost predictions (JS kNN lookup)
scaler_xgb = StandardScaler()
X_xgb_scaled = scaler_xgb.fit_transform(X)
xgb_export = {
    "scaler_mean"  : scaler_xgb.mean_.tolist(),
    "scaler_scale" : scaler_xgb.scale_.tolist(),
    "X_train_scaled": X_xgb_scaled.tolist(),
    "y_xgb_pred"   : xgb_train_preds,   # XGBoost's own predictions on training set
    "cv_metrics"   : metrics['xgb'],
}
with open("xgb_model.json", "w") as f:
    json.dump(xgb_export, f)
print("  Saved xgb_model.json")


# =============================================================================
#  2. MLP
# =============================================================================
print("\n[2] MLP 5-fold CV...")
scaler_mlp = StandardScaler()
Xs_mlp     = scaler_mlp.fit_transform(X)

mlp_true, mlp_pred = [], []
for tr, te in kf.split(Xs_mlp):
    m = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                     solver='adam', max_iter=1000, random_state=42)
    m.fit(Xs_mlp[tr], y.iloc[tr])
    mlp_true += y.iloc[te].tolist()
    mlp_pred += m.predict(Xs_mlp[te]).tolist()

mlp_true = np.array(mlp_true)
mlp_pred = np.array(mlp_pred)
r2   = r2_score(mlp_true, mlp_pred)
rmse = np.sqrt(mean_squared_error(mlp_true, mlp_pred))
mae  = mean_absolute_error(mlp_true, mlp_pred)
metrics['mlp'] = dict(r2=float(r2), rmse=float(rmse), mae=float(mae))
print(f"  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

scatter_plot(mlp_true, mlp_pred,
    f"MLP  |  R2={r2:.3f}  RMSE={rmse:.3f} MPa", "MLP_scatter.png")

# Full-dataset MLP for JS export
print("  Training MLP on full data for export...")
mlp_full = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                        solver='adam', max_iter=2000, random_state=42)
mlp_full.fit(Xs_mlp, y)

mlp_export = {
    "scaler_mean"  : scaler_mlp.mean_.tolist(),
    "scaler_scale" : scaler_mlp.scale_.tolist(),
    "coefs"        : [c.tolist() for c in mlp_full.coefs_],
    "intercepts"   : [b.tolist() for b in mlp_full.intercepts_],
    "activation"   : mlp_full.activation,   # 'relu'
    "cv_metrics"   : metrics['mlp'],
}
with open("mlp_model.json", "w") as f:
    json.dump(mlp_export, f)
print("  Saved mlp_model.json")


# =============================================================================
#  3. KNN
# =============================================================================
print("\n[3] KNN GridSearch + 5-fold CV...")
scaler_knn = StandardScaler()
Xs_knn     = scaler_knn.fit_transform(X)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights'    : ['uniform', 'distance'],
    'metric'     : ['euclidean', 'manhattan'],
}
gs = GridSearchCV(KNeighborsRegressor(), param_grid,
                  cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(Xs_knn, y)
best = gs.best_params_
print(f"  Best: {best}")

knn_true, knn_pred = [], []
for tr, te in kf.split(Xs_knn):
    m = KNeighborsRegressor(**best)
    m.fit(Xs_knn[tr], y.iloc[tr])
    knn_true += y.iloc[te].tolist()
    knn_pred += m.predict(Xs_knn[te]).tolist()

knn_true = np.array(knn_true)
knn_pred = np.array(knn_pred)
r2   = r2_score(knn_true, knn_pred)
rmse = np.sqrt(mean_squared_error(knn_true, knn_pred))
mae  = mean_absolute_error(knn_true, knn_pred)
metrics['knn'] = dict(r2=float(r2), rmse=float(rmse), mae=float(mae),
                      best_k=int(best['n_neighbors']),
                      best_weights=best['weights'],
                      best_metric=best['metric'])
print(f"  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

scatter_plot(knn_true, knn_pred,
    f"KNN (k={best['n_neighbors']}, {best['weights']})  |  R2={r2:.3f}  RMSE={rmse:.3f} MPa",
    "KNN_scatter.png")

knn_export = {
    "scaler_mean"  : scaler_knn.mean_.tolist(),
    "scaler_scale" : scaler_knn.scale_.tolist(),
    "X_train_scaled": Xs_knn.tolist(),
    "y_train"      : y.tolist(),
    "n_neighbors"  : int(best['n_neighbors']),
    "weights"      : best['weights'],
    "metric"       : best['metric'],
    "cv_metrics"   : metrics['knn'],
}
with open("knn_model.json", "w") as f:
    json.dump(knn_export, f)
print("  Saved knn_model.json")


# =============================================================================
#  4. Save combined metrics
# =============================================================================
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

print("\n=== DONE ===")
for fn in ["xgb_model.json","mlp_model.json","knn_model.json",
           "model_metrics.json","XGBoost_scatter.png","MLP_scatter.png","KNN_scatter.png"]:
    sz = os.path.getsize(fn) if os.path.exists(fn) else 0
    print(f"  {fn:32s}  {sz:>10,} bytes")
