"""
app.py — Streamlit version of the SFRC Flexural Strength Predictor website.

Converts the single-page HTML/JS site into a Streamlit app.
Models are loaded from the JSON files exported by train_export.py:
  - xgb_booster.json  (XGBoost native booster)
  - mlp_model.json    (MLP weights + scaler)
  - knn_model.json    (KNN training data + scaler + hyper-params)
  - model_metrics.json (R², RMSE, MAE)
Images (scatter plots, architecture diagrams, SHAP plot) are loaded from
the same directory.
"""

import json
import math
import os

import numpy as np
import streamlit as st
from PIL import Image
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Models — Predicting Flexural Strength of SFRC",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS (mirrors the dark theme from the HTML page)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Source Sans Pro', 'Segoe UI', sans-serif;
  }

  /* Dark background */
  .stApp {
    background: #0e1117;
    color: #fafafa;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(38,39,48,0.95);
    border-right: 1px solid #3a3c4a;
  }
  [data-testid="stSidebar"] * {
    color: #fafafa !important;
  }

  /* Metric pill style */
  .metric-pill {
    display: inline-block;
    background: #1a1c26;
    border: 1px solid #3a3c4a;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.82rem;
    color: #8b92a5;
    margin: 3px 4px 3px 0;
  }
  .metric-pill span {
    color: #fafafa;
    font-weight: 700;
  }

  /* Section headers */
  h1 { font-size: 1.9rem !important; font-weight: 700 !important; line-height: 1.3 !important; }
  h2 { font-size: 1.55rem !important; font-weight: 700 !important; color: #fafafa !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; color: #fafafa !important; }

  /* Result card */
  .result-card {
    background: #1c1e2e;
    border: 1px solid #3a3c4a;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .result-card .label {
    font-size: 0.72rem;
    color: #6c7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }
  .result-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 2px;
  }
  .result-card .unit {
    font-size: 0.75rem;
    color: #8b92a5;
  }
  .result-card .cv-info {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #3a3c4a;
    font-size: 0.7rem;
    color: #6c7280;
    line-height: 1.6;
  }
  .result-card .cv-info b { color: #8b92a5; }

  .val-xgb { color: #21c55d; }
  .val-mlp { color: #3b82f6; }
  .val-knn { color: #f59e0b; }

  /* Author info */
  .author-name { font-size: 0.85rem; font-weight: 700; margin-bottom: 2px; }
  .author-info { font-size: 0.78rem; color: #8b92a5; line-height: 1.55; }
  .author-info a { color: #4db8ff; text-decoration: none; }
  .author-info a:hover { text-decoration: underline; }
  .author-block { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #3a3c4a; }
  .author-block:last-child { border-bottom: none; }

  /* Model textbox */
  .model-box {
    background: #1a1c26;
    border: 1px solid #3a3c4a;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 0.84rem;
    color: #8b92a5;
    line-height: 1.75;
    margin: 8px 0;
  }
  .model-box p { margin-bottom: 10px; }
  .model-box p:last-child { margin-bottom: 0; }

  /* Equation row */
  .eq-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'Courier New', monospace;
    font-size: 0.88rem;
    color: #fafafa;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 8px 16px;
    margin: 6px 0;
  }
  .eq-num { color: #6c7280; font-size: 0.82rem; margin-left: 12px; white-space: nowrap; }

  /* Fig caption */
  .fig-cap {
    text-align: center;
    font-size: 0.8rem;
    color: #6c7280;
    margin-top: 4px;
  }

  /* Sidebar title */
  .sb-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #3a3c4a;
  }

  /* Hide default Streamlit chrome elements */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Streamlit slider style overrides */
  [data-testid="stSlider"] > div > div > div { background: #ff4b4b !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(filename):
    return os.path.join(BASE_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Load model data (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load and return all model artefacts."""
    # --- Metrics ---
    with open(asset("model_metrics.json")) as f:
        met = json.load(f)

    # --- XGBoost (native booster) ---
    booster = xgb.Booster()
    booster.load_model(asset("xgb_booster.json"))

    # --- MLP weights ---
    with open(asset("mlp_model.json")) as f:
        mlp_data = json.load(f)
    mlp_sm = np.array(mlp_data["scaler_mean"])
    mlp_ss = np.array(mlp_data["scaler_scale"])
    mlp_coefs = [np.array(c) for c in mlp_data["coefs"]]
    mlp_intercepts = [np.array(b) for b in mlp_data["intercepts"]]

    # --- KNN training data ---
    with open(asset("knn_model.json")) as f:
        knn_data = json.load(f)
    knn_sm = np.array(knn_data["scaler_mean"])
    knn_ss = np.array(knn_data["scaler_scale"])
    knn_X = np.array(knn_data["X_train_scaled"])
    knn_y = np.array(knn_data["y_train"])
    knn_k = int(knn_data["n_neighbors"])
    knn_weights = knn_data["weights"]
    knn_metric = knn_data["metric"]

    return met, booster, mlp_sm, mlp_ss, mlp_coefs, mlp_intercepts, \
           knn_sm, knn_ss, knn_X, knn_y, knn_k, knn_weights, knn_metric


met, booster, mlp_sm, mlp_ss, mlp_coefs, mlp_intercepts, \
    knn_sm, knn_ss, knn_X, knn_y, knn_k, knn_weights, knn_metric = load_models()

METRICS = met["metrics"]
FEATURES = met["features"]
FEAT_MIN = met["feat_min"]
FEAT_MAX = met["feat_max"]

# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def predict_mlp(x_raw):
    """Forward pass through MLP (relu hidden layers, linear output)."""
    z = (np.array(x_raw) - mlp_sm) / mlp_ss
    for i, (W, b) in enumerate(zip(mlp_coefs, mlp_intercepts)):
        z = z @ W + b
        if i < len(mlp_coefs) - 1:
            z = relu(z)
    return float(z)


def predict_knn(x_raw, k, weights, metric):
    """KNN prediction using stored training data."""
    z = (np.array(x_raw) - knn_sm) / knn_ss

    if metric == "euclidean":
        dists = np.sqrt(np.sum((knn_X - z) ** 2, axis=1))
    else:  # manhattan
        dists = np.sum(np.abs(knn_X - z), axis=1)

    idx = np.argsort(dists)[:k]
    nn_dists = dists[idx]
    nn_y = knn_y[idx]

    if weights == "distance":
        # Avoid division by zero for exact matches
        if np.any(nn_dists == 0):
            return float(np.mean(nn_y[nn_dists == 0]))
        w = 1.0 / nn_dists
        return float(np.average(nn_y, weights=w))
    else:
        return float(np.mean(nn_y))


def predict_xgb(x_raw):
    """XGBoost prediction using native booster."""
    arr = np.array(x_raw, dtype=np.float32).reshape(1, -1)
    dm = xgb.DMatrix(arr, feature_names=FEATURES)
    return float(booster.predict(dm)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Author information
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Authors\' information</div>', unsafe_allow_html=True)

    authors = [
        {
            "name": "Archana Tanawade",
            "dept": "Department of Civil Engineering",
            "inst": "Vishwakarma Institute of Technology",
            "email": "archana.tanawade@vit.edu",
            "orcid": "0000-0001-5923-2242",
            "orcid_url": "https://orcid.org/0000-0001-5923-2242",
        },
        {
            "name": "Shravan Wable",
            "dept": "Department of Civil Engineering",
            "inst": "Vishwakarma Institute of Technology",
            "email": "shravan.22210618@viit.ac.in",
            "orcid": "0009-0007-8910-3944",
            "orcid_url": "https://orcid.org/0009-0007-8910-3944",
        },
        {
            "name": "Srushti Chavhan",
            "dept": "Department of Civil Engineering",
            "inst": "Vishwakarma Institute of Technology",
            "email": "srushti.22210415@viit.ac.in",
            "orcid": "0009-0008-8045-7487",
            "orcid_url": "https://orcid.org/0009-0008-8045-7487",
        },
        {
            "name": "Isha Jumale",
            "dept": "Department of Civil Engineering",
            "inst": "Vishwakarma Institute of Technology",
            "email": "isha.22210930@viit.ac.in",
            "orcid": "0009-0004-6440-3420",
            "orcid_url": "https://orcid.org/0009-0004-6440-3420",
        },
        {
            "name": "Kartik Kumbhar",
            "dept": "Department of Civil Engineering",
            "inst": "Vishwakarma Institute of Technology",
            "email": "kartik.22110138@viit.ac.in",
            "orcid": "0009-0002-4518-6358",
            "orcid_url": "https://orcid.org/0009-0002-4518-6358",
        },
    ]

    for a in authors:
        st.markdown(f"""
        <div class="author-block">
          <div class="author-name">{a['name']}</div>
          <div class="author-info">
            {a['dept']}<br>
            {a['inst']}<br>
            Email: <a href="mailto:{a['email']}">{a['email']}</a><br>
            ORCID: <a href="{a['orcid_url']}" target="_blank">{a['orcid']}</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("Machine learning models-based web application for predicting flexural strength of Steel Fiber-Reinforced Concrete")

# ─── Helper: metric pills ────────────────────────────────────────────────────
def metric_pills(model_key, label):
    m = METRICS.get(model_key, {})
    r2_v   = m.get("r2",   "—")
    rmse_v = m.get("rmse", "—")
    mae_v  = m.get("mae",  "—")
    r2_s   = f"{r2_v:.4f}"   if isinstance(r2_v, float) else str(r2_v)
    rmse_s = f"{rmse_v:.4f}" if isinstance(rmse_v, float) else str(rmse_v)
    mae_s  = f"{mae_v:.4f}"  if isinstance(mae_v, float) else str(mae_v)
    st.markdown(f"""
    <div style="margin:6px 0 10px;">
      <span class="metric-pill">R² <span>{r2_s}</span></span>
      <span class="metric-pill">RMSE <span>{rmse_s} MPa</span></span>
      <span class="metric-pill">MAE <span>{mae_s} MPa</span></span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: ML Approaches
# ─────────────────────────────────────────────────────────────────────────────
st.header("1. Machine learning approaches")

# ── 1.1 XGBoost ──
with st.expander("1.1  Show structure of XGBoost model"):
    img_path = asset("xgboost_structure.jpg")
    if os.path.exists(img_path):
        st.image(img_path, caption="Overview on structure of XGBoost (Extreme Gradient Boosting) model", use_container_width=True)
    metric_pills("xgb", "XGBoost")
    sc_path = asset("XGBoost_scatter.png")
    if os.path.exists(sc_path):
        st.image(sc_path, caption="Figure 1a. XGBoost — Actual vs. Predicted (5-fold cross-validation, n=818)", use_container_width=True)

    st.markdown("""
    <div class="model-box">
      <p><b>XGBoost (Extreme Gradient Boosting)</b> is an ensemble tree-based method that sequentially builds
      decision trees, each correcting the residual errors of the previous ones. It combines gradient
      boosting with L1/L2 regularisation to prevent overfitting.</p>
      <p>The final prediction is a sum of <em>T</em> tree outputs:</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="eq-row">
      ŷ = Σ<sub>t=1</sub><sup>T</sup> f<sub>t</sub>(x)
      <span class="eq-num">(1)</span>
    </div>
    """, unsafe_allow_html=True)

# ── 1.2 MLP ──
with st.expander("1.2  Show structure of MLP (Multi-Layer Perceptron) model"):
    img_path = asset("mlp_diagram.jpg")
    if os.path.exists(img_path):
        st.image(img_path, caption="Overview on structure of Multi-Layer Perceptron (MLP) neural network", use_container_width=True)
    metric_pills("mlp", "MLP")
    sc_path = asset("MLP_scatter.png")
    if os.path.exists(sc_path):
        st.image(sc_path, caption="Figure 1b. MLP — Actual vs. Predicted (5-fold cross-validation, n=818)", use_container_width=True)

    m = METRICS.get("mlp", {})
    st.markdown(f"""
    <div class="model-box">
      <p><b>Multi-Layer Perceptron (MLP)</b> is a feedforward neural network with two hidden layers
      (100 and 50 neurons, ReLU activation) trained with the Adam optimiser over 1 000 epochs.</p>
      <p>Each neuron applies: <em>z = ReLU(Wx + b)</em></p>
    </div>
    """, unsafe_allow_html=True)

# ── 1.3 KNN ──
with st.expander("1.3  Show structure of k-Nearest Neighbor (KNN) model"):
    img_path = asset("knn_diagram.jpg")
    if os.path.exists(img_path):
        st.image(img_path, caption="Overview on structure of k-Nearest Neighbor (KNN) model", use_container_width=True)
    metric_pills("knn", "KNN")
    sc_path = asset("KNN_scatter.png")
    if os.path.exists(sc_path):
        st.image(sc_path, caption="Figure 1c. KNN — Actual vs. Predicted (5-fold cross-validation, n=818)", use_container_width=True)

    km = METRICS.get("knn", {})
    best_k = km.get("best_k", "—")
    best_w = km.get("best_weights", "—")
    best_m = km.get("best_metric", "—")
    st.markdown(f"""
    <div class="model-box">
      <p><b>k-Nearest Neighbors (KNN)</b> predicts by averaging the targets of the <em>k</em> closest
      training samples in feature space. Optimal hyper-parameters were selected via 5-fold grid search.</p>
      <p>Best configuration: <b>k = {best_k}</b>, weights = <b>{best_w}</b>, metric = <b>{best_m}</b></p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.header("2. SHAP Analysis")

with st.expander("2.1  Show SHAP feature importance (XGBoost)"):
    shap_path = asset("shap_plot.jpg")
    if os.path.exists(shap_path):
        st.markdown("""
        <div style="background:#1a1c26;border:1px solid #3a3c4a;border-radius:8px;padding:16px;margin:6px 0;">
        """, unsafe_allow_html=True)
        st.image(shap_path, use_container_width=True)
        st.markdown("""
        <div class="fig-cap">
          Figure 2. SHAP beeswarm — Impact of each feature on XGBoost model output.<br>
          Red = high feature value &nbsp;|&nbsp; Blue = low feature value &nbsp;|&nbsp; X-axis = SHAP value (impact on prediction in MPa)
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("shap_plot.jpg not found in the app directory.")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Prediction
# ─────────────────────────────────────────────────────────────────────────────
st.header("3. Predicting flexural strength of Steel Fiber-Reinforced Concrete")

st.subheader("Input parameters")

feat_defaults = {
    "Cementitious material": 450,
    "Coarse aggregate":      860,
    "Fine aggregate":        650,
    "Water dosage":          185,
    "Superplastizer":        0.0,
    "Fiber volume":          0.75,
    "Aspect ratio":          60,
}

feat_steps = {
    "Cementitious material": 1,
    "Coarse aggregate":      1,
    "Fine aggregate":        1,
    "Water dosage":          1,
    "Superplastizer":        0.1,
    "Fiber volume":          0.01,
    "Aspect ratio":          1,
}

feat_units = {
    "Cementitious material": "kg/m³",
    "Coarse aggregate":      "kg/m³",
    "Fine aggregate":        "kg/m³",
    "Water dosage":          "kg/m³",
    "Superplastizer":        "%",
    "Fiber volume":          "%",
    "Aspect ratio":          "(dimensionless)",
}

slider_values = {}
col1, col2 = st.columns(2)

for i, feat in enumerate(FEATURES):
    col = col1 if i % 2 == 0 else col2
    with col:
        val = col.slider(
            label=f"{feat}  ({feat_units[feat]})",
            min_value=float(FEAT_MIN[i]),
            max_value=float(FEAT_MAX[i]),
            value=float(feat_defaults.get(feat, (FEAT_MIN[i] + FEAT_MAX[i]) / 2)),
            step=float(feat_steps.get(feat, 1.0)),
            key=f"slider_{feat}",
        )
        slider_values[feat] = val

# ─── Predict button ───────────────────────────────────────────────────────────
st.subheader("Output parameter")

if st.button("⚡  Predict Flexural Strength", type="primary", use_container_width=False):
    x_raw = [slider_values[f] for f in FEATURES]

    with st.spinner("Running models…"):
        pred_xgb = predict_xgb(x_raw)
        pred_mlp = predict_mlp(x_raw)
        pred_knn = predict_knn(x_raw, knn_k, knn_weights, knn_metric)

    # Retrieve CV metrics
    mx = METRICS.get("xgb", {})
    mm = METRICS.get("mlp", {})
    mk = METRICS.get("knn", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="result-card">
          <div class="label">XGBoost</div>
          <div class="value val-xgb">{pred_xgb:.2f}</div>
          <div class="unit">MPa (Flexural Strength)</div>
          <div class="cv-info">
            <b>CV R²</b> {mx.get('r2', '—'):.4f}<br>
            <b>CV RMSE</b> {mx.get('rmse', '—'):.4f} MPa<br>
            <b>CV MAE</b> {mx.get('mae', '—'):.4f} MPa
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="result-card">
          <div class="label">MLP (Neural Network)</div>
          <div class="value val-mlp">{pred_mlp:.2f}</div>
          <div class="unit">MPa (Flexural Strength)</div>
          <div class="cv-info">
            <b>CV R²</b> {mm.get('r2', '—'):.4f}<br>
            <b>CV RMSE</b> {mm.get('rmse', '—'):.4f} MPa<br>
            <b>CV MAE</b> {mm.get('mae', '—'):.4f} MPa
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        knn_lbl_detail = ""
        if "best_k" in mk:
            knn_lbl_detail = f"k={mk['best_k']}, {mk.get('best_weights','')}"
        st.markdown(f"""
        <div class="result-card">
          <div class="label">KNN ({knn_lbl_detail})</div>
          <div class="value val-knn">{pred_knn:.2f}</div>
          <div class="unit">MPa (Flexural Strength)</div>
          <div class="cv-info">
            <b>CV R²</b> {mk.get('r2', '—'):.4f}<br>
            <b>CV RMSE</b> {mk.get('rmse', '—'):.4f} MPa<br>
            <b>CV MAE</b> {mk.get('mae', '—'):.4f} MPa
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.balloons()

else:
    # Empty placeholders
    c1, c2, c3 = st.columns(3)
    for col, label, cls in zip(
        [c1, c2, c3],
        ["XGBoost", "MLP (Neural Network)", "KNN"],
        ["val-xgb", "val-mlp", "val-knn"],
    ):
        col.markdown(f"""
        <div class="result-card">
          <div class="label">{label}</div>
          <div class="value {cls}" style="color:#3a3c4a;">—</div>
          <div class="unit">MPa (Flexural Strength)</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#464961;font-size:0.8rem;'>"
    "ML Models — Predicting Flexural Strength of Steel Fiber-Reinforced Concrete &nbsp;·&nbsp; "
    "Vishwakarma Institute of Technology, Department of Civil Engineering"
    "</div>",
    unsafe_allow_html=True,
)
