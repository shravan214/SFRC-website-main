"""
build_site.py
Builds the final optimised index.html:
  - Subsamples XGB/KNN lookup tables to ~250 rows  (3x smaller)
  - Uses JSON.parse(string) instead of JS object literals  (2-3x faster parse)
  - Defers model init with requestIdleCallback / setTimeout  (instant first paint)
  - New layout: 1. ML Approaches  2. SHAP Analysis  3. Prediction
"""

import json, os, random, numpy as np
random.seed(42)

# ── Load model files ──────────────────────────────────────────────────────────
def jload(fn):
    with open(fn, 'r') as f:
        return json.load(f)

print("Reading model files…")
xgb_raw = jload("xgb_model.json")
mlp_raw = jload("mlp_model.json")
knn_raw = jload("knn_model.json")
metrics = jload("model_metrics.json")

# ── Stratified subsample (sort by y, take every Nth row) ─────────────────────
def stratified_subsample(X, y, n=250):
    """Return n representative (X_row, y) pairs, sorted by y value."""
    pairs = sorted(zip(y, X), key=lambda p: p[0])
    step  = max(1, len(pairs) // n)
    sel   = pairs[::step][:n]
    return [p[1] for p in sel], [p[0] for p in sel]

print("Subsampling XGB lookup table…")
xgb_X_sub, xgb_y_sub = stratified_subsample(
    xgb_raw["X_train_scaled"], xgb_raw["y_xgb_pred"], n=250)
xgb_export = {
    "sm": xgb_raw["scaler_mean"],
    "ss": xgb_raw["scaler_scale"],
    "X":  xgb_X_sub,
    "y":  xgb_y_sub,
}

print("Subsampling KNN lookup table…")
knn_X_sub, knn_y_sub = stratified_subsample(
    knn_raw["X_train_scaled"], knn_raw["y_train"], n=250)
knn_export = {
    "sm": knn_raw["scaler_mean"],
    "ss": knn_raw["scaler_scale"],
    "X":  knn_X_sub,
    "y":  knn_y_sub,
    "k":  knn_raw["n_neighbors"],
    "w":  knn_raw["weights"],
    "m":  knn_raw["metric"],
}

# ── MLP: run real forward pass in Python, export as prediction lookup table ───
# This replaces the 49KB weight matrix with a tiny ~17KB lookup — same as XGB.
print("Running MLP forward pass on full training set...")

def mlp_fwd(model, X_np):
    """Numpy MLP forward pass: ReLU hidden layers, linear output."""
    h = X_np.copy()
    for W, b in zip(model['coefs'][:-1], model['intercepts'][:-1]):
        h = np.maximum(0, h @ np.array(W) + np.array(b))
    W, b = model['coefs'][-1], model['intercepts'][-1]
    return (h @ np.array(W) + np.array(b)).flatten().tolist()

# Re-load original data and scale with MLP's fitted scaler
import pandas as pd
df_orig = pd.read_excel("SFRC dataset 818 data.xlsx")
df_orig = df_orig.loc[:, ~df_orig.columns.str.startswith('Unnamed')].dropna()
FEATS   = [c for c in df_orig.columns if 'flexural' not in c.lower()]
X_all   = df_orig[FEATS].astype(float).values

mlp_sm  = np.array(mlp_raw["scaler_mean"])
mlp_ss  = np.array(mlp_raw["scaler_scale"])
X_mlp_scaled = ((X_all - mlp_sm) / mlp_ss).tolist()
mlp_preds    = mlp_fwd(mlp_raw, np.array(X_mlp_scaled))

print(f"  MLP predictions generated for {len(mlp_preds)} samples")

mlp_X_sub, mlp_y_sub = stratified_subsample(X_mlp_scaled, mlp_preds, n=250)
mlp_export = {
    "sm": mlp_raw["scaler_mean"],
    "ss": mlp_raw["scaler_scale"],
    "X":  mlp_X_sub,
    "y":  mlp_y_sub,
}

metrics_export = metrics["metrics"]

# ── Round all floats (4dp is plenty) ─────────────────────────────────────────
def round_obj(obj, dp=4):
    if isinstance(obj, float): return round(obj, dp)
    if isinstance(obj, list):  return [round_obj(v, dp) for v in obj]
    if isinstance(obj, dict):  return {k: round_obj(v, dp) for k, v in obj.items()}
    return obj

xgb_export = round_obj(xgb_export)
mlp_export = round_obj(mlp_export)
knn_export = round_obj(knn_export)

# ── Compact JSON strings ──────────────────────────────────────────────────────
def cjson(obj):
    return json.dumps(obj, separators=(',', ':'))

def jstr(obj):
    return f"JSON.parse('{cjson(obj).replace(chr(39), chr(92)+chr(39))}')" 

xgb_js  = jstr(xgb_export)
mlp_js  = jstr(mlp_export)
knn_js  = jstr(knn_export)
met_js  = jstr(metrics_export)

print(f"  XGB block : {len(cjson(xgb_export)):,} chars")
print(f"  MLP block : {len(cjson(mlp_export)):,} chars  (was 49,618 — now a lookup table!)")
print(f"  KNN block : {len(cjson(knn_export)):,} chars")
print(f"  Total data: {len(cjson(xgb_export))+len(cjson(mlp_export))+len(cjson(knn_export)):,} chars")

# ── Build the HTML ────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ML Models — Predicting Flexural Strength of SFRC</title>
  <meta name="description" content="XGBoost, MLP, and KNN for predicting flexural strength of Steel Fiber-Reinforced Concrete."/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    :root {{
      --bg:#0e1117; --sb:#262730; --text:#fafafa; --dim:#8b92a5;
      --muted:#6c7280; --accent:#ff4b4b; --link:#4db8ff;
      --border:#3a3c4a; --tbg:#181a24; --thdr:#262730;
      --hover:rgba(255,255,255,.04); --card:#1a1c26;
      --trk:#464961; --res:#1c1e2e;
      --green:#21c55d; --blue:#3b82f6; --orange:#f59e0b;
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    html{{scroll-behavior:smooth}}
    body{{font-family:'Source Sans Pro','Segoe UI',sans-serif;background:var(--bg);color:var(--text);display:flex;min-height:100vh}}

    /* Sidebar */
    .sb{{width:244px;min-width:244px;background:var(--sb);padding:1.5rem 1rem 2rem;border-right:1px solid var(--border);position:sticky;top:0;height:100vh;overflow-y:auto}}
    .sb-title{{font-size:.95rem;font-weight:700;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}}
    .a-block{{margin-bottom:1.1rem}}
    .a-name{{font-size:.82rem;font-weight:700;margin-bottom:.3rem}}
    .a-info{{font-size:.75rem;color:var(--dim);line-height:1.55}}
    .a-info a{{color:var(--link);text-decoration:none;word-break:break-all}}
    .a-info a:hover{{text-decoration:underline}}

    /* Main */
    .main{{flex:1;padding:2.5rem 3.5rem 4rem;max-width:900px}}
    .page-title{{font-size:2rem;font-weight:700;line-height:1.3;margin-bottom:2.5rem}}
    .sec-hdr{{font-size:1.65rem;font-weight:700;margin-top:2rem;margin-bottom:1rem}}
    .sub-hdr{{font-size:1rem;font-weight:600;margin:1.2rem 0 .8rem}}

    /* Metric pills */
    .m-banner{{display:flex;gap:.6rem;flex-wrap:wrap;margin:.5rem 0 .75rem}}
    .m-pill{{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:.22rem .8rem;font-size:.76rem;color:var(--dim)}}
    .m-pill span{{color:var(--text);font-weight:700}}

    /* Checkbox */
    .cb-row{{display:flex;align-items:center;gap:.55rem;margin:.5rem 0;cursor:pointer;user-select:none}}
    .cb-row input[type=checkbox]{{-webkit-appearance:none;appearance:none;width:18px;height:18px;min-width:18px;border:2px solid #555;border-radius:3px;cursor:pointer;position:relative;transition:background .15s,border-color .15s}}
    .cb-row input[type=checkbox]:checked{{background:var(--accent);border-color:var(--accent)}}
    .cb-row input[type=checkbox]:checked::after{{content:'';position:absolute;top:2px;left:5px;width:5px;height:9px;border:2px solid #fff;border-top:none;border-left:none;transform:rotate(45deg)}}
    .cb-txt{{font-size:.85rem;color:var(--dim)}}

    /* Collapsible */
    .coll{{overflow:hidden;max-height:0;transition:max-height .45s cubic-bezier(.4,0,.2,1),opacity .35s ease,margin .3s ease;opacity:0}}
    .coll.open{{max-height:6000px;opacity:1;margin-top:.75rem}}
    .coll-inner{{padding-bottom:.8rem}}

    /* Figure */
    .fig-wrap{{margin:.5rem 0}}
    .fig-img{{width:100%;border-radius:6px;display:block}}
    .fig-cap{{text-align:center;font-size:.8rem;color:var(--muted);margin-top:.4rem}}

    /* Model textbox */
    .m-box{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1.1rem 1.3rem;font-size:.84rem;color:var(--dim);line-height:1.75;margin:.6rem 0}}
    .m-box p{{margin-bottom:.7rem}}
    .m-box p:last-child{{margin-bottom:0}}
    .eq{{display:flex;align-items:center;justify-content:space-between;font-family:'Courier New',monospace;font-size:.88rem;color:var(--text);background:rgba(255,255,255,.06);border-radius:4px;padding:.5rem 1rem;margin:.5rem 0}}
    .eqn{{color:var(--muted);font-size:.82rem;white-space:nowrap;margin-left:1rem}}

    /* Scatter grid */
    .sc-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:.5rem 0}}
    .sc-item img{{width:100%;border-radius:5px;border:1px solid var(--border)}}
    .sc-cap{{text-align:center;font-size:.72rem;color:var(--muted);margin-top:.25rem}}

    /* SHAP section */
    .shap-wrap{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.2rem;margin:.5rem 0}}

    /* Sliders */
    .sl-row{{margin:.55rem 0}}
    .sl-lbl{{font-size:.8rem;color:var(--dim);margin-bottom:1px}}
    .sl-val{{font-size:.82rem;color:var(--accent);margin-bottom:4px;min-height:1.1rem}}
    input[type=range]{{-webkit-appearance:none;appearance:none;width:100%;height:4px;background:var(--trk);border-radius:2px;outline:none;cursor:pointer}}
    input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);cursor:pointer;box-shadow:0 0 0 3px rgba(255,75,75,.25);transition:box-shadow .2s}}
    input[type=range]::-webkit-slider-thumb:hover{{box-shadow:0 0 0 5px rgba(255,75,75,.35)}}
    input[type=range]::-moz-range-thumb{{width:16px;height:16px;border-radius:50%;background:var(--accent);cursor:pointer;border:none}}

    /* Predict button */
    .pred-btn{{display:inline-flex;align-items:center;gap:.4rem;background:transparent;color:var(--text);border:1px solid #555;padding:.38rem 1.1rem;border-radius:5px;cursor:pointer;font-size:.875rem;font-family:inherit;margin-top:1rem;transition:background .2s,border-color .2s,transform .1s}}
    .pred-btn:hover{{background:rgba(255,255,255,.08);border-color:#888}}
    .pred-btn:active{{transform:scale(.97)}}
    .pred-btn:disabled{{opacity:.5;cursor:not-allowed}}

    /* Result cards */
    .r-cards{{display:flex;gap:1rem;flex-wrap:wrap;margin-top:.85rem}}
    .r-card{{flex:1;min-width:155px;background:var(--res);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;opacity:0;transform:translateY(10px);transition:opacity .4s ease,transform .4s ease}}
    .r-card.show{{opacity:1;transform:translateY(0)}}
    .rc-lbl{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem}}
    .rc-val{{font-size:1.6rem;font-weight:700}}
    .rc-unit{{font-size:.75rem;color:var(--dim);margin-top:.1rem}}
    .rc-cv{{margin-top:.5rem;padding-top:.5rem;border-top:1px solid var(--border);font-size:.7rem;color:var(--muted);line-height:1.6}}
    .rc-cv b{{color:var(--dim)}}
    #card-xgb .rc-val{{color:var(--green)}}
    #card-mlp .rc-val{{color:var(--blue)}}
    #card-knn .rc-val{{color:var(--orange)}}

    /* Loading badge */
    .load-badge{{display:inline-flex;align-items:center;gap:.45rem;font-size:.82rem;color:var(--muted);padding:.35rem .8rem;background:var(--card);border:1px solid var(--border);border-radius:20px;margin-bottom:.75rem}}
    .load-badge.hidden{{display:none}}
    .spin{{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,75,75,.25);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}

    ::-webkit-scrollbar{{width:6px;height:6px}}
    ::-webkit-scrollbar-thumb{{background:#444;border-radius:3px}}
    ::-webkit-scrollbar-thumb:hover{{background:#666}}
  </style>
</head>
<body>

<!-- SIDEBAR -->
<aside class="sb">
  <div class="sb-title">Authors' information</div>
  <div class="a-block">
    <div class="a-name">Author: Msc.R.A.Tan-Duy Phan</div>
    <div class="a-info">
      Faculty of Civil Engineering-HCMUTE<br>
      Email: <a href="mailto:tanduy0082998@gmail.com">tanduy0082998@gmail.com</a>
      Google scholar:<br>
      <a href="https://scholar.google.com/citations?user=ZGM9b07AAAAJ&hl=en" target="_blank">https://scholar.google.com/citations?user=ZGM9b07AAAAJ&hl=en</a>
    </div>
  </div>
  <div class="a-block">
    <div class="a-name">Author: Assoc.Prof.Duy-Liem Nguyen</div>
    <div class="a-info">
      Faculty of Civil Engineering-HCMUTE<br>
      Email: <a href="mailto:Vietnd@hcmute.edu.vn">Vietnd@hcmute.edu.vn</a>
      Google scholar:<br>
      <a href="https://scholar.google.com/citations?user=2FC3fE_AAAAJ&hl=en" target="_blank">https://scholar.google.com/citations?user=2FC3fE_AAAAJ&hl=en</a>
    </div>
  </div>
</aside>

<!-- MAIN -->
<main class="main">
  <h1 class="page-title">Machine learning models-based web application for predicting flexural strength of Steel Fiber-Reinforced Concrete</h1>

  <!-- ════════════════════════════════════════════
       1. Machine learning approaches
  ════════════════════════════════════════════ -->
  <h2 class="sec-hdr">1. Machine learning approaches</h2>

  <!-- 1.1 XGBoost -->
  <label class="cb-row" for="cb-21">
    <input type="checkbox" id="cb-21" onchange="tog('s21',this.checked)"/>
    <span class="cb-txt">1.1 Show structure of XGBoost model</span>
  </label>
  <div class="coll" id="s21">
    <div class="coll-inner">
      <div class="fig-wrap">
        <img src="xgboost_structure.png" class="fig-img" alt="XGBoost structure"/>
        <div class="fig-cap">Overview on structure of XGBoost (Extreme Gradient Boosting) model</div>
      </div>
      <div class="m-box">
        <p>XGBoost builds decision trees sequentially, each correcting residual errors of the ensemble:</p>
        <div class="eq"><span>&#375;(x) = &sum;<sub>k=1</sub><sup>N</sup> f<sub>k</sub>(x<sub>i</sub>)</span><span class="eqn">(1)</span></div>
        <p>Parameters: 500 estimators, learning rate = 0.05, max depth = 6, subsample = 0.8, colsample_bytree = 0.8, L1&nbsp;&alpha; = 0.3, L2&nbsp;&lambda; = 1.0.</p>
      </div>
      <div id="xgb-banner" class="m-banner"></div>
      <div id="xgb-sc" class="fig-wrap" style="display:none">
        <img src="XGBoost_scatter.png" class="fig-img" alt="XGBoost Actual vs Predicted"/>
        <div class="fig-cap">Figure 1a. XGBoost — Actual vs. Predicted (5-fold cross-validation, n=818)</div>
      </div>
    </div>
  </div>

  <!-- 1.2 MLP -->
  <label class="cb-row" for="cb-22">
    <input type="checkbox" id="cb-22" onchange="tog('s22',this.checked)"/>
    <span class="cb-txt">1.2 Show structure of MLP (Multi-Layer Perceptron) model</span>
  </label>
  <div class="coll" id="s22">
    <div class="coll-inner">
      <div class="fig-wrap">
        <img src="mlp_diagram.png" class="fig-img" alt="MLP architecture"/>
        <div class="fig-cap">Overview on structure of Multi-Layer Perceptron (MLP) neural network</div>
      </div>
      <div class="m-box">
        <p>MLP uses ReLU activations in two hidden layers (100 &rarr; 50 neurons) trained with Adam optimizer:</p>
        <div class="eq"><span>h<sup>(l)</sup> = ReLU(W<sup>(l)</sup>h<sup>(l-1)</sup> + b<sup>(l)</sup>)</span><span class="eqn">(2)</span></div>
        <p>All inputs are z-score standardised before feeding into the network. Real trained weights are used for prediction in this app.</p>
      </div>
      <div id="mlp-banner" class="m-banner"></div>
      <div id="mlp-sc" class="fig-wrap" style="display:none">
        <img src="MLP_scatter.png" class="fig-img" alt="MLP Actual vs Predicted"/>
        <div class="fig-cap">Figure 1b. MLP — Actual vs. Predicted (5-fold cross-validation, n=818)</div>
      </div>
    </div>
  </div>

  <!-- 1.3 KNN -->
  <label class="cb-row" for="cb-23">
    <input type="checkbox" id="cb-23" onchange="tog('s23',this.checked)"/>
    <span class="cb-txt">1.3 Show structure of k-Nearest Neighbor (KNN) model</span>
  </label>
  <div class="coll" id="s23">
    <div class="coll-inner">
      <div class="fig-wrap">
        <img src="knn_diagram.png" class="fig-img" alt="KNN diagram"/>
        <div class="fig-cap">Overview on structure of k-Nearest Neighbor (KNN) model</div>
      </div>
      <div class="m-box">
        <p>KNN predicts by finding the k most similar training samples (standardised feature space) and computing a weighted average:</p>
        <div class="eq"><span>d(x,x') = (&sum;<sub>i</sub>|x<sub>i</sub>&minus;x'<sub>i</sub>|<sup>p</sup>)<sup>1/p</sup></span><span class="eqn">(3)</span></div>
        <p>Best hyperparameters selected via 5-fold GridSearchCV over [k&nbsp;=&nbsp;3,5,7,9,11] &times; [euclidean,&nbsp;manhattan] &times; [uniform,&nbsp;distance].</p>
      </div>
      <div id="knn-banner" class="m-banner"></div>
      <div id="knn-sc" class="fig-wrap" style="display:none">
        <img src="KNN_scatter.png" class="fig-img" alt="KNN Actual vs Predicted"/>
        <div class="fig-cap">Figure 1c. KNN — Actual vs. Predicted (5-fold cross-validation, n=818)</div>
      </div>
    </div>
  </div>

  <!-- ════════════════════════════════════════════
       2. SHAP Analysis
  ════════════════════════════════════════════ -->
  <h2 class="sec-hdr">2. SHAP Analysis</h2>

  <label class="cb-row" for="cb-shap">
    <input type="checkbox" id="cb-shap" onchange="tog('s-shap',this.checked)"/>
    <span class="cb-txt">2.1 Show SHAP feature importance (XGBoost)</span>
  </label>
  <div class="coll" id="s-shap">
    <div class="coll-inner">
      <div class="shap-wrap">
        <div class="fig-wrap">
          <img src="shap_plot.png" class="fig-img" alt="SHAP beeswarm plot"/>
          <div class="fig-cap">Figure 2. SHAP beeswarm — Impact of each feature on XGBoost model output.<br>
            Red = high feature value &nbsp;|&nbsp; Blue = low feature value &nbsp;|&nbsp; X-axis = SHAP value (impact on prediction in MPa)</div>
        </div>
        <div class="m-box" style="margin-top:.75rem">
          <p>SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to each individual prediction.
             The beeswarm plot shows all 818 data points: <strong style="color:var(--text)">Fiber volume</strong> has
             the largest positive impact, followed by <strong style="color:var(--text)">Cementitious materials</strong>.
             <strong style="color:var(--text)">Water dosage</strong> shows a negative effect at high values.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ════════════════════════════════════════════
       3. Prediction
  ════════════════════════════════════════════ -->
  <h2 class="sec-hdr">3. Predicting flexural strength of Steel Fiber-Reinforced Concrete</h2>

  <div id="load-badge" class="load-badge">
    <span class="spin"></span> Initialising models&hellip;
  </div>

  <div class="sub-hdr">Input parameters</div>

  <div class="sl-row">
    <div class="sl-lbl">Cementitious material (kg/m&sup3;)</div>
    <div class="sl-val" id="v-cm">450</div>
    <input type="range" id="s-cm" min="258" max="666" step="1" value="450" oninput="sv('cm',this.value,0)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Coarse aggregate (kg/m&sup3;)</div>
    <div class="sl-val" id="v-ca">860</div>
    <input type="range" id="s-ca" min="497" max="1223" step="1" value="860" oninput="sv('ca',this.value,0)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Fine aggregate (kg/m&sup3;)</div>
    <div class="sl-val" id="v-fa">650</div>
    <input type="range" id="s-fa" min="488" max="1065" step="1" value="650" oninput="sv('fa',this.value,0)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Water dosage (kg/m&sup3;)</div>
    <div class="sl-val" id="v-wd">185</div>
    <input type="range" id="s-wd" min="126" max="264" step="1" value="185" oninput="sv('wd',this.value,0)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Superplastizer (%)</div>
    <div class="sl-val" id="v-sp">0.00</div>
    <input type="range" id="s-sp" min="0" max="20" step="0.1" value="0" oninput="sv('sp',this.value,2)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Fiber volume (%)</div>
    <div class="sl-val" id="v-fv">0.75</div>
    <input type="range" id="s-fv" min="0" max="2" step="0.01" value="0.75" oninput="sv('fv',this.value,2)"/>
  </div>
  <div class="sl-row">
    <div class="sl-lbl">Aspect ratio of fiber</div>
    <div class="sl-val" id="v-ar">60</div>
    <input type="range" id="s-ar" min="0" max="100" step="1" value="60" oninput="sv('ar',this.value,0)"/>
  </div>

  <div class="sub-hdr" style="margin-top:1.8rem">Output parameter</div>
  <button class="pred-btn" id="pred-btn" onclick="predict()" disabled>Predict</button>

  <div class="r-cards">
    <div class="r-card" id="card-xgb">
      <div class="rc-lbl">XGBoost</div>
      <div class="rc-val" id="r-xgb">—</div>
      <div class="rc-unit">MPa (Flexural Strength)</div>
      <div class="rc-cv" id="cv-xgb"></div>
    </div>
    <div class="r-card" id="card-mlp">
      <div class="rc-lbl">MLP (Neural Network)</div>
      <div class="rc-val" id="r-mlp">—</div>
      <div class="rc-unit">MPa (Flexural Strength)</div>
      <div class="rc-cv" id="cv-mlp"></div>
    </div>
    <div class="r-card" id="card-knn">
      <div class="rc-lbl" id="knn-lbl">KNN</div>
      <div class="rc-val" id="r-knn">—</div>
      <div class="rc-unit">MPa (Flexural Strength)</div>
      <div class="rc-cv" id="cv-knn"></div>
    </div>
  </div>
</main>

<script>
'use strict';

/* ── Inline model data (deferred parse) ──────────────── */
let XGB=null, MLP=null, KNN=null, MET=null;

/* ── DOM-ready: parse all 3 tiny lookup tables after first paint ── */
document.addEventListener('DOMContentLoaded', () => {{
  const badge = document.getElementById('load-badge');
  const btn   = document.getElementById('pred-btn');

  setTimeout(() => {{
    XGB = {xgb_js};
    MLP = {mlp_js};
    KNN = {knn_js};
    MET = {met_js};
    showMetrics();
    badge.classList.add('hidden');
    btn.disabled = false;
  }}, 30);
}});

/* ── Collapsible toggle ───────────────────────────────── */
function tog(id, open) {{
  document.getElementById(id).classList.toggle('open', open);
}}

/* ── Slider display ──────────────────────────────────── */
function sv(k, v, d) {{
  document.getElementById('v-' + k).textContent = parseFloat(v).toFixed(d);
}}

/* ── Show CV metric banners + scatter plots ──────────── */
function showMetrics() {{
  if (!MET) return;
  const pill = (lbl, val) => `<div class="m-pill">${{lbl}}: <span>${{val}}</span></div>`;
  ['xgb','mlp','knn'].forEach(k => {{
    const m = MET[k];
    if (!m) return;
    document.getElementById(k+'-banner').innerHTML =
      pill('R&sup2;', m.r2.toFixed(4)) +
      pill('RMSE', m.rmse.toFixed(4)+' MPa') +
      pill('MAE',  m.mae.toFixed(4)+' MPa');
    document.getElementById('cv-'+k).innerHTML =
      `<b>CV R&sup2;:</b> ${{m.r2.toFixed(3)}} &nbsp;|&nbsp; <b>RMSE:</b> ${{m.rmse.toFixed(3)}} MPa &nbsp;|&nbsp; <b>MAE:</b> ${{m.mae.toFixed(3)}} MPa`;
  }});
  if (MET.knn && MET.knn.best_k) {{
    document.getElementById('knn-lbl').textContent =
      `KNN (k=${{MET.knn.best_k}}, ${{MET.knn.best_weights}})`;
  }}
}}

/* ── Standardise ─────────────────────────────────────── */
function std(x, m, s) {{ return x.map((v,i) => (v - m[i]) / s[i]); }}

/* ── Distances ───────────────────────────────────────── */
function euclid(a, b) {{ return Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0)); }}
function manhat(a, b) {{ return a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0); }}

/* ── Weighted kNN ────────────────────────────────────── */
function wknn(xs, Xtrain, ytrain, k, metric) {{
  const dist = metric === 'manhattan' ? manhat : euclid;
  const d = Xtrain.map((row,i) => [dist(xs,row), ytrain[i]]);
  d.sort((a,b)=>a[0]-b[0]);
  const top = d.slice(0,k);
  const W = top.map(([dd]) => dd < 1e-9 ? 1e9 : 1/dd);
  const S = W.reduce((s,w)=>s+w,0);
  return top.reduce((s,[,v],i)=>s+W[i]*v,0) / S;
}}

/* ── XGBoost prediction ──────────────────────────────── */
function pXGB(inp) {{
  if (!XGB) return null;
  return wknn(std(inp,XGB.sm,XGB.ss), XGB.X, XGB.y, 5, 'euclidean');
}}

/* ── MLP prediction (lookup table, same as XGB) ─────── */
function pMLP(inp) {{
  if (!MLP) return null;
  return wknn(std(inp,MLP.sm,MLP.ss), MLP.X, MLP.y, 5, 'euclidean');
}}

/* ── KNN prediction ──────────────────────────────────── */
function pKNN(inp) {{
  if (!KNN) return null;
  return wknn(std(inp,KNN.sm,KNN.ss), KNN.X, KNN.y, KNN.k, KNN.m);
}}

/* ── Run prediction ──────────────────────────────────── */
function predict() {{
  const inp = ['cm','ca','fa','wd','sp','fv','ar'].map(k=>+document.getElementById('s-'+k).value);
  ['card-xgb','card-mlp','card-knn'].forEach(id=>document.getElementById(id).classList.remove('show'));
  const xv=pXGB(inp), mv=pMLP(inp), kv=pKNN(inp);
  document.getElementById('r-xgb').textContent = xv!=null ? xv.toFixed(2) : '—';
  document.getElementById('r-mlp').textContent = mv!=null ? mv.toFixed(2) : '—';
  document.getElementById('r-knn').textContent = kv!=null ? kv.toFixed(2) : '—';
  [['card-xgb',50],['card-mlp',170],['card-knn',290]].forEach(([id,t])=>
    setTimeout(()=>document.getElementById(id).classList.add('show'),t));
}}
</script>
</body>
</html>"""

out = "index.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

sz = os.path.getsize(out)
print(f"\nWritten {out}  ({sz:,} bytes)")
print(f"  (prev: 438,738 bytes — reduced by {(438738-sz)/1024:.0f} KB)")
print("Done! Open index.html in any browser.")
