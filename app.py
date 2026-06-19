"""
app.py — Streamlit port of the SFRC Flexural Strength website.

Faithfully reproduces the original HTML/JS site:
  • Lottie background animation (injected into the parent page from an iframe)
  • Exact CSS variables & dark theme
  • Checkbox-driven collapsible sections
  • Red-accent range sliders
  • Three result cards (green / blue / amber)
  • MLP prediction fixed for NumPy 2.x (float(z.flat[0]))
"""

import json
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Models — Predicting Flexural Strength of SFRC",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def asset(name): return os.path.join(BASE_DIR, name)

# ── load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(asset("model_metrics.json")) as f:
        met = json.load(f)

    booster = xgb.Booster()
    booster.load_model(asset("xgb_booster.json"))

    with open(asset("mlp_model.json")) as f:
        md = json.load(f)
    mlp_sm   = np.array(md["scaler_mean"])
    mlp_ss   = np.array(md["scaler_scale"])
    mlp_W    = [np.array(c) for c in md["coefs"]]
    mlp_b    = [np.array(b) for b in md["intercepts"]]

    with open(asset("knn_model.json")) as f:
        kd = json.load(f)
    knn_sm      = np.array(kd["scaler_mean"])
    knn_ss      = np.array(kd["scaler_scale"])
    knn_X       = np.array(kd["X_train_scaled"])
    knn_y       = np.array(kd["y_train"])
    knn_k       = int(kd["n_neighbors"])
    knn_weights = kd["weights"]
    knn_metric  = kd["metric"]

    return (met, booster,
            mlp_sm, mlp_ss, mlp_W, mlp_b,
            knn_sm, knn_ss, knn_X, knn_y, knn_k, knn_weights, knn_metric)

(met, booster,
 mlp_sm, mlp_ss, mlp_W, mlp_b,
 knn_sm, knn_ss, knn_X, knn_y, knn_k, knn_weights, knn_metric) = load_models()

METRICS  = met["metrics"]
FEATURES = met["features"]
FEAT_MIN = met["feat_min"]
FEAT_MAX = met["feat_max"]

# ── prediction helpers ─────────────────────────────────────────────────────────
def relu(x): return np.maximum(0.0, x)

def predict_mlp(x_raw):
    z = (np.asarray(x_raw, dtype=float) - mlp_sm) / mlp_ss
    for i, (W, b) in enumerate(zip(mlp_W, mlp_b)):
        z = z @ W + b
        if i < len(mlp_W) - 1:
            z = relu(z)
    # .flat[0] works for any shape in NumPy 2.x (avoids DeprecationError)
    return float(np.ravel(z)[0])

def predict_knn(x_raw):
    z = (np.asarray(x_raw, dtype=float) - knn_sm) / knn_ss
    if knn_metric == "euclidean":
        dists = np.sqrt(np.sum((knn_X - z) ** 2, axis=1))
    else:
        dists = np.sum(np.abs(knn_X - z), axis=1)
    idx = np.argsort(dists)[:knn_k]
    nd, ny = dists[idx], knn_y[idx]
    if knn_weights == "distance":
        zero = nd == 0
        if zero.any():
            return float(np.mean(ny[zero]))
        w = 1.0 / nd
        return float(np.average(ny, weights=w))
    return float(np.mean(ny))

def predict_xgb(x_raw):
    arr = np.asarray(x_raw, dtype=np.float32).reshape(1, -1)
    dm  = xgb.DMatrix(arr, feature_names=FEATURES)
    return float(booster.predict(dm)[0])

# ── read Lottie JSON (served from same directory) ──────────────────────────────
@st.cache_data
def lottie_json():
    path = asset("VUI Animation.json")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "null"

LOTTIE_DATA = lottie_json()

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — exact replica of the original HTML stylesheet
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

/* ── CSS variables ── */
:root {
  --bg:#0e1117; --sb:#262730; --text:#fafafa; --dim:#8b92a5;
  --muted:#6c7280; --accent:#ff4b4b; --link:#4db8ff;
  --border:#3a3c4a; --card:#1a1c26; --res:#1c1e2e;
  --trk:#464961;
  --green:#21c55d; --blue:#3b82f6; --orange:#f59e0b;
}

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; background: var(--bg) !important; }

html, body, [class*="css"] {
  font-family: 'Source Sans Pro','Segoe UI',sans-serif !important;
  color: var(--text) !important;
}

/* ── app background – transparent so Lottie shows through ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main .block-container {
  background: transparent !important;
}

/* ── Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }

/* ── main content wrapper (mirrors .main in HTML) ── */
.main .block-container {
  position: relative;
  z-index: 1;
  background: rgba(14,17,23,0.62) !important;
  padding: 2.5rem 3.5rem 4rem !important;
  max-width: 900px !important;
}

/* ── sidebar (mirrors .sb in HTML) ── */
[data-testid="stSidebar"] {
  background: rgba(38,39,48,0.82) !important;
  backdrop-filter: blur(4px) !important;
  border-right: 1px solid var(--border) !important;
  z-index: 2 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 1.5rem 1rem 2rem !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── sidebar button (collapse) ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── page title ── */
.page-title {
  font-size: 2rem; font-weight: 700; line-height: 1.3; margin-bottom: 2.5rem;
}

/* ── section / sub headers ── */
.sec-hdr  { font-size: 1.65rem; font-weight: 700; margin-top: 2rem; margin-bottom: 1rem; }
.sub-hdr  { font-size: 1rem;   font-weight: 600; margin: 1.2rem 0 .8rem; }

/* ── metric pills ── */
.m-banner { display:flex; gap:.6rem; flex-wrap:wrap; margin:.5rem 0 .75rem; }
.m-pill   { background:var(--card); border:1px solid var(--border); border-radius:20px;
            padding:.22rem .8rem; font-size:.76rem; color:var(--dim); }
.m-pill span { color:var(--text); font-weight:700; }

/* ── collapsible (checkbox row) ── */
.cb-row { display:flex; align-items:center; gap:.55rem; margin:.5rem 0;
          cursor:pointer; user-select:none; }
.cb-row input[type=checkbox] {
  -webkit-appearance:none; appearance:none;
  width:18px; height:18px; min-width:18px;
  border:2px solid #555; border-radius:3px;
  cursor:pointer; position:relative;
  transition:background .15s,border-color .15s;
}
.cb-row input[type=checkbox]:checked { background:var(--accent); border-color:var(--accent); }
.cb-row input[type=checkbox]:checked::after {
  content:''; position:absolute; top:2px; left:5px;
  width:5px; height:9px; border:2px solid #fff;
  border-top:none; border-left:none; transform:rotate(45deg);
}
.cb-txt { font-size:.85rem; color:var(--dim); }

/* ── collapsible body ── */
.coll { overflow:hidden; max-height:0;
        transition:max-height .45s cubic-bezier(.4,0,.2,1),
                   opacity .35s ease, margin .3s ease;
        opacity:0; }
.coll.open { max-height:6000px; opacity:1; margin-top:.75rem; }
.coll-inner { padding-bottom:.8rem; }

/* ── figure ── */
.fig-wrap { margin:.5rem 0; }
.fig-img  { width:100%; border-radius:6px; display:block; }
.fig-cap  { text-align:center; font-size:.8rem; color:var(--muted); margin-top:.4rem; }

/* ── model textbox ── */
.m-box { background:var(--card); border:1px solid var(--border); border-radius:6px;
         padding:1.1rem 1.3rem; font-size:.84rem; color:var(--dim);
         line-height:1.75; margin:.6rem 0; }
.m-box p { margin-bottom:.7rem; }
.m-box p:last-child { margin-bottom:0; }

/* ── equation row ── */
.eq { display:flex; align-items:center; justify-content:space-between;
      font-family:'Courier New',monospace; font-size:.88rem; color:var(--text);
      background:rgba(255,255,255,.06); border-radius:4px;
      padding:.5rem 1rem; margin:.5rem 0; }
.eqn { color:var(--muted); font-size:.82rem; white-space:nowrap; margin-left:1rem; }

/* ── SHAP wrap ── */
.shap-wrap { background:var(--card); border:1px solid var(--border);
             border-radius:8px; padding:1.2rem; margin:.5rem 0; }

/* ── sliders ── */
.sl-row { margin:.55rem 0; }
.sl-lbl { font-size:.8rem; color:var(--dim); margin-bottom:1px; }
.sl-val { font-size:.82rem; color:var(--accent); margin-bottom:4px; min-height:1.1rem; }

/* Streamlit slider overrides */
[data-testid="stSlider"] > label { font-size:.8rem !important; color:var(--dim) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background:var(--accent) !important;
  border-color:var(--accent) !important;
  box-shadow:0 0 0 3px rgba(255,75,75,.25) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
  box-shadow:0 0 0 5px rgba(255,75,75,.35) !important;
}

/* ── predict button ── */
.pred-btn {
  display:inline-flex; align-items:center; gap:.4rem;
  background:transparent; color:var(--text);
  border:1px solid #555; padding:.38rem 1.1rem;
  border-radius:5px; cursor:pointer;
  font-size:.875rem; font-family:inherit; margin-top:1rem;
  transition:background .2s,border-color .2s,transform .1s;
}
.pred-btn:hover  { background:rgba(255,255,255,.08); border-color:#888; }
.pred-btn:active { transform:scale(.97); }

/* ── result cards ── */
.r-cards { display:flex; gap:1rem; flex-wrap:wrap; margin-top:.85rem; }
.r-card  {
  flex:1; min-width:155px;
  background:var(--res); border:1px solid var(--border);
  border-radius:8px; padding:1rem 1.2rem;
}
.rc-lbl  { font-size:.72rem; color:var(--muted);
           text-transform:uppercase; letter-spacing:.06em; margin-bottom:.25rem; }
.rc-val  { font-size:1.6rem; font-weight:700; }
.rc-unit { font-size:.75rem; color:var(--dim); margin-top:.1rem; }
.rc-cv   { margin-top:.5rem; padding-top:.5rem; border-top:1px solid var(--border);
           font-size:.7rem; color:var(--muted); line-height:1.6; }
.rc-cv b { color:var(--dim); }

.xgb-val { color:var(--green); }
.mlp-val { color:var(--blue);  }
.knn-val { color:var(--orange);}

/* ── loading badge ── */
.load-badge { display:inline-flex; align-items:center; gap:.45rem; font-size:.82rem;
              color:var(--muted); padding:.35rem .8rem;
              background:var(--card); border:1px solid var(--border);
              border-radius:20px; margin-bottom:.75rem; }
.spin { display:inline-block; width:13px; height:13px;
        border:2px solid rgba(255,75,75,.25); border-top-color:var(--accent);
        border-radius:50%; animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }

/* ── author sidebar ── */
.sb-title { font-size:.95rem; font-weight:700; margin-bottom:1rem;
            padding-bottom:.5rem; border-bottom:1px solid var(--border); }
.a-block  { margin-bottom:1.1rem; }
.a-name   { font-size:.82rem; font-weight:700; margin-bottom:.3rem; }
.a-info   { font-size:.75rem; color:var(--dim); line-height:1.55; }
.a-info a { color:var(--link); text-decoration:none; word-break:break-all; }
.a-info a:hover { text-decoration:underline; }

/* ── scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-thumb { background:#444; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#666; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOTTIE BACKGROUND — injected into the parent Streamlit page via iframe JS
# ══════════════════════════════════════════════════════════════════════════════
components.html(f"""
<script>
(function(){{
  var pdoc = window.parent.document;
  if (pdoc.getElementById('sfrc-lottie-bg')) return;   // already injected

  // Create fixed background div behind everything
  var bg = pdoc.createElement('div');
  bg.id = 'sfrc-lottie-bg';
  bg.style.cssText =
    'position:fixed;top:0;left:0;width:100%;height:100%;' +
    'z-index:0;pointer-events:none;overflow:hidden;opacity:0.5;';
  pdoc.body.insertBefore(bg, pdoc.body.firstChild);

  function startAnim(){{
    var animData = {LOTTIE_DATA};
    window.parent.lottie.loadAnimation({{
      container : bg,
      renderer  : 'svg',
      loop      : true,
      autoplay  : true,
      animationData: animData
    }});
  }}

  if (typeof window.parent.lottie !== 'undefined'){{
    startAnim();
  }} else {{
    var s = pdoc.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js';
    s.onload = startAnim;
    pdoc.head.appendChild(s);
  }}
}})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Authors' information
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">Authors\' information</div>',
                unsafe_allow_html=True)
    authors = [
        ("Archana Tanawade",  "archana.tanawade@vit.edu",
         "0000-0001-5923-2242", "https://orcid.org/0000-0001-5923-2242"),
        ("Shravan Wable",     "shravan.22210618@viit.ac.in",
         "0009-0007-8910-3944", "https://orcid.org/0009-0007-8910-3944"),
        ("Srushti Chavhan",   "srushti.22210415@viit.ac.in",
         "0009-0008-8045-7487", "https://orcid.org/0009-0008-8045-7487"),
        ("Isha Jumale",       "isha.22210930@viit.ac.in",
         "0009-0004-6440-3420", "https://orcid.org/0009-0004-6440-3420"),
        ("Kartik Kumbhar",    "kartik.22110138@viit.ac.in",
         "0009-0002-4518-6358", "https://orcid.org/0009-0002-4518-6358"),
    ]
    for name, email, orcid, orcid_url in authors:
        st.markdown(f"""
        <div class="a-block">
          <div class="a-name">{name}</div>
          <div class="a-info">
            Department of Civil Engineering<br>
            Vishwakarma Institute of Technology<br>
            Email: <a href="mailto:{email}">{email}</a><br>
            ORCID: <a href="{orcid_url}" target="_blank">{orcid}</a>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="page-title">Machine learning models-based web application for '
    'predicting flexural strength of Steel Fiber-Reinforced Concrete</h1>',
    unsafe_allow_html=True)

# ─── helper: metric pills ─────────────────────────────────────────────────────
def pills(key):
    m = METRICS.get(key, {})
    r2   = f"{m['r2']:.4f}"   if 'r2'   in m else "—"
    rmse = f"{m['rmse']:.4f}" if 'rmse' in m else "—"
    mae  = f"{m['mae']:.4f}"  if 'mae'  in m else "—"
    st.markdown(f"""
    <div class="m-banner">
      <span class="m-pill">R² <span>{r2}</span></span>
      <span class="m-pill">RMSE <span>{rmse} MPa</span></span>
      <span class="m-pill">MAE <span>{mae} MPa</span></span>
    </div>""", unsafe_allow_html=True)

# ─── helper: show image + caption safely ─────────────────────────────────────
def show_img(filename, caption, fig_id=""):
    p = asset(filename)
    if os.path.exists(p):
        extra = f' id="{fig_id}"' if fig_id else ''
        st.markdown(f'<div class="fig-wrap"{extra}>', unsafe_allow_html=True)
        st.image(p, use_container_width=True)
        st.markdown(f'<div class="fig-cap">{caption}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Machine learning approaches
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h2 class="sec-hdr">1. Machine learning approaches</h2>',
            unsafe_allow_html=True)

# -- 1.1 XGBoost --------------------------------------------------------------
show_xgb = st.checkbox("1.1 Show structure of XGBoost model", key="cb_xgb")
if show_xgb:
    show_img("xgboost_structure.jpg",
             "Overview on structure of XGBoost (Extreme Gradient Boosting) model")
    pills("xgb")

# -- 1.2 MLP ------------------------------------------------------------------
show_mlp = st.checkbox(
    "1.2 Show structure of MLP (Multi-Layer Perceptron) model", key="cb_mlp")
if show_mlp:
    show_img("mlp_diagram.jpg",
             "Overview on structure of Multi-Layer Perceptron (MLP) neural network")
    pills("mlp")

# -- 1.3 KNN ------------------------------------------------------------------
show_knn = st.checkbox(
    "1.3 Show structure of k-Nearest Neighbor (KNN) model", key="cb_knn")
if show_knn:
    show_img("knn_diagram.jpg",
             "Overview on structure of k-Nearest Neighbor (KNN) model")
    pills("knn")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h2 class="sec-hdr">2. SHAP Analysis</h2>', unsafe_allow_html=True)

show_shap = st.checkbox("2.1 Show SHAP feature importance (XGBoost)", key="cb_shap")
if show_shap:
    st.markdown('<div class="shap-wrap">', unsafe_allow_html=True)
    show_img("shap_plot.jpg", "")
    st.markdown("""
    <div class="fig-cap">
      Figure 2. SHAP beeswarm — Impact of each feature on XGBoost model output.<br>
      Red = high feature value &nbsp;|&nbsp; Blue = low feature value &nbsp;|&nbsp;
      X-axis = SHAP value (impact on prediction in MPa)
    </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h2 class="sec-hdr">3. Predicting flexural strength of '
    'Steel Fiber-Reinforced Concrete</h2>',
    unsafe_allow_html=True)

st.markdown('<div class="load-badge" id="load-badge" style="display:none">'
            '<span class="spin"></span> Initialising models…</div>'
            '<div style="display:none" id="load-badge-ready">'
            '<span style="color:var(--green)">●</span> Models ready</div>',
            unsafe_allow_html=True)

st.markdown('<div class="sub-hdr">Input parameters</div>', unsafe_allow_html=True)

# ── slider config ─────────────────────────────────────────────────────────────
SLIDERS = [
    ("Cementitious material", "kg/m³",  1.0,  450.0),
    ("Coarse aggregate",      "kg/m³",  1.0,  860.0),
    ("Fine aggregate",        "kg/m³",  1.0,  650.0),
    ("Water dosage",          "kg/m³",  1.0,  185.0),
    ("Superplastizer",        "%",      0.1,  0.0),
    ("Fiber volume",          "%",      0.01, 0.75),
    ("Aspect ratio",          "",       1.0,  60.0),
]

slider_vals = {}
for i, (feat, unit, step, default) in enumerate(SLIDERS):
    fi = FEATURES.index(feat)
    label = f"{feat} ({unit})" if unit else feat
    val = st.slider(
        label=label,
        min_value=float(FEAT_MIN[fi]),
        max_value=float(FEAT_MAX[fi]),
        value=float(default),
        step=float(step),
        key=f"sl_{feat}",
    )
    slider_vals[feat] = val

# ── predict button + result cards ─────────────────────────────────────────────
st.markdown('<div class="sub-hdr" style="margin-top:1.8rem">Output parameter</div>',
            unsafe_allow_html=True)

predict_clicked = st.button("Predict", key="pred_btn")

if predict_clicked:
    x_raw = [slider_vals[f] for f in FEATURES]
    with st.spinner("Running models…"):
        px = predict_xgb(x_raw)
        pm = predict_mlp(x_raw)
        pk = predict_knn(x_raw)

    mx = METRICS.get("xgb", {})
    mm = METRICS.get("mlp", {})
    mk = METRICS.get("knn", {})
    knn_detail = f"k={mk.get('best_k','—')}, {mk.get('best_weights','')}"

    st.markdown(f"""
    <div class="r-cards">

      <div class="r-card">
        <div class="rc-lbl">XGBoost</div>
        <div class="rc-val xgb-val">{px:.4f}</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
        <div class="rc-cv">
          <b>CV R²</b> {mx.get('r2',0):.4f}<br>
          <b>CV RMSE</b> {mx.get('rmse',0):.4f} MPa<br>
          <b>CV MAE</b> {mx.get('mae',0):.4f} MPa
        </div>
      </div>

      <div class="r-card">
        <div class="rc-lbl">MLP (Neural Network)</div>
        <div class="rc-val mlp-val">{pm:.4f}</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
        <div class="rc-cv">
          <b>CV R²</b> {mm.get('r2',0):.4f}<br>
          <b>CV RMSE</b> {mm.get('rmse',0):.4f} MPa<br>
          <b>CV MAE</b> {mm.get('mae',0):.4f} MPa
        </div>
      </div>

      <div class="r-card">
        <div class="rc-lbl">KNN ({knn_detail})</div>
        <div class="rc-val knn-val">{pk:.4f}</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
        <div class="rc-cv">
          <b>CV R²</b> {mk.get('r2',0):.4f}<br>
          <b>CV RMSE</b> {mk.get('rmse',0):.4f} MPa<br>
          <b>CV MAE</b> {mk.get('mae',0):.4f} MPa
        </div>
      </div>

    </div>""", unsafe_allow_html=True)

else:
    # empty placeholder cards
    st.markdown("""
    <div class="r-cards">
      <div class="r-card">
        <div class="rc-lbl">XGBoost</div>
        <div class="rc-val" style="color:#3a3c4a;">—</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
      </div>
      <div class="r-card">
        <div class="rc-lbl">MLP (Neural Network)</div>
        <div class="rc-val" style="color:#3a3c4a;">—</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
      </div>
      <div class="r-card">
        <div class="rc-lbl">KNN</div>
        <div class="rc-val" style="color:#3a3c4a;">—</div>
        <div class="rc-unit">MPa (Flexural Strength)</div>
      </div>
    </div>""", unsafe_allow_html=True)
