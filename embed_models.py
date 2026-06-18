"""
embed_models.py
Reads xgb_model.json, mlp_model.json, knn_model.json, model_metrics.json
and dataset.json, then inlines them into index.html as JS variables
so the page works as a standalone file:// without a server.
"""

import json, re, os

def load(fn):
    with open(fn, 'r') as f:
        return json.load(f)

print("Reading model files...")
xgb     = load("xgb_model.json")
mlp     = load("mlp_model.json")
knn     = load("knn_model.json")
metrics = load("model_metrics.json")
dataset = load("dataset.json")

# Compact JSON (no extra whitespace)
def jdump(obj):
    return json.dumps(obj, separators=(',', ':'))

inline_block = f"""<script id="inline-data">
const _XGB     = {jdump(xgb)};
const _MLP     = {jdump(mlp)};
const _KNN     = {jdump(knn)};
const _METRICS = {jdump(metrics)};
const _DB      = {jdump(dataset)};
</script>"""

print(f"  Data block size: {len(inline_block):,} chars")

# Read index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Remove any existing inline-data block
html = re.sub(r'<script id="inline-data">.*?</script>\s*', '',
              html, flags=re.DOTALL)

# Replace fetch-based init with inline-data init
old_init = '''(async function init() {
  await Promise.all([
    loadJSON('dataset.json').then(d => { DB = d; renderTable(); }),
    loadJSON('xgb_model.json').then(d  => { XGB = d;  onModelLoaded('xgb'); }),
    loadJSON('mlp_model.json').then(d  => { MLP = d;  onModelLoaded('mlp'); }),
    loadJSON('knn_model.json').then(d  => { KNN = d;  onModelLoaded('knn'); }),
    loadJSON('model_metrics.json').then(d => { METRICS = d; showMetrics(); }),
  ]).catch(console.warn);
})();'''

new_init = '''(function init() {
  // Data is inlined — no fetch needed, works with file:// protocol
  XGB     = _XGB;
  MLP     = _MLP;
  KNN     = _KNN;
  METRICS = _METRICS;
  DB      = _DB;
  renderTable();
  showMetrics();
  onModelLoaded('xgb');
  onModelLoaded('mlp');
  onModelLoaded('knn');
})();'''

if old_init in html:
    html = html.replace(old_init, new_init)
    print("  Replaced fetch init with inline init")
else:
    print("  WARNING: could not find init block to replace — check manually")

# Inject inline data block just before </body>
html = html.replace('</body>', inline_block + '\n</body>')

out = "index.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

sz = os.path.getsize(out)
print(f"  Written {out}  ({sz:,} bytes)")
print("Done! Open index.html directly in any browser.")
