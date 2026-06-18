"""
fix_model_loading.py
--------------------
Transforms the model-loading code in index.html from slow sequential
JSON.parse('...string...') calls into fast parallel loading using
<script type="application/json"> data tags.

Before:
  XGB = JSON.parse('{...}');  // sequential, slow: JS string + JSON parse
  MLP = JSON.parse('{...}');
  KNN = JSON.parse('{...}');

After:
  <script type="application/json" id="d-xgb">{...}</script>  <!-- raw text, no JS-string overhead -->
  ...parsed in parallel via Promise.all + setTimeout(0)...
"""

import re, os, sys

HTML_FILE = "index.html"

with open(HTML_FILE, "r", encoding="utf-8") as f:
    html = f.read()

print(f"Original size: {len(html):,} bytes")

# ── Step 1: Extract the three JSON.parse('...') payloads ─────────────────────
# Each payload looks like:
#   XGB = JSON.parse('{...}');
# The JSON string may contain escaped single-quotes (unlikely but handle it).
# We find the assignment lines and extract the raw JSON payload.

def extract_payload(var_name, source):
    """
    Find `VAR = JSON.parse('...');` and return (raw_json_string, full_match).
    Handles the fact that the JSON payload may be very long.
    """
    pattern = rf"{var_name}\s*=\s*JSON\.parse\('((?:[^'\\]|\\.)*)'\);"
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        return None, None
    # The captured group is the JS-escaped string content.
    # Unescape JS string escapes to get raw JSON.
    raw_json = m.group(1).replace("\\'", "'").replace('\\"', '"')
    return raw_json, m.group(0)

xgb_json, xgb_match = extract_payload("XGB", html)
mlp_json, mlp_match = extract_payload("MLP", html)
knn_json, knn_match = extract_payload("KNN", html)

if not xgb_json:
    print("ERROR: Could not find XGB = JSON.parse(...) in index.html")
    sys.exit(1)
if not mlp_json:
    print("ERROR: Could not find MLP = JSON.parse(...) in index.html")
    sys.exit(1)
if not knn_json:
    print("ERROR: Could not find KNN = JSON.parse(...) in index.html")
    sys.exit(1)

print(f"  XGB payload: {len(xgb_json):,} chars")
print(f"  MLP payload: {len(mlp_json):,} chars")
print(f"  KNN payload: {len(knn_json):,} chars")

# ── Step 2: Build the three <script type="application/json"> tags ─────────────
data_tags = f"""\
<script type="application/json" id="d-xgb">{xgb_json}</script>
<script type="application/json" id="d-mlp">{mlp_json}</script>
<script type="application/json" id="d-knn">{knn_json}</script>"""

# ── Step 3: Build the replacement async load() function ──────────────────────
# Replace the old sequential load() body with a parallel one.
# We keep the badge + tick pattern but parse all three at once.

old_load_block = re.search(
    r'async function load\(\)\s*\{.*?badge\.classList\.add\(\'hidden\'\)',
    html,
    re.DOTALL
)

if not old_load_block:
    print("ERROR: Could not find async function load() block")
    sys.exit(1)

# Find what comes after the models are loaded (MET and banner setup).
# Look for the MET = JSON.parse line and anything after it inside load().
# Capture everything after the three model JSON.parse lines up to the
# badge.classList.add('hidden') line.
after_models_match = re.search(
    r"KNN\s*=\s*JSON\.parse\('[^']*'\);\s*(.*?)badge\.classList\.add\('hidden'\)",
    html,
    re.DOTALL
)

after_models_code = ""
if after_models_match:
    after_models_code = after_models_match.group(1).strip()
    # Indent it nicely
    after_models_code = "\n    ".join(after_models_code.splitlines())

new_load_func = f"""async function load() {{
    badge.innerHTML = '<span class="spin"></span> Loading models&hellip;';

    // Parse all three models in parallel — each gets its own task frame
    // so the browser stays responsive. <script type="application/json"> tags
    // avoid the JS-string-literal overhead of JSON.parse('...string...').
    const parse = id => new Promise(res =>
      setTimeout(() => {{ res(JSON.parse(document.getElementById(id).textContent)); }}, 0)
    );

    [XGB, MLP, KNN] = await Promise.all([
      parse('d-xgb'),
      parse('d-mlp'),
      parse('d-knn'),
    ]);

    {after_models_code}
    badge.classList.add('hidden')"""

# Replace old load() body
html = html[:old_load_block.start()] + new_load_func + html[old_load_block.end():]

# ── Step 4: Remove the three now-stale JSON.parse lines ──────────────────────
# They've been moved to data tags; remove any remaining occurrences.
# (The new load() no longer has them, but just in case.)
for match_str in [xgb_match, mlp_match, knn_match]:
    if match_str and match_str in html:
        html = html.replace(match_str, "", 1)

# ── Step 5: Inject data tags just before </body> ──────────────────────────────
if "</body>" in html:
    html = html.replace("</body>", data_tags + "\n</body>")
    print("  Injected data tags before </body>")
else:
    print("WARNING: </body> not found; appending data tags at end")
    html += "\n" + data_tags

# ── Step 6: Write out ─────────────────────────────────────────────────────────
with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html)

new_size = os.path.getsize(HTML_FILE)
print(f"  Written {HTML_FILE}  ({new_size:,} bytes)")
print("Done! Models will now load in parallel with reduced parse overhead.")
