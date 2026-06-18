"""
fix_met_loading.py
------------------
Moves the remaining MET = JSON.parse('{...}') call to a
<script type="application/json"> tag, consistent with the other models.
"""
import re, os, json

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Find MET = JSON.parse('...');
pattern = r"MET\s*=\s*JSON\.parse\('((?:[^'\\]|\\.)*)'\);"
m = re.search(pattern, html, re.DOTALL)
if not m:
    print("MET JSON.parse not found — nothing to do.")
else:
    raw_json = m.group(1).replace("\\'", "'").replace('\\"', '"')
    # Validate
    try:
        json.loads(raw_json)
        print(f"  MET payload: valid JSON, {len(raw_json):,} chars")
    except Exception as e:
        print(f"  MET payload: INVALID JSON - {e}")

    # Add data tag (insert before the existing data tags)
    met_tag = f'<script type="application/json" id="d-met">{raw_json}</script>\n'
    html = html.replace('<script type="application/json" id="d-xgb">', met_tag + '<script type="application/json" id="d-xgb">')

    # Replace the assignment
    replacement = "MET = JSON.parse(document.getElementById('d-met').textContent);"
    html = html.replace(m.group(0), replacement)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Done — MET now loaded from <script type=application/json id=d-met>")
    print(f"  File size: {os.path.getsize('index.html'):,} bytes")
