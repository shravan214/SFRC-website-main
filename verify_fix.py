import re

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Check data tags exist
for tag_id in ['d-xgb', 'd-mlp', 'd-knn']:
    found = f'id="{tag_id}"' in html
    print(f'  <script type=application/json id={tag_id}>: {"FOUND" if found else "MISSING"}')

# Check parallel load function
parallel = 'Promise.all' in html
print(f'  Promise.all parallel load: {"FOUND" if parallel else "MISSING"}')

# Verify no leftover JSON.parse string calls for models
leftover_xgb = bool(re.search(r"XGB\s*=\s*JSON\.parse\('", html))
leftover_mlp = bool(re.search(r"MLP\s*=\s*JSON\.parse\('", html))
leftover_knn = bool(re.search(r"KNN\s*=\s*JSON\.parse\('", html))
print(f'  Leftover XGB JSON.parse string: {leftover_xgb}')
print(f'  Leftover MLP JSON.parse string: {leftover_mlp}')
print(f'  Leftover KNN JSON.parse string: {leftover_knn}')

# Show the new load() function
m = re.search(r'async function load\(\).*?badge\.classList\.add', html, re.DOTALL)
if m:
    snippet = m.group(0)[:1000]
    print(f'\nNew load() preview:\n{snippet}')
else:
    print('\nERROR: load() function not found!')

# Validate JSON payloads are actually valid JSON
import json
print('\nValidating JSON payloads...')
for tag_id, label in [('d-xgb', 'XGB'), ('d-mlp', 'MLP'), ('d-knn', 'KNN')]:
    m2 = re.search(rf'<script type="application/json" id="{tag_id}">(.*?)</script>', html, re.DOTALL)
    if not m2:
        print(f'  {label}: tag not found')
        continue
    raw = m2.group(1)
    try:
        obj = json.loads(raw)
        keys = list(obj.keys())
        print(f'  {label}: valid JSON, keys={keys}, size={len(raw):,} chars')
    except Exception as e:
        print(f'  {label}: INVALID JSON - {e}')
