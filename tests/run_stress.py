"""Run every image in tests/img through the live API; print detected items + top-1 match. Usage: run_stress.py [name-filter]"""
import os as _os
API = _os.environ.get('API', 'localhost:8000')  # target server host:port
import glob, json, os, subprocess, sys
f = sys.argv[1] if len(sys.argv) > 1 else ""
for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "img", "*.jpg"))):
    n = os.path.basename(p)
    if f not in n: continue
    r = subprocess.run(["curl", "-s", "-F", f"file=@{p}", API + "/api/search?extended=true"], capture_output=True, text=True).stdout
    try: d = json.loads(r)
    except Exception: print(n, "ERROR", r[:100]); continue
    items = [f"{i['label']}({i['confidence']:.2f}|{(i['matches'][0]['name'] or i['matches'][0]['tags'].get('category') or '?')[:22] if i['matches'] else '-'}|{i['matches'][0]['score']:.2f})" for i in d["items"]]
    print(f"{n[:38]:38s} {len(items)} items{(' blurry' if d.get('blurry') else ''):7s} " + "  ".join(items))
