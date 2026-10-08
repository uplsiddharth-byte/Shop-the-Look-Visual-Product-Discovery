"""Download unique catalog product images + validation scene images (parallel, resumable)."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f"{ROOT}/dataset"


def convert_to_url(s):
    return 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg' % (s[0:2], s[2:4], s[4:6], s)


def get(args):
    sig, folder = args
    path = f"{ROOT}/data/{folder}/{sig}.jpg"
    if os.path.exists(path):
        return sig, True
    for _ in range(3):
        try:
            with urllib.request.urlopen(convert_to_url(sig), timeout=20) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            return sig, True
        except Exception:
            time.sleep(1)
    return sig, False


if __name__ == "__main__":
    prods = sorted({json.loads(l)["product"] for l in open(f"{DS}/product_catelog.jsonl")})
    scenes = sorted({json.loads(l)["scene"] for l in open(f"{DS}/validation.jsonl")})
    jobs = [(s, "scenes") for s in scenes] + [(p, "catalog") for p in prods]
    t0, ok, bad = time.time(), 0, []
    with ThreadPoolExecutor(64) as ex:
        for i, (sig, good) in enumerate(ex.map(get, jobs), 1):
            ok += good
            if not good:
                bad.append(sig)
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} ok={ok} {time.time()-t0:.0f}s", flush=True)
    json.dump(bad, open(f"{ROOT}/data/failed.json", "w"))
    print(f"DONE {ok}/{len(jobs)} ok, {len(bad)} failed, {time.time()-t0:.0f}s total", flush=True)
