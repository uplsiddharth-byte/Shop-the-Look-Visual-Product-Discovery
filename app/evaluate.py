"""Validation: recall@k of the true product among top-k results pooled over all detected crops of the scene."""
import json, os, sys, pickle, time, numpy as np
from PIL import Image
from pipeline import Pipeline, ROOT
from embed import VIEWS

backend = sys.argv[1] if len(sys.argv) > 1 else "siglip"
val = [json.loads(l) for l in open(f"{ROOT}/dataset/validation.jsonl")]
P = Pipeline(backend, views=VIEWS, with_ext=False)  # ablation needs all views; the shipped pipeline uses only 'full'
pos = {p: i for i, p in enumerate(P.ids)}
val = [v for v in val if v["product"] in pos]  # products absent from catalog can't be recalled; used for threshold calibration
print("pairs with product in catalog:", len(val))

cache = f"{ROOT}/data/dets.pkl"
dets = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
t0 = time.time()
for s in sorted({v["scene"] for v in val} - set(dets)):
    dets[s] = P.detect(Image.open(f"{ROOT}/data/scenes/{s}.jpg").convert("RGB"))
pickle.dump(dets, open(cache, "wb"))
print(f"detection {time.time()-t0:.0f}s, avg {np.mean([len(d) for d in dets.values()]):.1f} boxes/scene")


Q = {}  # scene -> crop embeddings (computed once)
def crops_q(v, mode):
    key = (v["scene"], mode)
    if key not in Q:
        img = Image.open(f"{ROOT}/data/scenes/{v['scene']}.jpg").convert("RGB")
        d = dets[v["scene"]] if mode == "crops" else [dict(box=[0, 0, *img.size])]
        Q[key] = P.E.images([img.crop(tuple(int(x) for x in b["box"])) for b in d])
    return Q[key]


def rank_of(v, mode, views):
    q = crops_q(v, mode)
    best = np.max([(q @ P.embs[w].T).max(0) for w in views], axis=0)  # best crop x best view
    return int((best > best[pos[v["product"]]]).sum()) + 1


for mode in ("full_scene", "crops"):
    for views in (["full"], ["full", "top", "bottom"], list(P.embs)):
        r = np.array([rank_of(v, mode, views) for v in val])
        print(backend, mode, "+".join(views), {f"R@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 50, 100)}, "MRR", round(float((1 / r).mean()), 3), flush=True)
