"""Shared evaluation state: SigLIP embedder, catalog index, cached detections, zero-shot garment-family probabilities."""
import json, pickle, numpy as np
from PIL import Image
from embed import Embedder, ROOT
from explain import ATTRS, TEMPLATE, FAMILY, LABEL_FAM, Explainer
from pipeline import analyze

E = Embedder("siglip")
ids = open(f"{ROOT}/data/ids.txt").read().split("\n"); pos = {p: i for i, p in enumerate(ids)}
import os
EXT = os.environ.get("EXT") == "1"  # EXT=1: add the extended catalog to the index as extra distractors/candidates
cat = np.load(f"{ROOT}/data/emb_siglip_full.npy")
N_BASE = len(cat)
if EXT:
    cat = np.concatenate([cat, np.load(f"{ROOT}/data/emb_siglip_ext.npy")])
    ids = ids + open(f"{ROOT}/data/ids_ext.txt").read().split("\n")
dets = pickle.load(open(f"{ROOT}/data/dets_raw.pkl", "rb"))
import os
_b = set(json.load(open(f"{ROOT}/data/blocked.json"))) if os.path.exists(f"{ROOT}/data/blocked.json") else set()
block = np.array([i in _b for i in ids])  # ids already include extended ids when EXT=1
allval = [json.loads(l) for l in open(f"{ROOT}/dataset/validation.jsonl")]

words = ATTRS["category"]
_T = E.texts([TEMPLATE["category"].format(w) for w in words])
fams = sorted(set(FAMILY.values())); fam_of = np.array([fams.index(FAMILY[w]) for w in words])
_s = cat @ _T.T * 100; _p = np.exp(_s - _s.max(1, keepdims=True)); _p /= _p.sum(1, keepdims=True)
P_fam = np.stack([_p[:, fam_of == f].sum(1) for f in range(len(fams))], 1)  # (N, n_fams) garment-family probability per product

X = Explainer(E)
if EXT:  # same metadata override as the server
    from families import ext_family_matrix
    _M = ext_family_matrix(ids[N_BASE:]); _k = ~np.isnan(_M[:, 0])
    P_fam[N_BASE:][_k] = _M[_k]
_Q = {}
def queries(scene):
    """verified items for a validation scene (same select + verify logic as the server) -> (crop embeddings, labels)."""
    if scene not in _Q:
        img = Image.open(f"{ROOT}/data/scenes/{scene}.jpg").convert("RGB")
        d, q = analyze(img, dets[scene], E, X)
        _Q[scene] = (q, [b["label"] for b in d])
    return _Q[scene]
