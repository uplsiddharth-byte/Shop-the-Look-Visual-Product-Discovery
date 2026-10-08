"""Headline evaluation (what the UI shows: top-K per detected item) + exact-match threshold calibration -> data/config.json."""
import json, os, pickle, numpy as np
from PIL import Image
from embed import Embedder, ROOT
from explain import ATTRS, TEMPLATE, FAMILY
import tune_lib as T

LAM, MU, FLOOR, BETA, SLV = 0.2, 0.1, 0.5, 0.5, 0.1  # family bonus, colour bonus, similarity floor (tune_rank.py)
E, ids, pos, cat, dets, allval = T.E, T.ids, T.pos, T.cat, T.dets, T.allval
from explain import ATTRS, MIN_CONF, COLOR_SIM
PC = T.X.probs(cat, "color") @ COLOR_SIM
PS = T.X.probs(cat, "sleeve")
HUB = np.load(f"{T.ROOT}/data/hub.npy")[: len(ids)]
HUB = HUB - np.median(HUB[~T.block]); HUB[T.block] = 0
inv = [v for v in allval if v["product"] in pos]
absent = [v for v in allval if v["product"] not in pos]


def item_scores(v):
    q, labels = T.queries(v["scene"])
    S = q @ cat.T
    S[:, T.block] = -1
    A = S.copy()
    for r, (l, x) in enumerate(zip(labels, q)):
        fb = T.X.bonus_vec(T.P_fam, l)
        if fb is not None:
            A[r] += LAM * fb
        c, conf = T.X.tags(x)["color"]
        if conf >= 0.5:
            A[r] += MU * PC[:, ATTRS["color"].index(c)]
        sv, sc = T.X.tags(x)["sleeve"]
        if sc >= 0.5:
            A[r] += SLV * PS[:, ATTRS["sleeve"].index(sv)]
        if conf >= 0.6:  # shipped colour-first display: clearly wrong-colour products only fill leftover slots, so rank them after all others
            A[r] = np.where(PC[:, ATTRS["color"].index(c)] >= 0.5, A[r], A[r] - 10)
    return A - BETA * HUB, S, labels


# 1) per-item recall: GT product within the top-k list of ANY detected item
ranks = []
fam_hit = []
for v in inv:
    A, S, labels = item_scores(v)
    g = pos[v["product"]]
    # a hit needs the true product to be ranked AND above the similarity floor (below it the UI hides the match)
    ranks.append(min((int((a > a[g]).sum()) + 1 for a, s in zip(A, S) if s[g] >= FLOOR), default=len(ids)))
    gfam = T.P_fam[g].argmax()
    fam_hit.append(any(T.fams[gfam] == T.LABEL_FAM.get(l) for l in labels))
r = np.array(ranks)
out = {"n_pairs_in_catalog": len(inv), "n_pairs_absent": len(absent), "catalog_size": len(ids),
       **{f"recall@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 50, 100)},
       "random_recall@5": round(5 / len(ids), 5), "mrr": round(float((1 / r).mean()), 3),
       "gt_family_detected": round(float(np.mean(fam_hit)), 3)}

# 2) exact-match threshold: top-1 cosine (no bonus) on products known to be absent -> choose 95th percentile
top1_absent = [float(np.where(T.block, -1, T.queries(v["scene"])[0] @ cat.T).max()) for v in absent]
gt_cos = [float(max((T.queries(v["scene"])[0] @ cat[pos[v["product"]]]))) for v in inv]
thr = float(np.percentile(top1_absent, 95))
out.update(exact_threshold=round(thr, 3),
           absent_top1_cos_median=round(float(np.median(top1_absent)), 3),
           gt_cos_median=round(float(np.median(gt_cos)), 3),
           pct_gt_above_thr=round(float(np.mean(np.array(gt_cos) >= thr)), 3),
           false_exact_rate_on_absent=round(float(np.mean(np.array(top1_absent) >= thr)), 3))
cfg_f = f"{ROOT}/data/config.json"
cfg = json.load(open(cfg_f)) if os.path.exists(cfg_f) else {}
cfg.update(family_bonus=LAM, color_bonus=MU, min_sim=FLOOR, hub_beta=BETA, sleeve_bonus=SLV)
cfg["exact_threshold_ext" if T.EXT else "exact_threshold"] = out["exact_threshold"]
json.dump(cfg, open(cfg_f, "w"))
json.dump(out, open(f"{ROOT}/data/{'metrics_ext' if T.EXT else 'metrics'}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
