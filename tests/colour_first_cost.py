"""Cost of the colour-first rule on validation: hit@5 / hit@10 with and without excluding clearly wrong-colour products."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import tune_lib as T
from explain import ATTRS, COLOR_SIM
LAM, MU, FLOOR, BETA, SLV = 0.2, 0.1, 0.5, 0.5, 0.1
R = np.load(f"{T.ROOT}/data/hub.npy")[: len(T.ids)]; Rc = R - np.median(R[~T.block]); Rc[T.block] = 0
PCS = T.X.probs(T.cat, "color") @ COLOR_SIM; PS = T.X.probs(T.cat, "sleeve")
inv = [v for v in T.allval if v["product"] in T.pos]
res = {False: [], True: []}
for colour_first in (False, True):
    for v in inv:
        q, labels = T.queries(v["scene"]); g = T.pos[v["product"]]
        best = 10**9
        for x, l in zip(q, labels):
            s = x @ T.cat.T; s[T.block] = -1; a = s.copy()
            fb = T.X.bonus_vec(T.P_fam, l); a += LAM * fb if fb is not None else 0
            t = T.X.tags(x); ci = ATTRS["color"].index(t["color"][0])
            if t["color"][1] >= 0.5: a += MU * PCS[:, ci]
            if t["sleeve"][1] >= 0.5: a += SLV * PS[:, ATTRS["sleeve"].index(t["sleeve"][0])]
            a -= BETA * Rc
            if colour_first and t["color"][1] >= 0.6: a = np.where(PCS[:, ci] >= 0.5, a, a - 10)  # wrong colours only after all right ones
            if s[g] >= FLOOR: best = min(best, int((a > a[g]).sum()) + 1)
        res[colour_first].append(best)
for cf in (False, True):
    r = np.array(res[cf]); print("colour-first" if cf else "plain       ", {f"hit@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 100)})
