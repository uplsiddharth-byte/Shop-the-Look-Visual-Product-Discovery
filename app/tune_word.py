"""Word-level bonus: add WB * P(product looks like the detected word, e.g. heels vs sneakers vs sandals) to the family-level ranking."""
import numpy as np
import tune_lib as T
from explain import ATTRS, COLOR_SIM

LAM, MU, FLOOR, BETA, SLV = 0.2, 0.1, 0.5, 0.5, 0.1
R = np.load(f"{T.ROOT}/data/hub.npy")[: len(T.ids)]; Rc = R - np.median(R[~T.block]); Rc[T.block] = 0
PCS = T.X.probs(T.cat, "color") @ COLOR_SIM; PS = T.X.probs(T.cat, "sleeve"); PW = T.X.probs(T.cat, "category")
WORDS = ATTRS["category"]
inv = [v for v in T.allval if v["product"] in T.pos]
for wb in (0, 0.05, 0.1, 0.2, 0.3):
    r = []
    for v in inv:
        q, labels = T.queries(v["scene"]); g = T.pos[v["product"]]; best = 10**9
        for x, l in zip(q, labels):
            s = x @ T.cat.T; s[T.block] = -1; a = s.copy()
            fb = T.X.bonus_vec(T.P_fam, l); a += LAM * fb if fb is not None else 0
            t = T.X.tags(x)
            if t["color"][1] >= 0.5: a += MU * PCS[:, ATTRS["color"].index(t["color"][0])]
            if t["sleeve"][1] >= 0.5: a += SLV * PS[:, ATTRS["sleeve"].index(t["sleeve"][0])]
            if l in WORDS: a += wb * PW[:, WORDS.index(l)]
            a -= BETA * Rc
            if s[g] >= FLOOR: best = min(best, int((a > a[g]).sum()) + 1)
        r.append(best)
    r = np.array(r); print(f"word bonus {wb}", {f"R@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 100)}, flush=True)
