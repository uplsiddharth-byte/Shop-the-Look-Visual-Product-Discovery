"""Tune colour strength. Metrics: validation recall AND colour agreement of the shown top-5 with the photographed item's colour."""
import numpy as np
import tune_lib as T
from explain import ATTRS, MIN_CONF, COLOR_SIM

LAM, FLOOR, BETA, SLV = 0.2, 0.5, 0.5, 0.1
R = np.load(f"{T.ROOT}/data/hub.npy")[: len(T.ids)]; Rc = R - np.median(R[~T.block]); Rc[T.block] = 0
PC = T.X.probs(T.cat, "color"); PS = T.X.probs(T.cat, "sleeve")
prod_color = PC.argmax(1)
inv = [v for v in T.allval if v["product"] in T.pos]
scenes = sorted({v["scene"] for v in T.allval})
tags = {s: [T.X.tags(x) for x in T.queries(s)[0]] for s in scenes}


def adj(s, mu, exact_color):
    q, labels = T.queries(s)
    S = q @ T.cat.T; S[:, T.block] = -1; A = S.copy()
    for i, (l, x) in enumerate(zip(labels, q)):
        fb = T.X.bonus_vec(T.P_fam, l)
        if fb is not None: A[i] += LAM * fb
        t = tags[s][i]
        if t["color"][1] >= 0.5:
            ci = ATTRS["color"].index(t["color"][0])
            A[i] += mu * (PC[:, ci] if exact_color else PC @ COLOR_SIM[ci])
        if t["sleeve"][1] >= 0.5: A[i] += SLV * PS[:, ATTRS["sleeve"].index(t["sleeve"][0])]
    return A - BETA * Rc, S


for exact in (True, False):
    for mu in (0.05, 0.1, 0.2, 0.3, 0.4):
        r = []
        for v in inv:
            q, labels = T.queries(v["scene"])
            if not len(q): r.append(len(T.ids)); continue
            A, S = adj(v["scene"], mu, exact); g = T.pos[v["product"]]
            r.append(min((int((a > a[g]).sum()) + 1 for a, s in zip(A, S) if s[g] >= FLOOR), default=len(T.ids)))
        r = np.array(r)
        agree = []
        for s in scenes:
            q, labels = T.queries(s)
            if not len(q): continue
            A, _ = adj(s, mu, exact)
            for i, t in enumerate(tags[s]):
                if t["color"][1] < 0.5: continue
                ci = ATTRS["color"].index(t["color"][0])
                top = np.argsort(-A[i])[:5]
                agree += [float(COLOR_SIM[ci, prod_color[j]]) >= 0.5 for j in top]
        print(f"{'exact' if exact else 'similar-colours'} mu={mu}", {f"R@{k}": round(float((r <= k).mean()), 3) for k in (5, 10, 100)},
              f"colour agreement of shown top-5: {100*np.mean(agree):.1f}%", flush=True)
