"""(now also tunes the sleeve-length bonus) Tune the hubness penalty: adj = cos + lam*family + mu*colour - beta*(hub - median hub). Report recall AND how concentrated results are."""
import collections, numpy as np
import tune_lib as T
from explain import ATTRS, MIN_CONF

LAM, MU, FLOOR = 0.2, 0.05, 0.5
R = np.load(f"{T.ROOT}/data/hub.npy")[: len(T.ids)]
Rc = R - np.median(R[~T.block])
PC = T.X.probs(T.cat, "color")
PS = T.X.probs(T.cat, "sleeve")
inv = [v for v in T.allval if v["product"] in T.pos]
scenes = sorted({v["scene"] for v in T.allval})


def adj_scores(q, labels, beta, sl=0.0):
    S = q @ T.cat.T; S[:, T.block] = -1
    A = S.copy()
    for i, (l, x) in enumerate(zip(labels, q)):
        fb = T.X.bonus_vec(T.P_fam, l)
        if fb is not None: A[i] += LAM * fb
        c, conf = T.X.tags(x)["color"]
        if conf >= MIN_CONF: A[i] += MU * PC[:, ATTRS["color"].index(c)]
        sv, sc = T.X.tags(x)["sleeve"]
        if sc >= 0.5: A[i] += sl * PS[:, ATTRS["sleeve"].index(sv)]
    return A - beta * Rc, S


for sl in (0, 0.05, 0.1, 0.2, 0.3):
    r = []
    for v in inv:
        q, labels = T.queries(v["scene"])
        if not len(q): r.append(len(T.ids)); continue
        A, S = adj_scores(q, labels, 0.5, sl); g = T.pos[v["product"]]
        r.append(min((int((a > a[g]).sum()) + 1 for a, s in zip(A, S) if s[g] >= FLOOR), default=len(T.ids)))
    r = np.array(r)
    print(f"sleeve={sl}", {f"R@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 100)}, "MRR", round(float((1 / r).mean()), 3), flush=True)
