"""Tune ranking weights on validation: score = cosine + lam*P(family) + mu*P(colour). Also: how many true matches would a similarity floor hide?"""
import itertools, numpy as np
import tune_lib as T
from explain import ATTRS, MIN_CONF

PC = T.X.probs(T.cat, "color"); COLORS = ATTRS["color"]
inv = [v for v in T.allval if v["product"] in T.pos]
items = {}
for v in inv:
    q, labels = T.queries(v["scene"])
    items[v["scene"]] = (q, labels, [T.X.tags(x)["color"] for x in q])


def run(lam, mu):
    r, gt = [], []
    for v in inv:
        q, labels, cols = items[v["scene"]]
        if not len(q):
            r.append(len(T.ids)); gt.append(0); continue
        S = q @ T.cat.T; S[:, T.block] = -1
        g = T.pos[v["product"]]; gt.append(float(S[:, g].max()))
        A = S.copy()
        for i, l in enumerate(labels):
            f = T.LABEL_FAM.get(l)
            if f: A[i] += lam * T.P_fam[:, T.fams.index(f)]
            if cols[i][1] >= MIN_CONF: A[i] += mu * PC[:, COLORS.index(cols[i][0])]
        r.append(min(int((a > a[g]).sum()) + 1 for a in A))
    return np.array(r), np.array(gt)


rows = []
for lam, mu in itertools.product((0.1, 0.2, 0.3), (0, 0.05, 0.1, 0.2)):
    r, gt = run(lam, mu)
    rows.append((float((r <= 10).mean()) + 0.5 * float((r <= 100).mean()), lam, mu))
    print(f"lam={lam} mu={mu}", {f"R@{k}": round(float((r <= k).mean()), 3) for k in (1, 5, 10, 100)}, "MRR", round(float((1 / r).mean()), 3), flush=True)
print("best (R@10 + .5 R@100):", max(rows))
_, gt = run(0.1, 0)
for floor in (0.5, 0.55, 0.6, 0.65):
    print(f"floor {floor}: true matches hidden = {float((gt < floor).mean()):.3f}")
