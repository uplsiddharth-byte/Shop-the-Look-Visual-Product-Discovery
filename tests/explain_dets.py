"""Show every detector candidate for an image: select() decision, then verification numbers (family prob, non-clothing prob)."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from PIL import Image
import numpy as np
import pipeline as pl
from explain import Explainer, FAMS, LABEL_FAM, ATTRS

P = pl.Pipeline("siglip", views=["full"], with_ext=False); X = Explainer(P.E)
for path in sys.argv[1:]:
    img = Image.open(path).convert("RGB"); img.thumbnail((1024, 1024))
    raw = P.detect_raw(img)
    cands = pl.candidates(raw)
    print(f"\n== {os.path.basename(path)}  raw={len(raw)} candidates={len(cands)}")
    q = P.E.images([img.crop(tuple(int(v) for v in d["box"])) for d in cands]) if cands else []
    for d, v in zip(cands, q):
        fp, non, word, _ = X.check(v, worn=d["label"] in pl.WORN)
        own = fp[FAMS.index(LABEL_FAM[d["label"]])]
        print(f"  cand {d['label']:9s} {d['score']:.2f} {[round(x) for x in d['box']]}  own-family={own:.2f} best={FAMS[int(fp.argmax())]}:{fp.max():.2f}({word}) nonclothing={non:.2f}")
    dets, _ = pl.analyze(img, raw, P.E, X)
    print("  FINAL:", [(d["label"], round(d["score"], 2), [round(v) for v in d["box"]]) for d in dets])
