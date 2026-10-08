"""Run the raw detector on every validation scene (caches un-filtered boxes; selection/verification are re-tunable) -> data/dets_raw.pkl"""
import json, os, pickle
from PIL import Image
from pipeline import Pipeline, ROOT
P = Pipeline("siglip", views=["full"], with_ext=False)
f = f"{ROOT}/data/dets_raw.pkl"; dets = pickle.load(open(f, "rb")) if os.path.exists(f) else {}
for s in sorted({json.loads(l)["scene"] for l in open(f"{ROOT}/dataset/validation.jsonl")} - set(dets)):
    dets[s] = P.detect_raw(Image.open(f"{ROOT}/data/scenes/{s}.jpg").convert("RGB"))
pickle.dump(dets, open(f, "wb")); print("scenes with raw detections:", len(dets))
