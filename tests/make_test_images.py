"""Regenerate the reproducible test images in tests/img from the local data (needs data/extended built; see README).
Creates: 12 products x (full, top half, bottom half, left half) named half_<Type>_x<id>_<variant>.jpg, and 3 bottle photos nf_*.jpg.
Not reproducible here: the blur/crop/case_* images, which were cut from screenshots of the app (they are not in the repo)."""
import json, os, random
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EXT = f"{ROOT}/data/extended"; OUT = f"{HERE}/img"; os.makedirs(OUT, exist_ok=True)
meta = [json.loads(l) for l in open(f"{EXT}/meta.jsonl")]
random.seed(11)
for at in ["Tshirts", "Shirts", "Trousers", "Track Pants", "Tracksuits", "Jeans"]:
    cand = [m for m in meta if m["articleType"] == at and m["gender"] in ("Men", "Women")]
    for m in random.sample(cand, 2):
        im = Image.open(f"{EXT}/img/{m['id']}.jpg").convert("RGB"); w, h = im.size
        base = f"{OUT}/half_{at.replace(' ', '')}_{m['id']}"
        im.save(base + "_full.jpg"); im.crop((0, 0, w, h // 2)).save(base + "_tophalf.jpg")
        im.crop((0, h // 2, w, h)).save(base + "_bottomhalf.jpg"); im.crop((0, 0, w // 2, h)).save(base + "_lefthalf.jpg")
bottles = [m for m in meta if m["subCategory"] in ("Fragrance", "Lips", "Nails", "Skin Care")]
for m in random.sample(bottles, 3):
    Image.open(f"{EXT}/img/{m['id']}.jpg").convert("RGB").save(f"{OUT}/nf_{m['subCategory'].replace(' ', '')}_{m['id']}.jpg")
print("done:", len(os.listdir(OUT)), "images in", OUT)
