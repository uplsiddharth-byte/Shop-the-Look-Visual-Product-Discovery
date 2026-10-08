"""data/blocked.json = ids never recommended: blank/placeholder images (both catalogs) + non-fashion extended products."""
import json, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NON_FASHION = {"Personal Care", "Free Items", "Sporting Goods", "Home"}
SKIP_SUB = {"Innerwear", "Socks", "Loungewear and Nightwear", "Free Gifts", "Fragrance", "Lips", "Nails", "Makeup", "Skin", "Skin Care",
            "Eyes", "Hair", "Bath and Body", "Perfumes and Deodorants", "Water Bottle", "Mask and Peel", "Wristbands"}


# articleTypes that are not look items (cannot appear in a photographed outfit as one of our detectable classes)
SKIP_ARTICLE = {"Wallets", "Cufflinks", "Ties and Cufflinks", "Accessory Gift Set", "Suspenders", "Stockings", "Gloves", "Swimwear", "Water Bottle",
                "Umbrellas", "Wristbands", "Shoe Accessories", "Shoe Laces", "Key chain", "Perfume and Body Mist", "Hair Accessory", "Ring",
                "Jewellery Set", "Headband"}


def blank(path):
    g = np.asarray(Image.open(path).convert("L").resize((32, 32)), dtype=np.float32)
    return g.std() < 4  # essentially one flat colour


def scan(folder, ids):
    with ThreadPoolExecutor(8) as ex:
        return [i for i, b in zip(ids, ex.map(lambda i: blank(f"{folder}/{i}.jpg"), ids)) if b]


base = open(f"{ROOT}/data/ids.txt").read().split("\n")
meta = [json.loads(l) for l in open(f"{ROOT}/data/extended/meta.jsonl")]
blocked = set(json.load(open(f"{ROOT}/data/blocked.json"))) if os.path.exists(f"{ROOT}/data/blocked.json") else set()
b1 = scan(f"{ROOT}/data/catalog", base)
b2 = scan(f"{ROOT}/data/extended/img", [m["id"] for m in meta])
cat = [m["id"] for m in meta if m["masterCategory"] in NON_FASHION or m["subCategory"] in SKIP_SUB or m["articleType"] in SKIP_ARTICLE]
print(f"blank in catalog: {len(b1)}, blank in extended: {len(b2)}, non-fashion/skipped extended: {len(cat)}")
json.dump(sorted(blocked | set(b1) | set(b2) | set(cat)), open(f"{ROOT}/data/blocked.json", "w"))
print("blocked total:", len(blocked | set(b1) | set(b2) | set(cat)))
