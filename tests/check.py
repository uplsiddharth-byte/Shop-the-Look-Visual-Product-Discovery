"""Automated pass/fail over every image in tests/img against the LIVE API (start the server first).
Usage: python tests/check.py [name-filter]. Exit code 1 if anything fails."""
import os as _os
API = _os.environ.get('API', 'localhost:8000')  # target server host:port
import glob, json, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from explain import LABEL_FAM

HERE = os.path.dirname(__file__)
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
HALF = {"Tshirts": {"tops"}, "Shirts": {"tops"}, "Trousers": {"bottoms"}, "TrackPants": {"bottoms"}, "Jeans": {"bottoms"},
        "Tracksuits": {"tops", "bottoms", "outerwear"}}


def search(path):
    out = subprocess.run(["curl", "-s", "-m", "60", "-F", f"file=@{path}", API + "/api/search?extended=true"], capture_output=True, text=True).stdout
    return json.loads(out)


def fam(i): return LABEL_FAM.get(i["label"])
def labels(d): return [i["label"] for i in d["items"]]
def fams(d): return {fam(i) for i in d["items"]}
def top5_colors(i): return [m["tags"].get("color") for m in i["matches"]]


def verdict(name, d):
    """returns (ok, note) from the expectations for this image"""
    L, F = labels(d), fams(d)
    if name.startswith("nf_") or name == "case_car_lambo.jpg":
        return not d["items"], "no clothing expected, got " + str(L)
    m = re.match(r"half_(\w+?)_x(\d+)_(\w+)\.jpg", name)
    if m:
        ok = bool(F & HALF[m.group(1)])
        own = any(x["id"] == "x" + m.group(2) for i in d["items"] for x in i["matches"])
        return ok, f"own product in top-5: {own}; items {L}"
    if name == "blur1.jpg": return "dress" in L and len(L) == 1, f"one dress expected, got {L}"
    if name == "blur2.jpg": return True, f"informational (heavy blur): {L}"
    if name == "blur3.jpg": return L == ["dress"], f"only a dress expected, got {L}"
    if name == "crop_tee.jpg": return "tops" in F and bool(F & {"watches", "bracelets"}), f"top + wrist item expected, got {L}"
    if name in ("crop_trouser.jpg", "crop_trouser2.jpg"): return "bags" in F and not (F & {"belts", "ties"}) and L.count("handbag") == 1, f"one bag, no belt/tie, got {L}"
    if name == "no_belt.jpg": return "belts" not in F and "bottoms" in F, f"no belt expected, got {L}"
    if name == "short_sleeve.jpg":
        i = d["items"][0] if d["items"] else None
        n = sum(1 for x in (i["matches"] if i else []) if x["tags"].get("sleeve") == "short sleeve")
        return bool(i) and fam(i) == "tops" and i["tags"].get("sleeve") == "short sleeve" and n >= 3, f"short-sleeve top, {n}/5 short-sleeve matches, label {L}"
    if name == "wrong_op.jpg":
        dr = [i for i in d["items"] if i["label"] == "dress"]; bg = [i for i in d["items"] if i["label"] == "handbag"]
        ok = bool(dr) and dr[0]["tags"].get("color") == "red" and "black" not in top5_colors(dr[0]) and bool(bg) and set(top5_colors(bg[0])) <= {"yellow", "gold", "orange", "beige"}
        return ok, f"red dress without black matches, yellow bag; dress colours {top5_colors(dr[0]) if dr else None}"
    if name == "case_man_belt_shoes.jpg": return {"shoes", "bottoms", "tops"} <= F, f"shoes + bottoms + top expected, got {L}"
    if name == "case_man_shoes.jpg": return {"shoes", "bottoms", "tops"} <= F, f"shoes + bottoms + top expected, got {L}"
    if name == "case_suit.jpg": return "outerwear" in F and "eyewear" in F, f"suit/jacket + glasses expected, got {L}"
    return True, "no expectation"


fails = 0; n = 0; own_hits = [0, 0]
imgs = sorted(glob.glob(os.path.join(HERE, "img", "*.jpg")))
if not imgs:
    sys.exit("No test images in tests/img. Run `python tests/make_test_images.py` (needs the extended data bundle) "
             "or unpack shop-the-look-test-images.tar in the repo root (see README, section Tests).")
if len(imgs) < 68 and not FILTER:
    print(f"note: {len(imgs)} of the 68 test images are present, so this is a partial run (see README, section Tests)\n")
for p in imgs:
    name = os.path.basename(p)
    if FILTER not in name: continue
    try: d = search(p)
    except Exception as e: print("ERROR", name, e); fails += 1; continue
    ok, note = verdict(name, d); n += 1
    mm = re.match(r"half_.*_(full|tophalf|bottomhalf|lefthalf)\.jpg", name)
    if mm and "own product in top-5: True" in note: own_hits[0] += 1
    if mm: own_hits[1] += 1
    if not ok: fails += 1
    print(("PASS " if ok else "FAIL ") + name[:40].ljust(40), note[:150])
print(f"\n{n - fails}/{n} passed" + (f" | own product found in top-5 for {own_hits[0]}/{own_hits[1]} half/full garment images" if own_hits[1] else ""))
sys.exit(1 if fails else 0)
