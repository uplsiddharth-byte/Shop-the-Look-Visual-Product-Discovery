"""Shop-the-Look pipeline.
scene -> detect_raw (OWLv2) -> select (thresholds, overlap + containment rules) -> analyze (embed crops, verify each crop
really looks like its label and is not a non-clothing region) -> cosine search over the catalog.
`select` and `analyze` are plain functions so the server and the validation scripts run exactly the same logic."""
import json, os, numpy as np, torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from embed import Embedder, VIEWS, dev, ROOT
from explain import FAMS, LABEL_FAM, WORN, ATTRS, FAMILY

# prompt -> label shown to the user. Synonym prompts feed the same label (OWLv2 scores each prompt separately).
PROMPTS = {"a top": "top", "a shirt": "shirt", "a dress": "dress", "a jacket": "jacket", "a coat": "coat", "a pair of pants": "pants",
           "a pair of jeans": "jeans", "a skirt": "skirt", "a pair of shorts": "shorts", "a pair of shoes": "shoes",
           "a pair of boots": "boots", "a pair of sneakers": "sneakers", "a pair of high heels": "heels", "a handbag": "handbag",
           "a sweater": "sweater", "a pair of sunglasses": "sunglasses", "a pair of earrings": "earrings", "a necklace": "necklace",
           "a hat": "hat", "a belt": "belt", "a leather belt": "belt", "a watch": "watch", "a scarf": "scarf", "a bracelet": "bracelet",
           "a suit": "suit", "a blazer": "blazer", "a vest": "vest", "a tie": "tie", "a bow tie": "tie"}
GARMENTS = list(PROMPTS)
LABEL = PROMPTS
# Low-confidence classes (small, thin or occluded) need a lower cut-off than garments; default is 0.2.
MIN_SCORE = {"shoes": .1, "boots": .1, "sneakers": .1, "heels": .1, "belt": .2}
FOOTWEAR = {"shoes", "boots", "sneakers", "heels"}
BODY = {"dress", "suit"}  # one box that covers a whole outfit piece: garments inside it are not separate items
PARTS = {"top", "shirt", "sweater", "jacket", "coat", "blazer", "vest", "pants", "jeans", "skirt", "shorts"}
OWL = "google/owlv2-base-patch16-ensemble"
MAX_NONFASHION = 0.5   # crop rejected if it looks like a car, skin, streaks... with at least this probability
MIN_FAMILY = 0.1       # crop must look at least this much like the family of its label, unless the detector is confident
SURE = 0.45            # detector score above which the family check is skipped
MAX_ITEMS = 6
WEAK_SCORE, WEAK_FAM = 0.1, 0.7  # a garment detection scoring 0.1 to 0.2 is kept only if the crop looks >= 70% like that kind of garment
GARMENT_FAMS = {"tops", "bottoms", "dresses", "outerwear"}
RELABEL_MARGIN, RELABEL_MIN = 3.0, 0.4  # crop says another garment family is at least 3x likelier than the label's and >= 40%: trust the crop
# Classes that are easy to hallucinate (waistbands, hems and hands look like belts) need stronger evidence than other items:
# a confident detection, or a moderate one that the crop also clearly looks like that class.
STRICT = {"belt": dict(sure=0.62, fam=0.6, min=0.4)}


class Pipeline:
    def __init__(self, backend="siglip", views=("full",), with_ext=True):  # multi-view pooling hurt recall on validation
        self.E = Embedder(backend)
        self.owl = Owlv2ForObjectDetection.from_pretrained(OWL).to(dev).eval()
        self.owl_proc = Owlv2Processor.from_pretrained(OWL)
        self.ids = open(f"{ROOT}/data/ids.txt").read().split("\n")
        blocked = set(json.load(open(f"{ROOT}/data/blocked.json"))) if os.path.exists(f"{ROOT}/data/blocked.json") else set()
        self.block = np.array([i in blocked for i in self.ids])  # blank images, placeholders, non-fashion are never recommended
        self.embs = {v: np.load(f"{ROOT}/data/emb_{backend}_{v}.npy") for v in views}
        self.n_base = len(self.ids)
        self.ext_meta = {}
        if with_ext and os.path.exists(f"{ROOT}/data/emb_{backend}_ext.npy") and "full" in self.embs:  # extended catalog: full-image view only
            self.ids += open(f"{ROOT}/data/ids_ext.txt").read().split("\n")
            self.embs["full"] = np.concatenate([self.embs["full"], np.load(f"{ROOT}/data/emb_{backend}_ext.npy")])
            self.block = np.concatenate([self.block, np.array([i in blocked for i in self.ids[self.n_base:]])])
            self.ext_meta = {m["id"]: m for m in map(json.loads, open(f"{ROOT}/data/extended/meta.jsonl"))}

    @torch.no_grad()
    def detect_raw(self, img, thr=0.05):
        """All OWLv2 boxes above a very low score, minus specks. Selection happens in `select` so it can be re-tuned without re-detecting."""
        inp = self.owl_proc(text=[GARMENTS], images=img, return_tensors="pt").to(dev)
        out = self.owl(**inp)
        side = max(img.size)  # OWLv2 pads to a square
        res = self.owl_proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=torch.tensor([[side, side]]).to(dev))[0]
        dets = [dict(box=[max(0, b[0]), max(0, b[1]), min(img.width, b[2]), min(img.height, b[3])],
                     score=float(s), label=LABEL[GARMENTS[int(l)]]) for b, s, l in
                zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist())]
        L = max(img.size)  # a real item is at least ~6% of the long side in one direction (belts are thin in the other)
        return [d for d in dets if max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= .06 * L
                and min(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= .012 * L]

    def scores(self, q, extended=True):
        # cosine (L2-normalised); a product scores its best view. Exact search is instant at 28k-72k items.
        S = np.max([q @ e.T for e in self.embs.values()], axis=0)
        S[:, self.block] = -1
        if not extended:
            S[:, self.n_base:] = -1  # original catalog only
        return S


def candidates(dets):
    """Thresholds -> merge shoes -> NMS. (Overlap/containment rules run later, once labels are verified.)"""
    keep = []
    for d in dets:
        need = MIN_SCORE.get(d["label"], 0.2)
        # partly visible garments (a half-cropped shirt or trousers) score low; keep them as "weak" and let the crop check decide
        weak = d["score"] < need and d["score"] >= WEAK_SCORE and LABEL_FAM[d["label"]] in GARMENT_FAMS
        if d["score"] >= need or weak:
            keep.append(dict(d, weak=weak))
    dets = merge_footwear(keep)
    dets.sort(key=lambda d: -d["score"])
    keep = []
    for d in dets:  # greedy NMS across labels: one box per region
        if all(iou(d["box"], k["box"]) < 0.5 for k in keep):
            keep.append(d)
    return keep[:16]


def declutter(items):
    """Drop boxes that are part of, or duplicates of, a bigger box. items = [(det, embedding)] with verified labels."""
    out = []
    for d, v in items:
        inside = [k for k, _ in items if k is not d and inside_frac(d["box"], k["box"]) >= 0.7]
        # a top/skirt/jacket box inside a whole-dress or whole-suit box is part of that dress, not a separate product
        if any(k["label"] in BODY and d["label"] in PARTS and d["score"] < k["score"] + 0.15 for k in inside):
            continue
        # a box inside a bigger box of the same kind (handbag strap inside handbag, jeans inside pants) is a duplicate
        if any(LABEL_FAM[k["label"]] == LABEL_FAM[d["label"]] and (k["score"] > d["score"] or (k["score"] == d["score"] and id(k) < id(d))) for k in inside):
            continue
        out.append((d, v))
    return out


def analyze(img, raw, E, X):
    """raw detections -> verified items + crop embeddings: (dets, q) with q[i] the embedding of dets[i]'s crop.
    1. embed every candidate crop; 2. relabel when the crop clearly looks like another kind of item; 3. drop crops that look like
    non-clothing (car, streaks, bare background) or do not look like their label; 4. drop parts/duplicates of bigger boxes.
    If nothing survives but the whole image is clearly one clothing item (flat-lay product photo) it is used; otherwise no items."""
    cands = candidates(raw)
    items = []
    if cands:
        Q = E.images([img.crop(tuple(int(v) for v in d["box"])) for d in cands])
        for d, v in zip(cands, Q):
            d = dict(d)
            fp, non, word, pw = X.check(v, worn=d["label"] in WORN)
            own = fp[FAMS.index(LABEL_FAM[d["label"]])]
            bf = int(fp.argmax())
            lf = LABEL_FAM[d["label"]]
            if (lf in GARMENT_FAMS and FAMS[bf] in GARMENT_FAMS and FAMS[bf] != lf and fp[bf] >= RELABEL_MIN
                    and fp[bf] >= RELABEL_MARGIN * own):  # e.g. a short-sleeve shirt detected as "coat"; never into/out of shoes or accessories
                d["label"] = word
                own = fp[FAMS.index(LABEL_FAM[word])]
            elif (d["label"] in ATTRS["category"] and lf in GARMENT_FAMS and FAMILY[word] == lf and word != d["label"]
                  and pw[ATTRS["category"].index(d["label"])] < 0.2 and pw.max() >= 0.6):  # same family, wrong word: "skirt" -> "pants"
                d["label"] = word
            st = STRICT.get(d["label"])
            if st and not (d["score"] >= st["sure"] or (own >= st["fam"] and d["score"] >= st["min"])):
                continue
            if d.get("weak") and not (own >= WEAK_FAM and non < 0.1):
                continue
            if non < MAX_NONFASHION and (own >= MIN_FAMILY or (d["score"] >= SURE and own >= 0.03)):
                items.append((d, v))
        items = declutter(items)[:MAX_ITEMS]
    if items:
        return [d for d, _ in items], np.stack([v for _, v in items])
    v = E.images([img])[0]
    fp, non, _, _ = X.check(v)
    if non < MAX_NONFASHION and fp.max() >= 0.3:
        return [dict(box=[0, 0, img.width, img.height], score=0.0, label="item")], v[None]
    return [], np.zeros((0, v.shape[0]), np.float32)


def merge_footwear(dets):
    """Left and right shoe are detected separately; show one 'shoes' item covering both feet."""
    f = [d for d in dets if d["label"] in FOOTWEAR]
    if len(f) < 2:
        return dets
    box = [min(d["box"][0] for d in f), min(d["box"][1] for d in f), max(d["box"][2] for d in f), max(d["box"][3] for d in f)]
    best = max(f, key=lambda d: d["score"])
    return [d for d in dets if d["label"] not in FOOTWEAR] + [dict(box=box, score=best["score"], label=best["label"])]


def inside_frac(a, b):
    """Fraction of box a's area that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u else 0
