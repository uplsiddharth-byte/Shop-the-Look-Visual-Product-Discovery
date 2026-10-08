"""Explanations: zero-shot attribute tags (category/color/pattern/material/style) on the scene crop and on each match,
then a sentence built from which attributes agree. Works on embeddings we already have (no extra image passes)."""
import numpy as np

ATTRS = {
    "category": ["top", "blouse", "t-shirt", "sweater", "dress", "jacket", "coat", "blazer", "pants", "jeans", "skirt",
                 "shorts", "shoes", "boots", "sandals", "handbag", "jumpsuit", "cardigan", "sunglasses", "earrings", "necklace",
                 "hat", "belt", "watch", "scarf", "bracelet", "suit", "vest", "tie", "bow tie", "sneakers", "heels", "shirt", "polo shirt", "hoodie", "tank top", "kurta"],
    "color": ["black", "white", "grey", "beige", "brown", "red", "pink", "orange", "yellow", "green", "blue", "navy",
              "purple", "denim blue", "gold", "silver"],
    "pattern": ["solid", "striped", "floral", "plaid", "polka dot", "animal print", "lace", "geometric print"],
    "material": ["cotton", "denim", "leather", "knit", "silk", "lace", "chiffon", "wool", "suede"],
    "sleeve": ["short sleeve", "long sleeve", "sleeveless"],
    "style": ["casual", "formal", "bohemian", "sporty", "elegant", "streetwear", "vintage"],
}
TEMPLATE = {"category": "a photo of a {}", "color": "a {} colored garment", "pattern": "a {} pattern garment",
            "material": "a garment made of {}", "style": "a {} style outfit", "sleeve": "a {} top or shirt"}
# Words the user sees; categories that are near-synonyms count as the same family.
FAMILY = {"top": "tops", "blouse": "tops", "t-shirt": "tops", "sweater": "tops", "cardigan": "tops",
          "pants": "bottoms", "jeans": "bottoms", "shorts": "bottoms", "skirt": "bottoms",
          "dress": "dresses", "jumpsuit": "dresses", "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear",
          "suit": "outerwear", "vest": "outerwear", "shirt": "tops", "polo shirt": "tops", "hoodie": "tops", "tank top": "tops", "kurta": "tops",
          "shoes": "shoes", "boots": "shoes", "sandals": "shoes", "sneakers": "shoes", "heels": "shoes", "handbag": "bags",
          "sunglasses": "eyewear", "earrings": "earrings", "necklace": "necklaces", "bracelet": "bracelets", "watch": "watches",
          "hat": "hats", "belt": "belts", "scarf": "scarves", "tie": "ties", "bow tie": "ties"}
# Detector label -> garment family (used to favour catalog products of the same kind as the detected item)
LABEL_FAM = {"top": "tops", "shirt": "tops", "sweater": "tops", "dress": "dresses", "jacket": "outerwear", "coat": "outerwear",
             "pants": "bottoms", "jeans": "bottoms", "skirt": "bottoms", "shorts": "bottoms", "shoes": "shoes", "boots": "shoes",
             "sneakers": "shoes", "heels": "shoes", "handbag": "bags", "sunglasses": "eyewear", "earrings": "earrings",
             "necklace": "necklaces", "bracelet": "bracelets", "watch": "watches", "hat": "hats", "belt": "belts", "scarf": "scarves",
             "suit": "outerwear", "blazer": "outerwear", "vest": "outerwear", "tie": "ties", "bow tie": "ties"}
LABEL_FAM = {**FAMILY, **LABEL_FAM}


def _color_sim():
    """16x16 'how close are these two colour names' table: 1 on the diagonal, partial credit for neighbours (red~pink, navy~blue...)."""
    names = ATTRS["color"]
    near = {("black", "grey"): .4, ("black", "navy"): .4, ("black", "brown"): .2, ("grey", "silver"): .6, ("grey", "white"): .3,
            ("white", "beige"): .5, ("white", "silver"): .3, ("beige", "brown"): .4, ("beige", "gold"): .3, ("beige", "yellow"): .2,
            ("brown", "orange"): .2, ("brown", "gold"): .3, ("red", "pink"): .5, ("red", "orange"): .3, ("pink", "purple"): .3,
            ("orange", "yellow"): .4, ("orange", "gold"): .3, ("yellow", "gold"): .6, ("blue", "navy"): .6, ("blue", "denim blue"): .8,
            ("navy", "denim blue"): .5, ("blue", "purple"): .2, ("green", "blue"): .1}
    M = np.eye(len(names), dtype=np.float32)
    for (a, b), v in near.items():
        M[names.index(a), names.index(b)] = M[names.index(b), names.index(a)] = v
    return M

COLOR_SIM = _color_sim()
FAMS = sorted(set(FAMILY.values()))
# families the ranking treats as one group (a wrist item is detected as 'watch' or 'bracelet' interchangeably)
GROUP = {"watches": ["watches", "bracelets"], "bracelets": ["bracelets", "watches"]}
BODY_PARTS = {"face", "hand", "bare skin", "hair"}  # normal background for body-worn accessories, so ignored for them
WORN = {"watch", "bracelet", "necklace", "earrings", "sunglasses", "hat", "scarf", "tie", "belt"}
SLEEVE_FAMS = {"tops", "outerwear", "dresses"}  # the only kinds of item that have a sleeve length
MIN_CONF = 0.4  # softmax probability needed before an attribute is asserted in the UI or the explanation
NO_SURFACE = {"eyewear", "necklaces", "earrings", "bracelets", "watches", "belts", "ties"}  # pattern/material tags are meaningless for these


# Things a detector box can wrongly land on (streaks, skin, vehicles, scenery). A crop that looks like one of these is rejected.
NEGATIVES = ["car", "wheel", "building", "tree", "landscape", "food", "animal", "furniture", "electronic device", "text", "face",
             "hand", "bare skin", "hair", "wall", "road", "bottle", "blurry background", "light streaks"]


class Explainer:
    def __init__(self, embedder):
        self.T = {a: embedder.texts([TEMPLATE[a].format(w) for w in ws]) for a, ws in ATTRS.items()}
        self.neg = embedder.texts([f"a photo of a {w}" for w in NEGATIVES])
        self.fam_of = np.array([FAMS.index(FAMILY[w]) for w in ATTRS["category"]])

    def check(self, vec, worn=False):
        """One crop embedding -> (family probabilities over FAMS, probability it is NOT clothing, best category word).
        worn=True (watch, bracelet, necklace...): skin/hand/face/hair do not count as non-clothing."""
        s = np.concatenate([self.T["category"], self.neg]) @ vec * 100
        p = np.exp(s - s.max()); p /= p.sum()
        n = len(ATTRS["category"])
        neg = p[n:] * (0 if not worn else np.array([w not in BODY_PARTS for w in NEGATIVES]))
        return (np.array([p[:n][self.fam_of == f].sum() for f in range(len(FAMS))]), float(neg.sum()),
                ATTRS["category"][int(p[:n].argmax())], p[:n])

    def bonus_vec(self, PF, label):
        """Per-product family bonus column for a detected label (None if the label has no family)."""
        fam = LABEL_FAM.get(label)
        if not fam:
            return None
        return PF[:, [FAMS.index(f) for f in GROUP.get(fam, [fam])]].max(1)

    def probs(self, emb, attr):
        """(N, 512) product embeddings -> (N, len(ATTRS[attr])) zero-shot probabilities."""
        s = emb @ self.T[attr].T * 100
        p = np.exp(s - s.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def family_probs(self, emb):
        """(N, 512) product embeddings -> (N, len(FAMS)) zero-shot garment-family probability."""
        p = self.probs(emb, "category")
        return np.stack([p[:, self.fam_of == f].sum(1) for f in range(len(FAMS))], 1)

    def tags(self, vec):
        """vec: (512,) normalised image embedding -> {attr: (word, confidence)} using a softmax over prompts."""
        out = {}
        for a, ws in ATTRS.items():
            s = self.T[a] @ vec * 100
            p = np.exp(s - s.max()); p /= p.sum()
            out[a] = (ws[int(p.argmax())], float(p.max()))
        if FAMILY.get(out["category"][0]) not in SLEEVE_FAMS:
            out["sleeve"] = (out["sleeve"][0], 0.0)
        if FAMILY.get(out["category"][0]) in NO_SURFACE:
            out["pattern"] = out["material"] = out["style"] = (out["pattern"][0], 0.0)
        return out

    def explain(self, q_tags, m_tags, sim):
        agree, differ = [], []
        for a in ("category", "color", "pattern", "material", "style", "sleeve"):
            qw, qc = q_tags[a]; mw, mc = m_tags[a]
            if min(qc, mc) < MIN_CONF:  # only state what both models are reasonably sure about
                continue
            same = qw == mw or (a == "category" and FAMILY.get(qw) == FAMILY.get(mw))
            (agree if same else differ).append((a, qw, mw))
        txt = []
        if agree:
            parts = [f"{a} ({w})" for a, w, _ in agree]
            txt.append("Matches on " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + ".")
        if differ:
            d = [f"{a}: {q} vs {m}" for a, q, m in differ if a in ("color", "pattern", "category", "sleeve")][:2]
            if d:
                txt.append("Differs in " + "; ".join(d) + ".")
        if not txt:
            txt.append("Closest overall visual match.")
        return {"text": " ".join(txt), "matched": [a for a, _, _ in agree], "differs": [a for a, _, _ in differ]}
