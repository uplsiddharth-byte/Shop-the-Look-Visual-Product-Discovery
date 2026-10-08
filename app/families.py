"""Ground-truth garment family for extended-catalog products, from their articleType metadata (more reliable than zero-shot tags)."""
import json, numpy as np
from embed import ROOT
from explain import FAMS

ARTICLE_FAMILY = {
    "bags": ["Handbags", "Clutches", "Backpacks", "Duffel Bag", "Messenger Bag", "Laptop Bag", "Trolley Bag", "Travel Accessory", "Mobile Pouch", "Waist Pouch", "Tablet Sleeve", "Rucksacks"],
    "belts": ["Belts"], "ties": ["Ties", "Cufflinks and Tie Pins"] if False else ["Ties"],
    "bracelets": ["Bracelet", "Bangle"], "necklaces": ["Necklace and Chains", "Pendant"], "earrings": ["Earrings"], "watches": ["Watches"],
    "eyewear": ["Sunglasses"], "hats": ["Caps", "Hat"], "scarves": ["Scarves", "Stoles", "Mufflers"],
    "shoes": ["Casual Shoes", "Sports Shoes", "Heels", "Sandals", "Flip Flops", "Formal Shoes", "Flats", "Sports Sandals"],
    "tops": ["Tshirts", "Shirts", "Tops", "Sweatshirts", "Sweaters", "Tunics", "Kurtis", "Kurtas"],
    "bottoms": ["Jeans", "Shorts", "Trousers", "Track Pants", "Skirts", "Capris", "Leggings", "Jeggings", "Churidar", "Patiala", "Salwar", "Tights"],
    "dresses": ["Dresses", "Jumpsuit", "Sarees"],
    "outerwear": ["Jackets", "Blazers", "Waistcoat", "Rain Jacket", "Shrug", "Suits", "Tracksuits", "Nehru Jackets"],
}
A2F = {a: f for f, arts in ARTICLE_FAMILY.items() for a in arts}


def ext_family_matrix(ids_ext):
    """(n_ext, len(FAMS)) one-hot rows for products with a mapped articleType, NaN rows for the rest (use zero-shot there)."""
    meta = {m["id"]: m for m in map(json.loads, open(f"{ROOT}/data/extended/meta.jsonl"))}
    M = np.full((len(ids_ext), len(FAMS)), np.nan, np.float32)
    for r, i in enumerate(ids_ext):
        f = A2F.get(meta[i]["articleType"])
        if f:
            M[r] = 0.0; M[r, FAMS.index(f)] = 1.0
    return M
