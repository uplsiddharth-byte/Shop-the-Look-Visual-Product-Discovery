"""Embed the extended catalog (same model + full-image view as the main catalog) -> data/emb_siglip_ext.npy"""
import json, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from embed import Embedder, ROOT

E = Embedder("siglip")
meta = [json.loads(l) for l in open(f"{ROOT}/data/extended/meta.jsonl")]
ids = [m["id"] for m in meta]
load = lambda p: Image.open(f"{ROOT}/data/extended/img/{p}.jpg").convert("RGB")
out, B, t0 = [], 128, time.time()
with ThreadPoolExecutor(6) as ex:
    nxt = ex.map(load, ids)  # lazy, prefetching
    for i in range(0, len(ids), B):
        out.append(E.images([next(nxt) for _ in ids[i:i + B]]))
        if (i // B) % 20 == 0:
            print(f"{i}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
np.save(f"{ROOT}/data/emb_siglip_ext.npy", np.concatenate(out))
open(f"{ROOT}/data/ids_ext.txt", "w").write("\n".join(ids))
print("DONE", round(time.time() - t0), "s", flush=True)
