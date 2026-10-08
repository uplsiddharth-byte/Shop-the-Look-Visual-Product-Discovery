"""Extended catalog: unpack the 5 parquet shards in parallel -> data/extended/img/x<id>.jpg + data/extended/meta.jsonl"""
import json, os, sys, time
from multiprocessing import Pool
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/data/extended"; os.makedirs(f"{D}/img", exist_ok=True)
KEYS = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "usage", "productDisplayName"]


def shard(i):
    out = []
    pf = pq.ParquetFile(f"{D}/shard{i}.parquet")
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            pid = f"x{r['id']}"
            with open(f"{D}/img/{pid}.jpg", "wb") as f:
                f.write(r["image"]["bytes"])
            out.append({"id": pid, **{k: r[k] for k in KEYS}})
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(5) as p:
        rows = [r for part in p.map(shard, range(5)) for r in part]
    rows.sort(key=lambda r: r["id"])
    with open(f"{D}/meta.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    print(len(rows), "products extracted in", round(time.time() - t0), "s")
