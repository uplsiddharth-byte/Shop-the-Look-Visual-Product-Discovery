"""Download the runtime data (index, config, images) into data/ from a Hugging Face dataset repo, unless it is already there.
Used by the Space start-up script. Needs DATA_REPO (e.g. user/shop-the-look-data) and, for a private repo, HF_TOKEN."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data"
NEED = ["ids.txt", "emb_siglip_full.npy", "hub.npy", "blocked.json", "config.json"]

if all(os.path.exists(f"{DATA}/{f}") for f in NEED) and os.path.isdir(f"{DATA}/catalog"):
    print("data/ already present, nothing to download")
    sys.exit(0)
repo = os.environ.get("DATA_REPO")
if not repo:
    sys.exit("data/ is missing and DATA_REPO is not set: set DATA_REPO (and HF_TOKEN for a private repo)")
from huggingface_hub import snapshot_download
print(f"downloading data from {repo} into {DATA} ...", flush=True)
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=DATA, token=os.environ.get("HF_TOKEN"), max_workers=16)
import tarfile
for tar, dest in (("catalog.tar", "catalog"), ("ext_img.tar", "extended/img")):  # image folders travel as archives
    path = f"{DATA}/{tar}"
    if os.path.exists(path):
        print(f"unpacking {tar} ...", flush=True)
        os.makedirs(f"{DATA}/{dest}", exist_ok=True)
        with tarfile.open(path) as t:
            t.extractall(f"{DATA}/{dest}", filter="data")
        os.remove(path)  # free the disk
print("done", flush=True)
