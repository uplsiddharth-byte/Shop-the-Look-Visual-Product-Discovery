"""Publish the app to Hugging Face: a PRIVATE dataset repo with the runtime data, plus a Docker Space that runs the server.

Run after logging in (`hf auth login`, a token with Write access):
    python deploy/publish_to_hf.py              # upload everything, create the Space, wait for it to start
    python deploy/publish_to_hf.py --dry-run    # no network: build the Space folder and list what would be uploaded

The Space needs a token to read the private data repo at start-up. Preferred: create a READ-only token and pass it as
SPACE_READ_TOKEN; otherwise the logged-in token is used (it can also write, so a read-only one is safer).
Environment: SPACE_READ_TOKEN (optional), --data-repo / --space to override the default names.
"""
import argparse, glob, os, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMALL_FILES = ["ids.txt", "ids_ext.txt", "emb_siglip_full.npy", "emb_siglip_ext.npy", "hub.npy", "blocked.json", "config.json", "extended/meta.jsonl"]
SMALL_DIRS = ["scenes"]                       # 321 sample photos: fine as loose files
TARS = {"catalog.tar": "catalog", "ext_img.tar": "extended/img"}   # 28k + 44k JPEGs packed (Hub repos dislike tens of thousands of files in a folder)
RUNTIME_PY = ["embed.py", "explain.py", "families.py", "fetch_data.py", "pipeline.py", "server.py"]


def assemble_space(dest: Path):
    """Space repo layout: Dockerfile, README.md (Space card), requirements.txt, start.sh, app/ (runtime code only), web/."""
    for f in (ROOT / "deploy/hf_space").iterdir():
        shutil.copy(f, dest / f.name)
    (dest / "app").mkdir()
    for f in RUNTIME_PY:
        shutil.copy(ROOT / "app" / f, dest / "app" / f)
    shutil.copytree(ROOT / "web", dest / "web")


def data_summary():
    n = sum(1 for f in SMALL_FILES if (ROOT / "data" / f).exists()) + sum(len(os.listdir(ROOT / "data" / d)) for d in SMALL_DIRS) + len(TARS)
    size = sum(os.path.getsize(ROOT / "data" / f) for f in SMALL_FILES if (ROOT / "data" / f).exists())
    size += sum(os.path.getsize(p) for d in SMALL_DIRS for p in glob.glob(str(ROOT / "data" / d / "*")))
    size += sum(os.path.getsize(p) for d in TARS.values() for p in glob.glob(str(ROOT / "data" / d / "*.jpg")))
    return n, size / 1048576


def stage_data(dest: Path):
    """Copy the small files and build the two image archives (plain tar: JPEGs do not compress)."""
    import tarfile
    for f in SMALL_FILES:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / f, dest / f)
    for d in SMALL_DIRS:
        shutil.copytree(ROOT / "data" / d, dest / d)
    for tar, d in TARS.items():
        print(f"  packing {d} into {tar} ...", flush=True)
        with tarfile.open(dest / tar, "w") as t:
            for p in sorted(glob.glob(str(ROOT / "data" / d / "*.jpg"))):
                t.add(p, arcname=os.path.basename(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-repo"); ap.add_argument("--space")
    a = ap.parse_args()

    n, mb = data_summary()
    print(f"data to upload: {n} files in the dataset repo (images packed in 2 archives), {mb:,.0f} MB")
    tmp = Path(tempfile.mkdtemp())
    assemble_space(tmp)
    listing = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
    print(f"Space folder: {len(listing)} files:", ", ".join(listing[:12]), "...")
    if a.dry_run:
        shutil.rmtree(tmp)
        return

    from huggingface_hub import HfApi, get_token
    api = HfApi()
    who = api.whoami()["name"]
    data_repo = a.data_repo or f"{who}/shop-the-look-data"
    space = a.space or f"{who}/shop-the-look"
    token = os.environ.get("SPACE_READ_TOKEN") or get_token()
    if not os.environ.get("SPACE_READ_TOKEN"):
        print("note: no SPACE_READ_TOKEN set, the Space will use your logged-in token (prefer a read-only one)")

    print(f"1/4 private dataset {data_repo}: uploading data (resumable; can be re-run) ...", flush=True)
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp())
    stage_data(stage)
    api.upload_large_folder(repo_id=data_repo, repo_type="dataset", folder_path=str(stage))
    shutil.rmtree(stage)

    print(f"2/4 Space {space}: creating ...", flush=True)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space, "HF_TOKEN", token)
    api.add_space_variable(space, "DATA_REPO", data_repo)

    print("3/4 uploading code ...", flush=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(tmp), commit_message="Deploy Shop the Look")
    shutil.rmtree(tmp)

    print("4/4 waiting for the build and start-up (first start downloads about 5 GB, allow 10 to 20 minutes) ...", flush=True)
    url = f"https://{who.lower()}-{space.split('/')[1]}.hf.space"
    last = None
    for _ in range(180):
        stage = api.get_space_runtime(space).stage
        if stage != last:
            print("  stage:", stage, flush=True); last = stage
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            break
        time.sleep(10)
    print(f"Space page: https://huggingface.co/spaces/{space}\nApp URL:    {url}\nFinal stage: {last}")
    sys.exit(0 if last == "RUNNING" else 1)


if __name__ == "__main__":
    main()
