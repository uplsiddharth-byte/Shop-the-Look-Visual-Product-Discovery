# Shop the Look

Upload an outfit photo: the app detects each garment or accessory, matches it to a product catalog, says whether the exact product exists, and explains every recommendation. Pretrained models only (OWLv2 detector, Marqo-FashionSigLIP embeddings); no training.

**[Try the live demo](https://uplsiddharth-byte.github.io/Shop-the-Look-Visual-Product-Discovery/)** (static: the six sample photos use results recorded from the real app; your own photos need the backend, see "Run the demo" below). Full write-up: [technical document (PDF)](docs/Shop_the_Look_Technical_Document.pdf).

| Upload screen | Results |
|---|---|
| ![Upload screen](docs/img/screen-upload.png) | ![Results for a red dress and a yellow handbag](docs/img/screen-results.png) |

## Results (288 validation pairs; true product within the top K shown for any detected item)
| Catalog searched | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|
| Provided catalog (28,094 products) | 4.9% | 10.4% | 15.3% | 28.8% |
| + optional extended catalog (72,166 products, default in the UI) | 4.9% | 9.4% | 13.9% | 26.7% |
| Random | | 0.02% | 0.04% | 0.36% |

Validation labels are loose (the paired product often only resembles the item in the scene), so these understate usefulness. Automated checks: 68 test images and 12 hostile-upload cases all pass, also from a clean CPU-only install (about 1.4 s per photo). Details and limitations are in the technical document.

## Run the demo (indexes already built in `data/`)
```bash
./run.sh                                   # open http://localhost:8000 (HOST, PORT, DEVICE=cpu|cuda|mps optional)
curl localhost:8000/api/health             # model load takes 30 to 60 s
```
Deploying elsewhere: see `DEPLOY.md` (what to ship, memory, security notes, Docker).

## Rebuild from scratch
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app/fetch.py              # catalog + validation scene images (554 s, 64 threads)
cd app
../.venv/bin/python embed.py siglip        # catalog embeddings (4 views for the ablation; ~36 min on M-series, ~9 min for 'full' only)
../.venv/bin/python detect_all.py          # detections for the 321 validation scenes
../.venv/bin/python final_eval.py          # headline metrics + exact-match threshold -> data/metrics.json, data/config.json
../.venv/bin/python evaluate.py siglip     # ablation: whole-scene vs crops, single vs multi-view
```

## Extended catalog (optional, +44,072 products)
```bash
# 5 parquet shards of benitomartin/fashion-product-images-small-384x512 -> data/extended/shard{0..4}.parquet (parallel curl, ~5 min)
cd app && ../.venv/bin/python ext_extract.py && ../.venv/bin/python ext_embed.py && ../.venv/bin/python build_blocklist.py
EXT=1 ../.venv/bin/python final_eval.py   # metrics with the extended index -> data/metrics_ext.json
```
The UI switch "Extended catalog" (or `?extended=false` on `/api/search`) turns it off.

## Tests (start the server first; set `API=localhost:8001` to target another server)
```bash
.venv/bin/python tests/check.py            # 68 images, expected result per group, exit 1 on any failure
.venv/bin/python tests/edge_cases.py       # 12 hostile/unusual uploads and their required HTTP status
.venv/bin/python tests/explain_dets.py tests/img/x.jpg   # why each detector box was kept or dropped
```

## Layout
```
app/        backend: detection, embedding, ranking, explanations, FastAPI server, evaluation + tuning scripts
web/        the UI (plain HTML/CSS/JS, no build step; Geist font and Phosphor icons vendored in web/vendor)
tests/      check.py (image suite), edge_cases.py (hostile uploads), explain_dets.py, make_test_images.py
docs/       technical document (HTML source + PDF)
data/       NOT in git: images, embeddings, caches (rebuild with the steps above, or see DEPLOY.md)
dataset/    NOT in git: the catalog and validation .jsonl files provided with the assignment
```
Key files: `app/pipeline.py` (detect, relabel, verify, declutter), `app/explain.py` (tags, families, explanations), `app/server.py` (API), `data/config.json` (tuned weights), `DEPLOY.md` (deployment), `app/build_demo_site.py` (builds the static demo).

## Third-party material
Models: OWLv2 (`google/owlv2-base-patch16-ensemble`) and Marqo-FashionSigLIP, downloaded from Hugging Face at first start. Fonts and icons in `web/vendor`: Geist (SIL OFL) and Phosphor Icons (MIT). The optional extended catalog is the public Fashion Product Images dataset (Hugging Face mirror `benitomartin/fashion-product-images-small-384x512`; license not stated on the mirror). Catalog and validation data come from the assignment and are not redistributed here.
