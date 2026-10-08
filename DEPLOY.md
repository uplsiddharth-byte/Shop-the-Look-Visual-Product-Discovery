# Deploying Shop the Look

One Python process (FastAPI + the models) that also serves the web UI and the product images. No database, no build step.

## What was verified (on this machine, 8 Oct 2026)
- A **fresh virtualenv built only from `requirements.txt`**, forced to **CPU** (`DEVICE=cpu`), started from `run.sh`-equivalent and passed **68/68 image checks** and **12/12 edge-case checks** (`tests/check.py`, `tests/edge_cases.py` with `API=localhost:8001`). Same results as the Apple-GPU run.
- Latency per photo: about 0.6 s (Apple GPU), about 1.4 s (Apple M-series CPU). Slower CPUs will be slower; one request runs at a time (a lock protects the models), so size the machine for your traffic.
- Memory: about 0.7 GB RSS while serving (more during the first request). Plan for 2 GB.
- **Not verified:** the Dockerfile (Docker's background service was not running here), CUDA, and Linux. Treat them as likely to work, not proven.

## 1. What must be on the server
| Item | Size | Purpose |
|---|---|---|
| Python 3.12 + `pip install -r requirements.txt` | about 2 GB | torch, transformers, open-clip, fastapi |
| Model weights, downloaded from Hugging Face on first start (needs internet once) | about 2 GB | OWLv2 detector, Marqo-FashionSigLIP |
| `data/emb_siglip_full.npy`, `emb_siglip_ext.npy`, `ids.txt`, `ids_ext.txt`, `hub.npy`, `blocked.json`, `config.json`, `extended/meta.jsonl` | about 220 MB | the search index and tuned settings |
| `data/catalog/` (28,094 JPEGs) | 592 MB | product images shown in results |
| `data/extended/img/` (44,072 JPEGs) | 2.3 GB | extended-catalog images (optional: switch the feature off by omitting `emb_siglip_ext.npy` and this folder) |
| `data/scenes/` | 12 MB | only for the sample thumbnails |
| `web/` | 0.3 MB | the UI |

Not needed at runtime: `dataset/`, `tests/`, `docs/`, `app/tune_*.py`, `data/dets_raw.pkl`, `data/*.log`.

If the catalog images should come from a CDN instead, change the two `app.mount("/catalog"...)` / `("/ext"...)` lines in `app/server.py` and the `image` URLs it returns.

## 2. Start
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
HOST=0.0.0.0 PORT=8000 ./run.sh          # DEVICE=cpu|cuda|mps optional (auto-detected)
curl localhost:8000/api/health            # {"status":"ok","products":72166,"device":"..."}
```
The server needs 30 to 60 s to load models before it answers. If `data/` is incomplete it exits with a message naming the missing files.
After the first successful start you can set `HF_HUB_OFFLINE=1` so it never contacts Hugging Face again.

## 3. Production notes
- **No authentication.** There is a built-in limit of `RATE_PER_MIN` (default 20) photos per visitor per minute (visitors are identified by the proxy's `CF-Connecting-IP` / `X-Forwarded-For`; requests from the machine itself are exempt) and at most `MAX_WAITING` (default 6) queued requests, after which the server answers 429 or 503. For a public site still put it behind a reverse proxy that adds HTTPS and, if needed, auth.
- Uploads: max 10 MB and 50 megapixels, enforced before decoding; bad files get a clean 400/413.
- Run **one worker** (`uvicorn` default). More workers each load the models (2 GB each) and gain nothing on one GPU.
- UI files are served with `Cache-Control: no-cache` so redeploys are picked up immediately.
- Settings and thresholds live in `data/config.json` (tuned on the validation set; see `docs/`). Changing detector prompts or weights means re-running `app/final_eval.py`.
- The extended catalog is from the public Fashion Product Images dataset (Hugging Face mirror `benitomartin/fashion-product-images-small-384x512`). Its license is not stated on the mirror: check before commercial use.

## 4. Docker (untested here)
```bash
docker build -t shop-the-look .
docker run -p 8000:8000 -v "$PWD/data:/app/data" -v hf-cache:/root/.cache/huggingface shop-the-look
```
The image contains code and dependencies only; mount `data/` as a volume. The container uses CPU.

## 5. Putting the real app online (uploads work)
GitHub Pages cannot run the models, so a live app needs a server. Two ways, both tested here:
- **Quick public link from your own machine** (works while the machine is on, awake and the server is running; the link changes each run; the server on your machine becomes reachable from the internet):
  ```bash
  ./run.sh &                                              # the app on localhost:8000
  cloudflared tunnel --url http://localhost:8000          # prints https://<random>.trycloudflare.com  (brew install cloudflared)
  ```
- **Permanent, free: a Hugging Face Space** (Docker, free CPU, 16 GB RAM; about 5 to 15 s per photo; sleeps after 2 idle days and wakes in a few minutes). One command after `hf auth login` (a free account and a Write token):
  ```bash
  python deploy/publish_to_hf.py --dry-run     # what would be uploaded (no network)
  python deploy/publish_to_hf.py               # uploads the data to a PRIVATE dataset repo, creates the Space, waits for it to start
  ```
  The Space downloads the data (about 3 GB) and the models (about 2 GB) on every cold start. Preferably create a read-only token and pass it as `SPACE_READ_TOKEN`. The Space folder was verified locally with its own `start.sh` (68/68 image checks, 12/12 edge cases); the Docker build itself runs on Hugging Face and was not run here.

### Keeping the Mac-hosted link running by itself (macOS)
```bash
python3 deploy/mac/install.py              # installs two login jobs: the app (kept awake with caffeinate) and the tunnel
python3 deploy/mac/install.py --uninstall  # stops and removes both
tail -f ~/Library/Logs/shop-the-look/*.log
```
- Both jobs start at login and are restarted if they crash (tested: app back in about 10 s, tunnel back in about 85 s).
- A quick tunnel gets a **new random address on every restart**. The tunnel job therefore publishes the current address to the `gh-pages` branch (`live.json`), and the stable page `.../live.html` redirects visitors there, or shows an offline page when the app does not answer. Use that stable link, not the trycloudflare address.
- After a restart the stable link can say "offline" for one to three minutes: GitHub Pages needs that long to serve the new address (an "errored" Pages build right after a restart is a harmless race when two pushes land seconds apart).
- Limits: it only runs while you are logged in and the Mac is awake (closing the lid sleeps the Mac; `caffeinate` only stops idle sleep), the Mac must stay online, and each restart makes one small commit on `gh-pages` (the helper clone lives in `~/.shop-the-look/ghpages`). A permanent address needs a hosted server (see Hugging Face above, which needs PRO) or a Cloudflare account with a domain for a named tunnel.
- When you rebuild the static preview, keep `live.json` and `live.html` on `gh-pages` (copy the new files over; do not delete the branch contents).

## 6. Static demo on GitHub Pages (preview only, no uploads)
GitHub Pages cannot run the models, so the hosted demo is static: `python app/build_demo_site.py` records the live server's answers for the six sample photos and packages the UI with a small shim (`demo.js`) that serves them. It is published from the `gh-pages` branch. Rebuild and republish after any change to the ranking or the UI. It contains photos from the datasets, so take it down (Settings, Pages) if that is not acceptable.
