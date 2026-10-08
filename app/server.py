"""FastAPI backend: POST /api/search (image) -> detected items + top catalog matches with explanations."""
import asyncio, collections, io, json, os, time, numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pipeline import Pipeline, analyze, ROOT
from families import ext_family_matrix
from embed import dev as DEVICE
from explain import Explainer, LABEL_FAM, FAMS, MIN_CONF, ATTRS, COLOR_SIM

# Fail early with a readable message if the data the server needs is not there (a fresh checkout has no data/ yet).
_need = ["ids.txt", f"emb_{os.environ.get('BACKEND', 'siglip')}_full.npy", "catalog"]
_missing = [n for n in _need if not os.path.exists(f"{ROOT}/data/{n}")]
if _missing:
    raise SystemExit(f"Missing data in {ROOT}/data: {', '.join(_missing)}. Build it (see README: fetch.py, embed.py) or copy the prepared data/ folder (see DEPLOY.md).")

BACKEND = os.environ.get("BACKEND", "siglip")
CFG = json.load(open(f"{ROOT}/data/config.json")) if os.path.exists(f"{ROOT}/data/config.json") else {}
EXACT_THR = CFG.get("exact_threshold", 0.88)
EXACT_THR_EXT = CFG.get("exact_threshold_ext", EXACT_THR)  # 95th pct of top-1 cosine on products known to be absent (final_eval.py)
BONUS = CFG.get("family_bonus", 0.1)
TOP_K = 5
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000  # reject absurd dimensions before decoding (decompression-bomb protection)
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "20"))  # searches per visitor per minute (loopback/local requests are exempt)
MAX_WAITING = int(os.environ.get("MAX_WAITING", "6"))      # requests allowed to queue for the model before we answer 503
_hits, _waiting = collections.defaultdict(collections.deque), 0
LOCK = asyncio.Lock()    # the models are not safe to run from two requests at once on one device

COLOR_BONUS = CFG.get("color_bonus", 0.0)  # tuned on validation (tune_rank.py)
MIN_SIM = CFG.get("min_sim", 0.0)           # matches below this cosine are hidden instead of shown as "closest"
DUP = 0.985                                 # two catalog photos this similar are the same product listed twice

P = Pipeline(BACKEND)
X = Explainer(P.E)
FULL = P.embs["full"]
PF = X.family_probs(FULL)  # garment-family probability per catalog product (zero-shot, 94% accurate on labelled products)
if len(P.ids) > P.n_base:  # extended products carry a real articleType: use it instead of the zero-shot guess
    M = ext_family_matrix(P.ids[P.n_base:])
    known = ~np.isnan(M[:, 0])
    PF[P.n_base:][known] = M[known]
PC = X.probs(FULL, "color")  # colour probability per product (zero-shot)
PCS = PC @ COLOR_SIM  # (N, 16): how close each product is to each colour name (red~pink, navy~blue...)
PS = X.probs(FULL, "sleeve")  # sleeve-length probability per product (zero-shot)
COLORS = ATTRS["color"]
SLEEVE_BONUS = CFG.get("sleeve_bonus", 0.0)  # short vs long sleeve agreement, only for tops/outerwear/dresses
HUB_BETA = CFG.get("hub_beta", 0.0)  # penalty for "hub" products that sit close to every query (tune_hub.py)
HUB = np.load(f"{ROOT}/data/hub.npy")[: len(P.ids)] if os.path.exists(f"{ROOT}/data/hub.npy") else np.zeros(len(P.ids), np.float32)
HUB = HUB - np.median(HUB[~P.block]); HUB[P.block] = 0  # centred; blocked products are excluded elsewhere
app = FastAPI(title="Shop the Look")


@app.middleware("http")
async def revalidate_ui(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith(("/catalog", "/ext", "/scenes")):
        resp.headers["Cache-Control"] = "no-cache"  # UI files: always revalidate, so an updated page is never stale in a demo
    return resp


@app.post("/api/search")
async def search(request: Request, file: UploadFile = File(...), extended: bool = Query(True)):
    global _waiting
    # visitor = the address the proxy/tunnel reports; requests straight from this machine (tests, local use) are not limited
    ip = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host)
    if ip not in ("127.0.0.1", "::1", "localhost"):
        now, q = time.time(), _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(429, f"Too many requests: please wait a moment (limit {RATE_PER_MIN} photos per minute).")
        q.append(now)
        if len(_hits) > 5000:  # forget visitors that have been quiet for a minute
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                _hits.pop(k, None)
    if _waiting >= MAX_WAITING:
        raise HTTPException(503, "The server is busy with other photos: please try again in a few seconds.")
    raw = await file.read(MAX_BYTES + 1)  # never buffer more than the limit, however large the upload
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "Image dimensions too large (max 50 megapixels)")
        img.draft("RGB", (2048, 2048))  # JPEG: decode at reduced size, saves memory on big photos
        img = ImageOps.exif_transpose(img).convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image dimensions too large (max 50 megapixels)")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "That file isn't a readable image")
    img.thumbnail((1024, 1024))
    _waiting += 1
    try:
        async with LOCK:
            return await asyncio.get_running_loop().run_in_executor(None, _search, img, extended)
    finally:
        _waiting -= 1


def _search(img, extended):
    t0 = time.time()
    dets, q = analyze(img, P.detect_raw(img), P.E, X)
    S = P.scores(q, extended) if len(dets) else []  # (n_items, n_products)
    thr = EXACT_THR_EXT if extended and len(P.ids) > P.n_base else EXACT_THR
    items = []
    for d, qv, s in zip(dets, q, S):
        qt = X.tags(qv)
        adj = s.copy()  # rank with family + colour bonuses, report/threshold on raw cosine
        fb = X.bonus_vec(PF, d["label"])
        if fb is not None:
            adj += BONUS * fb
        if qt["color"][1] >= 0.5:  # confident colour: results should be that colour (a red dress must not return a black one)
            adj += COLOR_BONUS * PCS[:, COLORS.index(qt["color"][0])]
        if qt["sleeve"][1] >= 0.5:  # confident short/long/sleeveless crop (garments only): prefer products with the same sleeve
            adj += SLEEVE_BONUS * PS[:, ATTRS["sleeve"].index(qt["sleeve"][0])]
        adj -= HUB_BETA * HUB
        # Displayed match = visual similarity minus a penalty when colour or sleeve disagree with the photographed item.
        # Order and percentage use this score, so a black dress can never be shown as the "closest" to a red one;
        # the raw cosine stays in `visual` and alone decides the "Exact match" badge.
        colsim = PCS[:, COLORS.index(qt["color"][0])] if qt["color"][1] >= 0.5 else np.ones(len(s), np.float32)
        slsim = PS[:, ATTRS["sleeve"].index(qt["sleeve"][0])] if qt["sleeve"][1] >= 0.5 else np.ones(len(s), np.float32)
        disp = s - COLOR_BONUS * (1 - colsim) - SLEEVE_BONUS * (1 - slsim)
        top = []
        order = np.argsort(-adj)[:200]  # best first; skip weak matches and the same product listed twice
        # colour-first: products of a clearly wrong colour only fill slots left over after all right-colour candidates
        for need_colour in (qt["color"][1] >= 0.6, False):
            for j in order:
                if len(top) == TOP_K:
                    break
                if j in top or s[j] < MIN_SIM or (need_colour and colsim[j] < 0.5) or any(FULL[j] @ FULL[k] > DUP for k in top):
                    continue
                top.append(j)
        top.sort(key=lambda j: -disp[j])
        matches = []
        for j in top:
            mt = X.tags(FULL[j])
            ext = j >= P.n_base
            matches.append(dict(id=P.ids[j], source="extended" if ext else "catalog", name=P.ext_meta[P.ids[j]]["productDisplayName"] if ext else None,
                                image=f"/{'ext' if ext else 'catalog'}/{P.ids[j]}.jpg", score=float(max(disp[j], 0.0)), visual=float(s[j]),
                                exact=bool(s[j] >= thr), tags={a: w for a, (w, c) in mt.items() if c >= MIN_CONF},
                                explanation=X.explain(qt, mt, float(disp[j]))))
        if matches:  # an item with no close match is not shown (no box, no row)
            items.append(dict(label=d["label"], confidence=d["score"], box=[round(v) for v in d["box"]],
                              tags={a: w for a, (w, c) in qt.items() if c >= MIN_CONF}, exact_found=any(m["exact"] for m in matches), matches=matches))
    return dict(width=img.width, height=img.height, items=items, elapsed=round(time.time() - t0, 2),
                reason=None if items else ("no_clothing" if not dets else "no_close_match"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)  # no icon; avoids a 404 in the browser console


@app.get("/api/health")
def health():
    return {"status": "ok", "products": len(P.ids), "device": DEVICE}


@app.get("/api/info")
def info():
    return dict(catalog=P.n_base, extended=len(P.ids) - P.n_base)


@app.get("/api/samples")
def samples():
    sc = sorted({json.loads(l)["scene"] for l in open(f"{ROOT}/dataset/validation.jsonl")})
    return [f"/scenes/{s}.jpg" for s in sc[:12]]


app.mount("/catalog", StaticFiles(directory=f"{ROOT}/data/catalog"), name="catalog")
if os.path.isdir(f"{ROOT}/data/extended/img"):
    app.mount("/ext", StaticFiles(directory=f"{ROOT}/data/extended/img"), name="ext")
app.mount("/scenes", StaticFiles(directory=f"{ROOT}/data/scenes"), name="scenes")
if os.path.isdir(f"{ROOT}/web"):
    app.mount("/", StaticFiles(directory=f"{ROOT}/web", html=True), name="web")
