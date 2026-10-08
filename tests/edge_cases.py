"""Hostile / unusual inputs against the LIVE API with the status code each must return. Exit 1 on any mismatch.
Generates its own files (including a 30000x30000 image and an 11 MB blob) in a temp dir."""
import os as _os
API = _os.environ.get('API', 'localhost:8000')  # target server host:port
import io, json, os, subprocess, sys, tempfile
from PIL import Image
import numpy as np

HERE = os.path.dirname(__file__)
tee = os.path.join(HERE, "img", "crop_tee.jpg")
if not os.path.exists(tee):  # not in the repo: fall back to any scene photo from the data bundle
    scenes = os.path.join(HERE, "..", "data", "scenes")
    tee = os.path.join(scenes, sorted(f for f in os.listdir(scenes) if f.endswith(".jpg"))[0])
d = tempfile.mkdtemp()
def w(name, data): open(os.path.join(d, name), "wb").write(data)
def img_bytes(im, fmt, **kw): b = io.BytesIO(); im.save(b, fmt, **kw); return b.getvalue()

cases = {  # name -> (bytes, expected HTTP status, expected `reason` or None)
    "empty.jpg": (b"", 400, None),
    "not_image.jpg": (b"this is not an image" * 100, 400, None),
    "truncated.jpg": (open(tee, "rb").read()[:3000], 400, None),
    "tiny_1x1.png": (img_bytes(Image.new("RGB", (1, 1), "red"), "PNG"), 200, "no_clothing"),
    "grey_noise.png": (img_bytes(Image.fromarray((np.random.rand(300, 200) * 255).astype("uint8")), "PNG"), 200, "no_clothing"),
    "rgba_noise.png": (img_bytes(Image.fromarray((np.random.rand(300, 200, 4) * 255).astype("uint8"), "RGBA"), "PNG"), 200, "no_clothing"),
    "panorama.png": (img_bytes(Image.new("RGB", (6000, 200), "white"), "PNG"), 200, "no_clothing"),
    "tee.webp": (img_bytes(Image.open(tee), "WEBP"), 200, None),
    "tee.gif": (img_bytes(Image.open(tee).convert("P"), "GIF"), 200, None),
    "tee.bmp": (img_bytes(Image.open(tee), "BMP"), 200, None),
    "over_10MB.bin": (os.urandom(11 * 1024 * 1024), 413, None),
    "huge_dimensions.png": (img_bytes(Image.new("RGB", (30000, 30000), "white"), "PNG"), 413, None),
}
bad = 0
for name, (data, status, reason) in cases.items():
    w(name, data)
    out = subprocess.run(["curl", "-s", "-m", "120", "-o", os.path.join(d, "out.json"), "-w", "%{http_code}", "-F", f"file=@{os.path.join(d, name)}", API + "/api/search"], capture_output=True, text=True).stdout
    body = open(os.path.join(d, "out.json"), encoding="utf-8").read()
    got_reason = json.loads(body).get("reason") if out == "200" else None
    ok = out == str(status) and (reason is None or got_reason == reason)
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name.ljust(22), out, got_reason or body[:60])
alive = subprocess.run(["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", API + "/api/info"], capture_output=True, text=True).stdout
print(f"\n{len(cases) - bad}/{len(cases)} passed; server still alive: {alive == '200'}")
sys.exit(1 if bad or alive != "200" else 0)
