"""Image embedders (FashionCLIP / Marqo-FashionSigLIP) + multi-view catalog embedding.
Usage: python embed.py <backend: fclip|siglip>  -> data/emb_<backend>_<view>.npy for views full/top/bottom/center"""
import os, sys, time, numpy as np, torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # DEVICE=cpu forces CPU
VIEWS = ["full", "top", "bottom", "center"]


class Embedder:
    def __init__(self, backend="siglip"):
        self.backend = backend
        if backend == "fclip":
            from transformers import CLIPModel, CLIPProcessor
            self.m = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(dev).eval()
            self.p = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        else:
            import open_clip
            self.m, _, self.pre = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
            self.m = self.m.to(dev).eval()
            self.tok = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    @torch.no_grad()
    def images(self, imgs):
        if self.backend == "fclip":
            f = self.m.get_image_features(pixel_values=self.p(images=imgs, return_tensors="pt")["pixel_values"].to(dev))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
        else:
            f = self.m.encode_image(torch.stack([self.pre(i) for i in imgs]).to(dev))
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    @torch.no_grad()
    def texts(self, txts):
        if self.backend == "fclip":
            t = self.p(text=txts, return_tensors="pt", padding=True).to(dev)
            f = self.m.get_text_features(**t)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
        else:
            f = self.m.encode_text(self.tok(txts).to(dev))
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()


def view(img, name):
    w, h = img.size
    return {"full": img, "top": img.crop((0, 0, w, int(h * .55))), "bottom": img.crop((0, int(h * .45), w, h)),
            "center": img.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))}[name]


if __name__ == "__main__":
    backend = sys.argv[1]
    E = Embedder(backend)
    ids = sorted(f[:-4] for f in os.listdir(f"{ROOT}/data/catalog") if f.endswith(".jpg"))
    open(f"{ROOT}/data/ids.txt", "w").write("\n".join(ids))
    outs = {v: [] for v in VIEWS}
    B, t0 = 128, time.time()
    for i in range(0, len(ids), B):
        imgs = [Image.open(f"{ROOT}/data/catalog/{p}.jpg").convert("RGB") for p in ids[i:i+B]]
        for v in VIEWS:
            outs[v].append(E.images([view(im, v) for im in imgs]))
        if (i // B) % 20 == 0:
            print(f"{i}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    for v in VIEWS:
        np.save(f"{ROOT}/data/emb_{backend}_{v}.npy", np.concatenate(outs[v]))
    print("DONE", time.time() - t0, flush=True)
