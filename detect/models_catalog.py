"""Catalog of real 3D models per class + CLIP retrieval: detected crop -> best model.

Sources
  ABO  (Amazon Berkeley Objects, CC BY 4.0): ~8k real furniture products with metric
       GLB models and catalog photos. https://amazon-berkeley-objects.s3.amazonaws.com
  Poly Haven (CC0): used for plants (ABO only has empty planters).

usage:
  python detect/models_catalog.py build            # once: metadata -> thumbs -> CLIP embeddings
  python detect/models_catalog.py match engine/scenes/bedroom.json   # sets "model" on each object
"""
import argparse, csv, gzip, glob, json, pathlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from priors import SIZE_PRIORS, MATCH_GROUPS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data"
ABO = DATA / "abo"
THUMBS = DATA / "thumbs"
CATALOG = DATA / "catalog.json"
EMB = DATA / "catalog_clip.npy"
ENGINE = REPO / "engine"
MODELS_DIR = ENGINE / "models"

ABO_S3 = "https://amazon-berkeley-objects.s3.amazonaws.com"
CLIP_MODEL = "openai/clip-vit-base-patch32"
UA = {"User-Agent": "cs180-scene-builder/0.1"}
REFIT_RANGE = (0.7, 1.4)
MAX_PRIOR_MISMATCH = 1.0   # mean |log size ratio| vs class prior (~2.7x off on average)


# class -> list of (ABO product types, name must contain any of, name must not contain any of)
ABO_RULES = {
    "bed": [(["BED", "BED_FRAME"], None, ["headboard"])],
    "chair": [(["CHAIR"], None, ["recliner", "accent", "arm", "club", "office", "gaming", "rocking"])],
    "armchair": [(["CHAIR", "FURNITURE_COVER"], ["accent", "arm", "club", "recliner"], ["slipcover"])],
    "sofa": [(["SOFA", "HOME_FURNITURE_AND_DECOR"], ["sofa", "loveseat", "sectional", "couch"], ["slipcover", "cover"])],
    "table": [(["TABLE"], ["dining", "kitchen"], None)],
    "coffee_table": [(["TABLE"], ["coffee", "cocktail"], None)],
    "nightstand": [(["TABLE", "DRESSER", "CABINET"], ["night", "bedside"], None)],
    "desk": [(["DESK", "TABLE"], ["desk"], None), (["DESK"], None, None)],
    "bookshelf": [(["SHELF", "HOME_FURNITURE_AND_DECOR", "CABINET", "FURNITURE"],
                   ["bookcase", "bookshelf", "shelving", "etagere", "ladder shelf", "shelf unit"],
                   ["floating", "wall", "corner shelf"])],
    "lamp": [(["LAMP"], ["floor"], None), (["LAMP"], None, ["table lamp", "desk lamp"])],
    "cabinet": [(["CABINET"], None, ["night", "bedside", "wardrobe", "tv"])],
    "dresser": [(["DRESSER", "STORAGE_BOX", "CABINET"], ["dresser", "chest of drawers", "drawer chest"], ["bedside", "night"])],
    "wardrobe": [(["CABINET", "HOME_FURNITURE_AND_DECOR", "FURNITURE"], ["wardrobe", "armoire"], None)],
    "tv_stand": [(["FLAT_SCREEN_DISPLAY_MOUNT", "CABINET", "TABLE"], ["tv", "media", "entertainment"], None)],
    "stool": [(["STOOL_SEATING", "OTTOMAN"], None, None)],
    "bench": [(["BENCH"], None, None)],
    "basket": [(["HOME", "STORAGE_BOX", "HOME_FURNITURE_AND_DECOR"], ["basket"], None)],
}


def _en_name(d):
    for n in d.get("item_name", []):
        if n.get("language_tag", "").startswith("en"):
            return n["value"]
    return (d.get("item_name") or [{"value": ""}])[0]["value"]


def _match(name, ptype, rule):
    types, inc, exc = rule
    n = name.lower()
    return (ptype in types and (not inc or any(k in n for k in inc))
            and not (exc and any(k in n for k in exc)))


def build_abo():
    models = {r["3dmodel_id"]: r for r in csv.DictReader(gzip.open(ABO / "3dmodels.csv.gz", "rt"))}
    images = {r["image_id"]: r["path"] for r in csv.DictReader(gzip.open(ABO / "images.csv.gz", "rt"))}
    listings = {}
    for f in glob.glob(str(ABO / "listings/metadata/*.json.gz")):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            mid = d.get("3dmodel_id")
            if mid in models and mid not in listings and d.get("main_image_id") in images:
                listings[mid] = d
    out = []
    for cls, rules in ABO_RULES.items():
        for rule in rules:  # first rule with results wins (later ones are fallbacks)
            hits = [(mid, d) for mid, d in listings.items()
                    if _match(_en_name(d), d["product_type"][0]["value"], rule)]
            if hits:
                break
        for mid, d in hits:
            m = models[mid]
            out.append({
                "source": "abo", "id": mid, "class": cls, "name": _en_name(d)[:120],
                "extent": [float(m["extent_x"]), float(m["extent_y"]), float(m["extent_z"])],
                "glb_url": f"{ABO_S3}/3dmodels/original/{m['path']}",
                "thumb_url": f"{ABO_S3}/images/small/{images[d['main_image_id']]}",
                "license": "CC BY 4.0 (Amazon Berkeley Objects)",
            })
        print(f"  abo {cls:<13} {len(hits)}")
    return out


def build_polyhaven():
    req = urllib.request.Request("https://api.polyhaven.com/assets?t=models", headers=UA)
    assets = json.load(urllib.request.urlopen(req))
    out = []
    for aid, a in assets.items():
        if "plants" not in a.get("categories", []):
            continue
        dims = a.get("dimensions")  # mm, [x, y(?), z]
        out.append({
            "source": "polyhaven", "id": aid, "class": "plant", "name": a.get("name", aid),
            "extent": [dims[0] / 1000, dims[2] / 1000, dims[1] / 1000] if dims and len(dims) == 3 else None,
            "glb_url": None,  # resolved at download time via /files
            "thumb_url": f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=256&height=256",
            "license": "CC0 (Poly Haven)",
        })
    print(f"  polyhaven plant {len(out)}")
    return out


def _fetch(url, path):
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers=UA)
        path.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    fetch failed {url}: {e}")
        return False


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def embed(self, pil_images, batch=64):
        outs = []
        for i in range(0, len(pil_images), batch):
            inp = self.proc(images=pil_images[i:i + batch], return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                e = self.model.get_image_features(**inp)
            if not isinstance(e, self.torch.Tensor):  # transformers >= 5 returns an output object
                e = e.pooler_output
            outs.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 512), np.float32)

    def embed_text(self, texts):
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True).to(self.dev)
        with self.torch.no_grad():
            e = self.model.get_text_features(**inp)
        if not isinstance(e, self.torch.Tensor):
            e = e.pooler_output
        return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def _open_rgb(path):
    """Thumbnails with transparency (Poly Haven) go on white, like catalog photos."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def build():
    THUMBS.mkdir(parents=True, exist_ok=True)
    print("collecting candidates")
    cat = build_abo() + build_polyhaven()
    print(f"downloading {len(cat)} thumbnails")
    paths = [THUMBS / f"{c['source']}_{c['id']}.{'png' if c['source'] == 'polyhaven' else 'jpg'}" for c in cat]
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda a: _fetch(*a), [(c["thumb_url"], p) for c, p in zip(cat, paths)]))
    cat = [dict(c, thumb=str(p.relative_to(DATA))) for c, p, k in zip(cat, paths, ok) if k]
    print(f"embedding {len(cat)} thumbnails with CLIP")
    E = Clip().embed([_open_rgb(DATA / c["thumb"]) for c in cat])
    CATALOG.write_text(json.dumps(cat))
    np.save(EMB, E.astype(np.float32))
    print(f"wrote {CATALOG} and {EMB}")


# ------------------------------------------------------------------ matching

def shrink_glb(path, max_side=1024, min_bytes=6_000_000):
    """Re-encode a .glb's embedded textures (ABO ships 4K PNGs, 25-45 MB per model)
    down to max_side JPEG (PNG kept only where alpha is used). Geometry is untouched."""
    import io, struct
    from PIL import Image
    path = pathlib.Path(path)
    b = path.read_bytes()
    if len(b) < min_bytes:
        return
    jlen = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + jlen])
    bin_off = 20 + jlen + 8
    blen = struct.unpack_from("<I", b, 20 + jlen)[0]
    binb = b[bin_off:bin_off + blen]
    views = [binb[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]] for v in j["bufferViews"]]
    for img in j.get("images", []):
        if "bufferView" not in img:
            continue
        im = Image.open(io.BytesIO(views[img["bufferView"]]))
        im.load()
        alpha = im.mode in ("RGBA", "LA", "PA") and im.getchannel("A").getextrema()[0] < 255
        im = im.convert("RGBA" if alpha else "RGB")
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        if alpha:
            im.save(out, "PNG", optimize=True); img["mimeType"] = "image/png"
        else:
            im.save(out, "JPEG", quality=88); img["mimeType"] = "image/jpeg"
        views[img["bufferView"]] = out.getvalue()
    newbin = bytearray()
    for v, data in zip(j["bufferViews"], views):
        newbin += b"\0" * (-len(newbin) % 4)
        v["byteOffset"], v["byteLength"] = len(newbin), len(data)
        newbin += data
    newbin += b"\0" * (-len(newbin) % 4)
    j["buffers"][0]["byteLength"] = len(newbin)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(newbin)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                     + struct.pack("<I4s", len(newbin), b"BIN\0") + bytes(newbin))


def download_model(c):
    """Returns the model path relative to engine/, downloading if needed."""
    if c["source"] == "abo":
        rel = pathlib.Path("models/abo") / f"{c['id']}.glb"
        dst = ENGINE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not _fetch(c["glb_url"], dst):
            return None
        shrink_glb(dst)
        return str(rel)
    # Poly Haven: glTF with separate bin + textures
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{c['id']}", headers=UA)
    files = json.load(urllib.request.urlopen(req))
    g = files["gltf"]
    res = "1k" if "1k" in g else sorted(g)[0]
    entry = g[res]["gltf"]
    base = ENGINE / "models/polyhaven" / c["id"]
    base.mkdir(parents=True, exist_ok=True)
    gltf_name = pathlib.Path(entry["url"].split("?")[0]).name
    ok = _fetch(entry["url"], base / gltf_name)
    for relpath, info in entry.get("include", {}).items():
        p = base / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        ok &= _fetch(info["url"], p)
    return str((base / gltf_name).relative_to(ENGINE)) if ok else None


def _gltf_json(path):
    path = pathlib.Path(path)
    if path.suffix == ".glb":
        import struct
        b = path.read_bytes()
        n, _ = struct.unpack_from("<I4s", b, 12)
        return json.loads(b[20:20 + n])
    return json.loads(path.read_text())


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    T = np.eye(4)
    if "scale" in n:
        T = np.diag([*n["scale"], 1.0]) @ T
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4); M[:3, :3] = R
        T = M @ T
    if "translation" in n:
        M = np.eye(4); M[:3, 3] = n["translation"]
        T = M @ T
    return T


def gltf_extent(path):
    """True [x, y, z] size of a .glb/.gltf in its own frame (what three.js Box3 sees),
    from accessor min/max pushed through the node transforms."""
    g = _gltf_json(path)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def visit(i, parent):
        nonlocal lo, hi
        n = g["nodes"][i]
        M = parent @ _node_matrix(n)
        if "mesh" in n:
            for prim in g["meshes"][n["mesh"]]["primitives"]:
                acc = g["accessors"][prim["attributes"]["POSITION"]]
                a, b = np.array(acc["min"]), np.array(acc["max"])
                corners = np.array([[x, y, z, 1] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
                w = (M @ corners.T).T[:, :3]
                lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        for c in n.get("children", []):
            visit(c, M)

    for i in g["scenes"][g.get("scene", 0)]["nodes"]:
        visit(i, np.eye(4))
    return (hi - lo).tolist()


def size_score(extent, size):
    """How well a model's real dimensions match the estimated box (0 = perfect).
    Compares log ratios so 2x too big and 2x too small cost the same; footprint
    axes may be swapped (we don't know which side is the front for sure)."""
    if not extent:
        return 0.5
    e, s = np.log(np.maximum(extent, 1e-3)), np.log(np.maximum(size, 1e-3))
    a = np.abs(e - s).mean()
    b = np.abs(e[[2, 1, 0]] - s).mean()
    return float(min(a, b))


def match(scene_path, top_k=5, size_weight=0.2, refit=True, missing_only=False):
    from PIL import Image
    scene_path = pathlib.Path(scene_path)
    scene = json.loads(scene_path.read_text())
    cat = json.loads(CATALOG.read_text())
    E = np.load(EMB)
    by_class = {}
    for i, c in enumerate(cat):
        by_class.setdefault(c["class"], []).append(i)

    clip = Clip()
    credits = {}
    style = None   # mean crop embedding of the seen objects: predicted ones should look like they belong
    seen_crops = []
    for o in scene["objects"]:
        src = o.get("source", {})
        if src.get("box") and src.get("image"):
            img = Image.open(REPO / src["image"]).convert("RGB")
            seen_crops.append(img.crop(tuple(src["box"])))
    if seen_crops:
        style = clip.embed(seen_crops).mean(0)
        style /= np.linalg.norm(style)
    text_cache = {}
    for o in scene["objects"]:
        if missing_only and o.get("model"):
            continue
        group = next((g for g in MATCH_GROUPS if o["class"] in g), (o["class"],))
        idx = [i for cls in group for i in by_class.get(cls, [])]
        src = o.get("source", {})
        if not idx:
            continue
        if src.get("box") and src.get("image"):
            img = Image.open(REPO / src["image"]).convert("RGB")
            x1, y1, x2, y2 = src["box"]
            pad = 0.05 * max(x2 - x1, y2 - y1)
            crop = img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad)))
            q = clip.embed([crop])[0]
        else:  # predicted (never photographed): match on the class name + style of what was seen
            kind = "table " if src.get("on_top_of") else ""      # a lamp on a nightstand is a table lamp
            text = f"a photo of a {kind}{o['class'].replace('_', ' ')}"
            if src.get("room_type"):        # furnishing a room: what such a room would have
                text = f"a {o['class'].replace('_', ' ')} for a {src['room_type'].replace('_', ' ')}"
            if text not in text_cache:
                text_cache[text] = clip.embed_text([text])[0]
            q = text_cache[text]
            if style is not None:
                q = q / np.linalg.norm(q) + 0.5 * style
                q /= np.linalg.norm(q)
        # drop candidates whose real size is far from anything plausible for the class
        priors = [SIZE_PRIORS[g] for g in group if g in SIZE_PRIORS]
        if priors:
            idx = [i for i in idx
                   if min(size_score(cat[i]["extent"], p) for p in priors) < MAX_PRIOR_MISMATCH] or idx
        sims = E[idx] @ q
        scores = sims - size_weight * np.array([size_score(cat[i]["extent"], o["size"]) for i in idx])
        order = np.argsort(-scores)
        chosen = None
        for j in order[:top_k]:  # fall through if a download fails
            c = cat[idx[j]]
            path = download_model(c)
            if path:
                chosen = (c, float(sims[j]))
                break
        if not chosen:
            print(f"  {o['id']}: no model downloaded")
            continue
        c, sim = chosen
        o["model"] = path
        o["model_info"] = {"source": c["source"], "id": c["id"], "name": c["name"],
                           "clip_sim": round(sim, 3), "license": c["license"]}
        credits[c["id"]] = f"{c['name']} ({c['license']})"
        if refit:
            # Real products have real sizes: keep the model's true proportions and scale
            # only a little (0.7-1.4x) toward our estimate (geometric mean of the width and
            # height ratios; depth is the least observed). Physics box = model box.
            ext = np.array(gltf_extent(ENGINE / path))
            r = np.array(o["size"]) / ext
            k = float(np.clip(np.sqrt(r[0] * r[1]), *REFIT_RANGE))
            if c["source"] == "polyhaven":  # no trustworthy real size: fit the estimate
                k = float(np.min(np.array(o["size"]) / ext))
            o["size"] = [round(float(v), 3) for v in ext * k]
            o["model_info"]["scale_vs_real"] = round(k, 3)
        print(f"  {o['id']:<12} -> {c['source']}:{c['id']} sim {sim:.2f}  {c['name'][:60]}")
    scene["credits"] = sorted(set(scene.get("credits", [])) | set(credits.values()))   # keep map/photo credits
    scene_path.write_text(json.dumps(scene, indent=2))
    print(f"updated {scene_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("shrink", help="shrink textures of already-downloaded ABO models")
    m = sub.add_parser("match")
    m.add_argument("scene")
    m.add_argument("--no-refit", action="store_true")
    m.add_argument("--missing", action="store_true", help="only objects without a model yet")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "shrink":
        for p in sorted((MODELS_DIR / "abo").glob("*.glb")):
            before = p.stat().st_size
            shrink_glb(p)
            print(f"{p.name}: {before / 1e6:.1f} -> {p.stat().st_size / 1e6:.1f} MB")
    else:
        match(args.scene, refit=not args.no_refit, missing_only=args.missing)


if __name__ == "__main__":
    sys.exit(main())
