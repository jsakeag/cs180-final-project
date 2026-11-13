"""Paint Berkeley: a photo of a building becomes that building's facade, and what's in
the facade reshapes the building.

1. Rectify. The facade's 4 corners clicked in the photo (TL, TR, BR, BL) map to a
   metric rectangle: wall length x building height. A homography (proj4a's compute_h)
   unwarps it into a flat facade texture. Clicks don't need to be precise.
2. Read the facade. Depth Anything V2 (outdoor) gives per-pixel depth; unwarped and
   high-passed it becomes the facade's relief map (columns out, windows in), which
   the engine uses to displace the facade into a 3D surface. Grounding DINO (open-vocabulary detector) finds columns, arches,
   windows, doors, pediments, balconies on the rectified facade; CLIP reads the column
   order (doric / ionic / corinthian), the overall style, and whether the roof showing
   above the facade is red clay tile.
3. Rules. Detections become shape rules the engine builds procedurally:
     >= 2 aligned columns   -> colonnade standing in front of the face (+ entablature)
     pediment               -> triangular pediment over the colonnade
     >= 3 similar arches    -> arcade: arch surrounds stepped out from the face
     red tile roof          -> hip roof in terracotta over the whole building
4. Paint. scene["paintings"] says which painting covers which wall of which building.
   Any painting can go on any wall ("imagine"): it repeats to fit the wall's length and
   stretches to its height, and its rules come along.

usage:
  python detect/paint.py add  --photo P --corners "x,y x,y x,y x,y" --building "Wheeler Hall" --facing S --name wheeler_south
  python detect/paint.py apply --painting wheeler_south --building "Dwinelle Hall" --facing E   # imagine
  python detect/paint.py list
"""
import argparse, json, math, pathlib, shutil, sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
ENGINE = REPO / "engine"
PAINTINGS = ENGINE / "paintings"
DEFAULT_SCENE = ENGINE / "scenes" / "campus_berkeley_core.json"
sys.path.insert(0, str(ROOT))
from homography import compute_h  # noqa: E402

PX_PER_M = 30
MAX_SIDE = 2048
DINO = "IDEA-Research/grounding-dino-tiny"
DEPTH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
RELIEF_RANGE = (-1.0, 1.0)    # m, stored in an 8-bit PNG (~0.8 cm steps); real facades rarely exceed this
RELIEF_PX_PER_M = 6           # relief grid resolution (the engine displaces a mesh at this density)
RELIEF_SIGMA_M = 2.0          # depth error is low-frequency: keep only structure finer than this
ELEMENTS = ["column", "arch", "window", "door", "pediment", "balcony"]
COMPASS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}     # engine: north = -z


# ------------------------------------------------------------------ geometry

def outward_normals(P):
    """Outward unit normal of each edge of a polygon on the (x, z) plane."""
    P = np.asarray(P, float)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    out = []
    for k in range(len(P)):
        e = P[(k + 1) % len(P)] - P[k]
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-9)
        out.append(n if area > 0 else -n)
    return out


def pick_edge(block, facing=None, edge=None, near=None):
    """An edge by index, by compass direction (longest edge facing roughly that way), or
    by a map click (edge nearest to the point)."""
    P = np.asarray(block["polygon"], float)
    if edge is not None:
        return int(edge)
    if near is not None:
        q = np.asarray(near, float)

        def dist(k):
            a, b = P[k], P[(k + 1) % len(P)]
            t = np.clip(np.dot(q - a, b - a) / (np.dot(b - a, b - a) + 1e-9), 0, 1)
            return np.linalg.norm(q - (a + t * (b - a)))
        return int(min(range(len(P)), key=dist))
    want = np.array(COMPASS[facing.upper()], float)
    normals = outward_normals(P)
    lens = [np.linalg.norm(P[(k + 1) % len(P)] - P[k]) for k in range(len(P))]
    cands = [k for k in range(len(P)) if normals[k] @ want > math.cos(math.radians(50))]
    if not cands:
        raise SystemExit(f"no wall of {block['name']} faces {facing}")
    return int(max(cands, key=lambda k: lens[k]))


def wall_run(block, k, tol_deg=12):
    """A wall is a run of consecutive polygon edges facing the same way (OSM often splits
    a straight facade at every porch or step). Returns (first_edge, last_edge, length)."""
    P = np.asarray(block["polygon"], float)
    n = len(P)
    N = outward_normals(P)
    same = lambda i: N[i % n] @ N[k] > math.cos(math.radians(tol_deg))  # noqa: E731
    first, last = k, k
    while same(first - 1) and (k - first) < n - 1:
        first -= 1
    while same(last + 1) and (last - first) < n - 1:
        last += 1
    a, b = P[first % n], P[(last + 1) % n]
    return first % n, last % n, float(np.linalg.norm(b - a))


def find_block(scene, name, near=None):
    blocks = scene["schematic"]["blocks"]
    if name in (None, "", "auto"):     # the building whose outline is nearest the map click
        if near is None:
            raise SystemExit("--building auto needs --near")
        q = np.asarray(near, float)

        def d(b):
            P = np.asarray(b["polygon"], float)
            best = np.inf
            for k in range(len(P)):
                a, c = P[k], P[(k + 1) % len(P)]
                t = np.clip(np.dot(q - a, c - a) / (np.dot(c - a, c - a) + 1e-9), 0, 1)
                best = min(best, np.linalg.norm(q - (a + t * (c - a))))
            return best
        i = min(range(len(blocks)), key=lambda i: d(blocks[i]))
        return i, blocks[i]
    if str(name).isdigit():
        return int(name), blocks[int(name)]
    hits = [(i, b) for i, b in enumerate(blocks) if b.get("name") and name.lower() in b["name"].lower()]
    if not hits:
        raise SystemExit(f"no building matching {name!r}")
    return max(hits, key=lambda ib: len(ib[1]["polygon"]))


def rect_aspect(src, w, h):
    """True width/height of the rectangle whose image is the quad `src` (TL TR BR BL).
    H maps the unit square to the image; its first two columns are the rectangle's axes
    as seen by the camera, K^-1 h1 and K^-1 h2. They must be orthogonal (one equation,
    which gives f), and their length ratio is the aspect. Near-frontal views make f
    unobservable; then a typical phone focal length (1.2 x width) is assumed."""
    Hs = compute_h(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), np.asarray(src, float))
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    a1, a2 = (T @ Hs)[:, 0], (T @ Hs)[:, 1]
    f2 = -(a1[0] * a2[0] + a1[1] * a2[1]) / (a1[2] * a2[2]) if abs(a1[2] * a2[2]) > 1e-12 else -1
    f = math.sqrt(f2) if 0.3 * w < math.sqrt(max(f2, 0)) < 4 * w else 1.2 * w
    k1 = np.array([a1[0] / f, a1[1] / f, a1[2]])
    k2 = np.array([a2[0] / f, a2[1] / f, a2[2]])
    return float(np.linalg.norm(k1) / np.linalg.norm(k2)), f, f2 > 0 and 0.3 * w < math.sqrt(max(f2, 0)) < 4 * w


# ------------------------------------------------------------------ reading the facade

class Models:
    _dino = _clip = None

    @classmethod
    def dino(cls):
        if cls._dino is None:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            cls._dino = (AutoProcessor.from_pretrained(DINO),
                         AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(dev).eval(), dev, torch)
        return cls._dino

    _depth = None

    @classmethod
    def depth(cls):
        if cls._depth is None:
            import torch
            from transformers import pipeline
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            cls._depth = pipeline("depth-estimation", model=DEPTH, device=dev)
        return cls._depth

    @classmethod
    def clip(cls):
        if cls._clip is None:
            from models_catalog import Clip
            cls._clip = Clip()
        return cls._clip


def detect_elements(facade_bgr, threshold=0.25):
    from PIL import Image
    proc, model, dev, torch = Models.dino()
    im = Image.fromarray(cv2.cvtColor(facade_bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=im, text=". ".join(ELEMENTS) + ".", return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=threshold, text_threshold=0.2,
                                                      target_sizes=[im.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = []
    for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
        cls = next((e for e in ELEMENTS if e in str(l)), None)
        if cls:
            dets.append({"class": cls, "score": round(float(s), 3), "box": [round(v, 1) for v in b]})
    return dets


def facade_relief(img, H, Wp, Hp, ppm):
    """The facade's 3D relief (m, + = sticks out) in rectified facade pixels.
    Monocular depth (Depth Anything V2, outdoor) is unwarped with the facade homography.
    At 20-40 m its absolute error is meters but smooth, while real relief (columns,
    window recesses, cornices) is < 1.5 m and sharp, so the depth is high-passed:
    relief = local mean (Gaussian, RELIEF_SIGMA_M) - depth. Vegetation and sky are
    masked out by color so a tree in front doesn't become a bump on the wall."""
    from PIL import Image
    D = np.array(Models.depth()(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))["predicted_depth"], np.float32)
    D = cv2.resize(D, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    R = cv2.warpPerspective(D, H, (Wp, Hp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    tex = cv2.warpPerspective(img, H, (Wp, Hp), borderMode=cv2.BORDER_REPLICATE)
    hsv = cv2.cvtColor(tex, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    veg = (hue > 25) & (hue < 95) & (sat > 50)
    sky = (hue > 95) & (hue < 130) & (sat > 60) & (val > 120)
    m = cv2.morphologyEx((veg | sky).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop specks
    mask = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
    w = (~mask).astype(np.float32)
    sig = RELIEF_SIGMA_M * ppm
    local = cv2.GaussianBlur(R * w, (0, 0), sig) / np.maximum(cv2.GaussianBlur(w, (0, 0), sig), 1e-3)
    rel = np.where(mask, 0.0, local - R)
    rel = cv2.GaussianBlur(rel, (0, 0), 0.6 * ppm / RELIEF_PX_PER_M)
    return np.clip(rel, *RELIEF_RANGE), float(mask.mean())


def clip_choice(images, options):
    """Zero-shot: which option text best fits (mean over images). Returns key, probs."""
    from PIL import Image
    clip = Models.clip()
    ims = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in images if i.size]
    if not ims:
        return None, {}
    e = clip.embed(ims).mean(0)
    e /= np.linalg.norm(e)
    keys = list(options)
    T = clip.embed_text([options[k] for k in keys])
    p = np.exp(100 * (T @ e))
    p /= p.sum()
    return keys[int(p.argmax())], {k: round(float(v), 3) for k, v in zip(keys, p)}


STYLES = {
    "beaux_arts": "a white stone beaux-arts classical building facade",
    "mission": "a plain stucco building with a red tile roof and rows of windows",
    "brutalist": "a brutalist raw concrete building",
    "modern_glass": "a modern glass and steel building",
    "brick": "a red brick building",
    "craftsman": "a wooden arts and crafts building",
}
ORDERS = {
    "doric": "a plain doric column with a simple round capital",
    "ionic": "an ionic column with scroll volutes on the capital",
    "corinthian": "a corinthian column with ornate acanthus leaf capital",
}
ROOFS = {
    "tile": "a red clay tile roof",
    "sky": "blue sky",
    "flat": "the flat top edge of a building against the sky",
}


def make_rules(dets, facade, W_m, H_m, ppm, roof_strip, facade_hi=None, ppm_hi=None):
    """Detections (pixels in the rectified facade) -> rules in facade meters
    (x from the left end of the wall, y up from the ground)."""
    rules, info = [], {}
    to_m = lambda b: (b[0] / ppm, H_m - b[3] / ppm, b[2] / ppm, H_m - b[1] / ppm)  # noqa: E731  x0,y0,x1,y1

    cols = [d for d in dets if d["class"] == "column"
            and (d["box"][3] - d["box"][1]) > 2.2 * (d["box"][2] - d["box"][0])]   # tall and narrow
    if len(cols) >= 2:
        M = np.array([to_m(d["box"]) for d in cols])
        base, top = float(np.median(M[:, 1])), float(np.median(M[:, 3]))
        keep = M[(np.abs(M[:, 1] - base) < 1.5) & (np.abs(M[:, 3] - top) < 1.5)]
        xs = sorted(((keep[:, 0] + keep[:, 2]) / 2).tolist())
        diam = float(np.median(keep[:, 2] - keep[:, 0])) if len(keep) else 1.0
        # a colonnade's columns stand a few diameters apart; two lamp posts 50 m apart are not one
        if len(xs) >= 2 and float(np.min(np.diff(xs))) <= 8 * diam:
            # columns stand at a regular spacing: extend the run across gaps the detector missed
            gaps = np.diff(xs)
            step = float(np.min(gaps)) if len(gaps) else 1.0
            filled = [xs[0]]
            for x in xs[1:]:
                while x - filled[-1] > 1.6 * step:
                    filled.append(filled[-1] + step)
                filled.append(x)
            radius = float(np.clip(np.median(keep[:, 2] - keep[:, 0]) / 2, 0.25, 1.2))
            # capitals, cut from a sharper rectification (CLIP can't read a 30 px capital)
            src, k = (facade_hi, ppm_hi / ppm) if facade_hi is not None else (facade, 1.0)
            crops = []
            for d in cols:
                x1, y1, x2, y2 = [v * k for v in d["box"]]
                h, w = y2 - y1, x2 - x1
                for frac in (0.15, 0.25):
                    crops.append(src[max(0, int(y1 - 0.05 * h)):int(y1 + frac * h),
                                     max(0, int(x1 - 0.2 * w)):int(x2 + 0.2 * w)])
            order, probs = clip_choice(crops, ORDERS)
            info["column_order"] = probs
            rules.append({"type": "colonnade", "x": [round(x, 2) for x in filled], "base": round(base, 2),
                          "top": round(top, 2), "radius": round(radius, 2), "order": order or "ionic",
                          "pediment": any(d["class"] == "pediment" for d in dets)})

    arches = [d for d in dets if d["class"] == "arch"]
    if len(arches) >= 3:
        M = np.array([to_m(d["box"]) for d in arches])
        w = M[:, 2] - M[:, 0]
        typical = float(np.median(w))
        same = M[np.abs(w - typical) < 0.35 * typical]
        if len(same) >= 3:
            rules.append({"type": "arcade", "arches": [[round(float(v), 2) for v in a] for a in same]})

    if roof_strip is not None and roof_strip.size:
        roof, probs = clip_choice([roof_strip], ROOFS)
        info["roof"] = probs
        if roof == "tile":
            rules.append({"type": "roof", "shape": "hip", "color": "terracotta"})
    return rules, info


# ------------------------------------------------------------------ commands

def add(args):
    scene_path = pathlib.Path(args.scene)
    scene = json.loads(scene_path.read_text())
    bi, block = find_block(scene, args.building, args.near)
    k0, k1, L = wall_run(block, pick_edge(block, args.facing, args.edge, args.near))
    s0, s1 = args.span if args.span else (0.0, L)
    img = cv2.imread(args.photo)
    if img is None:
        raise SystemExit(f"can't read {args.photo}")
    src = np.array([[float(v) for v in c.split(",")] for c in args.corners.split()], float)
    if src.shape != (4, 2):
        raise SystemExit("--corners needs 4 points: TL TR BR BL as x,y")
    # the facade's real proportions come from the photo, its height from the map; the
    # painting then repeats along the wall (a photo often shows only part of a facade)
    H_m = float(block["height"])
    aspect, f_est, f_ok = rect_aspect(src, img.shape[1], img.shape[0])
    W_m = float(np.clip(H_m * aspect, 2.0, max(2.0, s1 - s0)))
    print(f"facade aspect {aspect:.2f} (focal {'%.0f px from the quad' % f_est if f_ok else 'assumed'}) "
          f"-> {W_m:.1f} x {H_m:.1f} m, repeated along the {s1 - s0:.1f} m wall")
    ppm = min(PX_PER_M, MAX_SIDE / max(W_m, H_m))
    Wp, Hp = int(round(W_m * ppm)), int(round(H_m * ppm))
    dst = np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)
    H = compute_h(src, dst)
    facade = cv2.warpPerspective(img, H, (Wp, Hp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ppm_hi = min(80, 4096 / max(W_m, H_m))
    Wh, Hh = int(round(W_m * ppm_hi)), int(round(H_m * ppm_hi))
    H_hi = compute_h(src, np.array([[0, 0], [Wh, 0], [Wh, Hh], [0, Hh]], float))
    facade_hi = cv2.warpPerspective(img, H_hi, (Wh, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # the roof shows above the facade's top edge in the photo
    top_y = int(min(src[0, 1], src[1, 1]))
    h_quad = max(src[3, 1], src[2, 1]) - top_y
    x0, x1 = int(max(0, min(src[:, 0]))), int(min(img.shape[1], max(src[:, 0])))
    roof_strip = img[max(0, top_y - int(0.25 * h_quad)):max(1, top_y + int(0.03 * h_quad)), x0:x1]

    print(f"{block['name']} wall {k0}-{k1}: rectified {Wp}x{Hp} px")
    relief, masked = facade_relief(img, H, Wp, Hp, ppm)
    print(f"  relief from depth: p5 {np.percentile(relief, 5):+.2f} m, p95 {np.percentile(relief, 95):+.2f} m "
          f"({masked:.0%} masked as vegetation/sky)")
    dets = detect_elements(facade)
    by = {}
    for d in dets:
        by[d["class"]] = by.get(d["class"], 0) + 1
    print(f"  elements: {by}")
    style, style_p = clip_choice([facade], STYLES)
    rules, info = make_rules(dets, facade, W_m, H_m, ppm, roof_strip, facade_hi, ppm_hi)
    info["style"] = style_p
    print(f"  style: {style}  rules: {[r['type'] + (' (' + r['order'] + ')' if r['type'] == 'colonnade' else '') for r in rules] or 'none'}")

    name = args.name or f"{block['name'].split()[0].lower()}_{k0}"
    out = PAINTINGS / name
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "facade.jpg"), facade, [cv2.IMWRITE_JPEG_QUALITY, 88])
    rw, rh = max(2, int(round(W_m * RELIEF_PX_PER_M))), max(2, int(round(H_m * RELIEF_PX_PER_M)))
    rg = cv2.resize(relief, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32)
    rg = cv2.medianBlur(rg, 3)                     # 0.5 m median: speckle out, edges of real structure kept
    edge = int(round(1.0 * RELIEF_PX_PER_M))       # flatten toward the wall's ends so it meets the corners
    fade = np.minimum.outer(np.minimum(np.arange(rh), np.arange(rh)[::-1]), np.minimum(np.arange(rw), np.arange(rw)[::-1]))
    rg *= np.clip(fade / max(1, edge), 0, 1)
    r8 = (np.clip((rg - RELIEF_RANGE[0]) / (RELIEF_RANGE[1] - RELIEF_RANGE[0]), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "relief.png"), r8)
    vis = facade.copy()
    for d in dets:
        x1_, y1_, x2_, y2_ = map(int, d["box"])
        cv2.rectangle(vis, (x1_, y1_), (x2_, y2_), (0, 0, 255) if d["class"] == "column" else (255, 120, 0), 2)
        cv2.putText(vis, d["class"], (x1_, y1_ + 14), 0, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(out / "elements.jpg"), vis)
    painting = {"id": name, "building": block["name"], "size_m": [round(W_m, 2), round(H_m, 2)],
                "texture": f"paintings/{name}/facade.jpg", "style": style, "rules": rules,
                "relief": {"image": f"paintings/{name}/relief.png", "range": list(RELIEF_RANGE)},
                "elements": dets, "analysis": info,
                "source": {"photo": pathlib.Path(args.photo).name, "corners": src.tolist(), "credit": args.credit}}
    (out / "painting.json").write_text(json.dumps(painting, indent=2))
    apply_to(scene, name, bi, k0, [s0, s1], k1)
    scene_path.write_text(json.dumps(scene))
    print(f"  -> {out / 'painting.json'}; painted {block['name']} wall {k0}-{k1}")


def apply_to(scene, painting, bi, k, span, k_end=None):
    ps = [p for p in scene.get("paintings", []) if not (p["block"] == bi and p["edge"] == k)]
    ps.append({"painting": painting, "block": bi, "edge": k, "edge_end": k if k_end is None else k_end,
               "span": [round(span[0], 2), round(span[1], 2)]})
    scene["paintings"] = ps
    if not any("paintings" in c for c in scene.get("credits", [])):
        scene.setdefault("credits", []).append("Facade paintings: see engine/paintings/*/painting.json for photo credits")


def apply(args):
    scene_path = pathlib.Path(args.scene)
    scene = json.loads(scene_path.read_text())
    if not (PAINTINGS / args.painting / "painting.json").exists():
        raise SystemExit(f"no painting {args.painting}")
    bi, block = find_block(scene, args.building, args.near)
    k0, k1, L = wall_run(block, pick_edge(block, args.facing, args.edge, args.near))
    apply_to(scene, args.painting, bi, k0, args.span or [0.0, L], k1)
    scene_path.write_text(json.dumps(scene))
    print(f"painted {block['name']} wall {k0}-{k1} ({L:.1f} m) with {args.painting}")


def list_paintings(args):
    for p in sorted(PAINTINGS.glob("*/painting.json")):
        d = json.loads(p.read_text())
        print(f"{d['id']:<20} {d['building']:<28} {d['style']:<12} {[r['type'] for r in d['rules']]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("add", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--scene", default=str(DEFAULT_SCENE))
        p.add_argument("--building", default="auto", help="name substring, block index, or auto (nearest to --near)")
        p.add_argument("--facing", choices=list("NSEWnsew"))
        p.add_argument("--edge", type=int)
        p.add_argument("--near", type=float, nargs=2, metavar=("X", "Z"), help="map click near the wall")
        p.add_argument("--span", type=float, nargs=2, metavar=("S0", "S1"), help="part of the wall (m)")
        if name == "add":
            p.add_argument("--photo", required=True)
            p.add_argument("--corners", required=True, help='"x,y x,y x,y x,y" = TL TR BR BL in the photo')
            p.add_argument("--name")
            p.add_argument("--credit", default="")
        else:
            p.add_argument("--painting", required=True)
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd in ("add", "apply") and args.facing is None and args.edge is None and args.near is None:
        raise SystemExit("say which wall: --facing, --edge or --near")
    {"add": add, "apply": apply, "list": list_paintings}[args.cmd](args)


if __name__ == "__main__":
    main()
