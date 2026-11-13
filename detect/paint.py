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
        from room_prior import in_poly
        inside = [i for i, b in enumerate(blocks) if in_poly(q, np.asarray(b["polygon"], float))[0]]
        if inside:                           # a click inside a building means that building
            return inside[0], blocks[inside[0]]

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


def detect_tiled(facade_bgr, threshold=0.25):
    """Grounding DINO resizes its input to ~800 px, so on a long facade every window is a
    few pixels. Run it on overlapping tiles about 1.6x as wide as the facade is tall,
    then merge duplicates (same class, IoU > 0.5) across tiles."""
    Hh, Wh = facade_bgr.shape[:2]
    tw = int(min(Wh, 1.6 * Hh))
    xs = [0] if tw >= Wh else list(range(0, Wh - tw, int(tw * 0.75))) + [Wh - tw]
    dets = []
    for x0 in xs:
        for d in detect_elements(facade_bgr[:, x0:x0 + tw], threshold):
            b = d["box"]
            d["box"] = [b[0] + x0, b[1], b[2] + x0, b[3]]
            dets.append(d)
    dets.sort(key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(not (k["class"] == d["class"] and box_iou(k["box"], d["box"]) > 0.5) for k in keep):
            keep.append(d)
    return keep


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


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
    return np.clip(rel, *RELIEF_RANGE), mask


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


# ------------------------------------------------------------------ fitting architectural parts

def _hex(bgr):
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _cluster(vals, tol):
    """1-D clustering: sorted values split where the gap exceeds tol. Returns labels, centers."""
    order = np.argsort(vals)
    labels = np.zeros(len(vals), int)
    centers, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if vals[b] - vals[a] > tol:
            centers.append(float(np.median([vals[i] for i in cur])))
            cur = []
        cur.append(b)
        labels[b] = len(centers)
    centers.append(float(np.median([vals[i] for i in cur])))
    return labels, centers


def fit_parts(dets, tex, relief, mask, W_m, H_m, ppm_det, ppm):
    """Turn detections + depth into concrete parts, in facade meters (x from the left end,
    y up from the ground):
      openings  windows / doors / arched openings, snapped into a grid of rows and columns
                (facades repeat); grid cells the detector missed are filled where the photo
                is dark like glass; recess depth = how far the depth map sinks inside vs around
      columns   tall narrow detections; a regular run fills gaps; protrusion from the depth
      ledges    cornices / string courses: rows where the depth sticks out all along the wall
    tex / relief / mask are the rectified facade at `ppm` px/m; detections at `ppm_det`."""
    gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wall_gray = float(np.median(gray[~mask])) if (~mask).any() else float(np.median(gray))
    wall_color = np.median(tex[~mask], 0) if (~mask).any() else np.median(tex.reshape(-1, 3), 0)
    M = lambda b: (b[0] / ppm_det, H_m - b[3] / ppm_det, b[2] / ppm_det, H_m - b[1] / ppm_det)  # noqa: E731
    P = lambda x, y: (int(np.clip(x * ppm, 0, tex.shape[1] - 1)), int(np.clip((H_m - y) * ppm, 0, tex.shape[0] - 1)))  # noqa: E731

    def region(x0, y0, x1, y1, arr):
        (a, b), (c, d) = P(x0, y1), P(x1, y0)
        return arr[b:max(b + 1, d), a:max(a + 1, c)]

    # --- columns
    cols = []
    for d in dets:
        if d["class"] != "column":
            continue
        x0, y0, x1, y1 = M(d["box"])
        if (y1 - y0) > 2.2 * (x1 - x0) and (y1 - y0) > 0.25 * H_m:
            cols.append((x0, y0, x1, y1))
    columns = []
    if len(cols) >= 2:
        C = np.array(cols)
        base, top = float(np.median(C[:, 1])), float(np.median(C[:, 3]))
        C = C[(np.abs(C[:, 1] - base) < 1.5) & (np.abs(C[:, 3] - top) < 1.5)]
        diam = float(np.median(C[:, 2] - C[:, 0])) if len(C) else 0
        xs = sorted(((C[:, 0] + C[:, 2]) / 2).tolist())
        if len(xs) >= 2 and min(np.diff(xs)) <= 8 * diam:
            step = float(np.median(np.diff(xs)))
            run = [xs[0]]
            for x in xs[1:]:                    # fill a gap only if it's ~ a whole number of steps
                k = (x - run[-1]) / step
                if round(k) >= 2 and abs(k - round(k)) < 0.25:
                    run += [run[-1] + j * (x - run[-1]) / round(k) for j in range(1, int(round(k)))]
                run.append(x)
            band = lambda x: region(x - diam / 2, base, x + diam / 2, top, relief)  # noqa: E731
            vegb = lambda x: region(x - diam / 2, base, x + diam / 2, top, mask)  # noqa: E731
            strength = np.median([np.median(band(x)) for x in xs])
            win_boxes = [M(d["box"]) for d in dets if d["class"] in ("window", "door")]
            for direction in (-1, 1):           # extend the run while the photo still shows a column
                x = run[0] if direction < 0 else run[-1]
                while strength > 0 and 0.5 * diam < x + direction * step < W_m - 0.5 * diam:
                    x += direction * step
                    if (np.median(band(x)) < 0.5 * strength or vegb(x).mean() > 0.3
                            or any(b[0] < x < b[2] and b[1] < (base + top) / 2 < b[3] for b in win_boxes)):
                        break
                    run.insert(0, x) if direction < 0 else run.append(x)
            radius = float(np.clip(0.8 * diam / 2, 0.15, 0.8))   # boxes include the wider capital
            protrude = float(np.clip(np.median([np.median(band(x)) for x in run]), radius * 0.6, 2.5 * radius))
            col_color = np.median(np.concatenate([region(x - diam / 4, base, x + diam / 4, top, tex).reshape(-1, 3) for x in run]), 0)
            columns = [{"x": round(x, 2), "base": round(base, 2), "top": round(top, 2), "radius": round(radius, 2),
                        "protrude": round(protrude, 2), "color": _hex(col_color)} for x in run]

    # --- openings
    ops = []
    for d in dets:
        if d["class"] not in ("window", "door", "arch"):
            continue
        x0, y0, x1, y1 = M(d["box"])
        w, h = x1 - x0, y1 - y0
        if w < 0.4 or h < 0.5 or w * h > 0.25 * W_m * H_m or w > 0.4 * W_m:
            continue
        if any(abs(c["x"] - (x0 + x1) / 2) < 0.6 * c["radius"] and w < 2.5 * c["radius"] for c in columns):
            continue                            # the column itself, not a window
        ops.append({"kind": d["class"], "box": [x0, y0, x1, y1], "score": d["score"]})
    ops.sort(key=lambda o: -o["score"])
    kept = []
    for o in ops:                               # arches win over windows inside them
        if all(box_iou(o["box"], k["box"]) < 0.4 for k in kept):
            kept.append(o)
    openings = []
    if kept:
        B = np.array([o["box"] for o in kept])
        hgt = float(np.median(B[:, 3] - B[:, 1]))
        wid = float(np.median(B[:, 2] - B[:, 0]))
        rlab, rows = _cluster(((B[:, 1] + B[:, 3]) / 2).tolist(), 0.5 * hgt)
        clab, colsx = _cluster(((B[:, 0] + B[:, 2]) / 2).tolist(), 0.35 * wid)
        grid = {}
        for i, o in enumerate(kept):
            grid[(rlab[i], clab[i])] = o
        row_box = {r: (float(np.median([B[i, 1] for i in range(len(B)) if rlab[i] == r])),
                       float(np.median([B[i, 3] for i in range(len(B)) if rlab[i] == r])),
                       float(np.median([B[i, 2] - B[i, 0] for i in range(len(B)) if rlab[i] == r])),
                       max(set(kept[i]["kind"] for i in range(len(B)) if rlab[i] == r),
                           key=lambda k: sum(1 for i in range(len(B)) if rlab[i] == r and kept[i]["kind"] == k)))
                   for r in range(len(rows))}
        col_count = {c: int(np.sum(clab == c)) for c in range(len(colsx))}
        for r, (y0, y1, w, kind) in row_box.items():
            for c, xc in enumerate(colsx):
                x0, x1 = xc - w / 2, xc + w / 2
                inferred = (r, c) not in grid
                if inferred:
                    if col_count[c] < 2 or x0 < 0 or x1 > W_m:
                        continue
                    if any(abs(cc["x"] - xc) < cc["radius"] + w / 2 for cc in columns):
                        continue
                    inside = region(x0, y0, x1, y1, gray)
                    msk = region(x0, y0, x1, y1, mask)
                    if msk.mean() > 0.15 or np.median(inside) > 0.85 * wall_gray:
                        continue                # not dark like a window there: leave the wall
                rin = np.median(region(x0, y0, x1, y1, relief))
                ring = np.concatenate([region(x0 - 0.4, y0, x0, y1, relief).ravel(), region(x1, y0, x1 + 0.4, y1, relief).ravel(),
                                       region(x0, y1, x1, y1 + 0.4, relief).ravel()])
                depth = float(np.clip((np.median(ring) if ring.size else 0) - rin, 0.15, 0.6))
                openings.append({"kind": kind, "x0": round(x0, 2), "y0": round(y0, 2), "x1": round(x1, 2), "y1": round(y1, 2),
                                 "arch": kind == "arch", "depth": round(depth, 2), "inferred": inferred})

    # --- ledges: rows where the depth sticks out along most of the wall
    ledges = []
    prof = np.array([np.median(row[~m]) if (~m).sum() > 0.3 * len(row) else 0.0 for row, m in zip(relief, mask)])
    prof = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (0, 0), max(1.0, 0.15 * ppm)).ravel()
    for i in range(1, len(prof) - 1):
        if prof[i] >= 0.12 and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
            half = prof[i] / 2
            a, b = i, i
            while a > 0 and prof[a] > half:
                a -= 1
            while b < len(prof) - 1 and prof[b] > half:
                b += 1
            y = H_m - i / ppm
            t = float(np.clip((b - a) / ppm, 0.2, 1.2))
            if all(abs(l["y"] - y) > 1.0 for l in ledges):
                band = tex[a:b + 1].reshape(-1, 3)
                ledges.append({"y": round(y, 2), "thickness": round(t, 2), "protrude": round(float(np.clip(prof[i], 0.1, 0.8)), 2),
                               "color": _hex(np.median(band, 0))})
    ledges = sorted(ledges, key=lambda l: -l["protrude"])[:4]
    return {"columns": columns, "openings": openings, "ledges": ledges, "wall_color": _hex(wall_color)}


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
    H_m = float(block["height"]) if args.side == "out" else min(STOREY, float(block["height"]))
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
    relief, veg_mask = facade_relief(img, H, Wp, Hp, ppm)
    print(f"  relief from depth: p5 {np.percentile(relief, 5):+.2f} m, p95 {np.percentile(relief, 95):+.2f} m "
          f"({veg_mask.mean():.0%} masked as vegetation/sky)")
    dets_hi = detect_tiled(facade_hi)
    k_hi = ppm / ppm_hi
    dets = [{**d, "box": [round(v * k_hi, 1) for v in d["box"]]} for d in dets_hi]   # in facade (ppm) pixels
    by = {}
    for d in dets:
        by[d["class"]] = by.get(d["class"], 0) + 1
    print(f"  elements: {by}")
    style, style_p = clip_choice([facade], STYLES)
    rules, info = make_rules(dets, facade, W_m, H_m, ppm, roof_strip, facade_hi, ppm_hi)
    parts = fit_parts(dets, facade, relief, veg_mask, W_m, H_m, ppm, ppm)
    print(f"  parts: {len(parts['columns'])} columns, {len(parts['openings'])} openings "
          f"({sum(o['inferred'] for o in parts['openings'])} inferred from the grid), {len(parts['ledges'])} ledges")
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
    for o in parts["openings"]:
        a, b = int(o["x0"] * ppm), int((H_m - o["y1"]) * ppm)
        c, d = int(o["x1"] * ppm), int((H_m - o["y0"]) * ppm)
        cv2.rectangle(vis, (a, b), (c, d), (0, 200, 255) if o["inferred"] else (0, 255, 0), 1)
    for col in parts["columns"]:
        x = int(col["x"] * ppm)
        cv2.line(vis, (x, int((H_m - col["top"]) * ppm)), (x, int((H_m - col["base"]) * ppm)), (255, 0, 255), 2)
    for l in parts["ledges"]:
        y = int((H_m - l["y"]) * ppm)
        cv2.line(vis, (0, y), (vis.shape[1], y), (255, 255, 0), 1)
    cv2.imwrite(str(out / "elements.jpg"), vis)
    painting = {"id": name, "building": block["name"], "size_m": [round(W_m, 2), round(H_m, 2)],
                "texture": f"paintings/{name}/facade.jpg", "style": style, "rules": rules,
                "relief": {"image": f"paintings/{name}/relief.png", "range": list(RELIEF_RANGE)},
                "parts": parts,
                "elements": dets, "analysis": info,
                "source": {"photo": pathlib.Path(args.photo).name, "corners": src.tolist(), "credit": args.credit}}
    (out / "painting.json").write_text(json.dumps(painting, indent=2))
    apply_to(scene, name, bi, k0, [s0, s1], k1, args.side)
    update_doors(scene)
    scene_path.write_text(json.dumps(scene))
    print(f"  -> {out / 'painting.json'}; painted {block['name']} wall {k0}-{k1}")


DOOR_MIN_H, DOOR_MAX_SILL, DEFAULT_DOOR = 1.8, 0.8, (2.0, 2.8)   # m
STOREY = 4.5   # interior paintings cover the ground storey (the engine's STOREY)


def wall_frame_2d(block, k0, k1, span):
    """Same frame the engine uses (wallFrame): origin = the wall's left end seen from
    outside, x = viewer's right along the wall. Returns origin, x axis (plan x, z)."""
    P = np.asarray(block["polygon"], float)
    a, b = P[k0], P[(k1 + 1) % len(P)]
    t = (b - a) / np.linalg.norm(b - a)
    n = outward_normals(P)[k0]
    x = np.array([n[1], -n[0]])                  # up x n, on the plan
    origin = a + t * span[0] if x @ t > 0 else a + t * span[1]
    return origin, x


def update_doors(scene):
    """Doorways of every painted building, in world coordinates: ground-level doors and
    arches found in its paintings (scaled to the wall like the engine does), or one in
    the middle of a painted wall that showed none. Buildings with doors are enterable."""
    blocks = scene["schematic"]["blocks"]
    for b in blocks:
        b.pop("doors", None)
    for app in scene.get("paintings", []):
        if app.get("side") == "in":
            continue                             # interior paintings don't make doorways
        pj = PAINTINGS / app["painting"] / "painting.json"
        if not pj.exists():
            continue
        p = json.loads(pj.read_text())
        block = blocks[app["block"]]
        L = app["span"][1] - app["span"][0]
        Ws, Hs = p["size_m"]
        reps = max(1, round(L / Ws))
        sx, sy = L / (reps * Ws), block["height"] / Hs
        origin, xaxis = wall_frame_2d(block, app["edge"], app.get("edge_end", app["edge"]), app["span"])
        ranges = []
        for i in range(reps):
            for o in (p.get("parts") or {}).get("openings", []):
                y0, y1 = o["y0"] * sy, o["y1"] * sy
                if o["kind"] in ("door", "arch") and y0 <= DOOR_MAX_SILL and y1 - y0 >= DOOR_MIN_H:
                    x0, x1 = (o["x0"] + i * Ws) * sx, (o["x1"] + i * Ws) * sx
                    if x1 - x0 >= 1.0:
                        ranges.append((x0, x1, min(y1, 4.5)))
        if not ranges:
            w, h = DEFAULT_DOOR
            ranges = [(L / 2 - w / 2, L / 2 + w / 2, h)]
        doors = block.setdefault("doors", [])
        for x0, x1, h in ranges:
            doors.append({"from": (origin + xaxis * x0).round(3).tolist(), "to": (origin + xaxis * x1).round(3).tolist(),
                          "height": round(float(h), 2)})
    return sum(len(b.get("doors", [])) for b in blocks)


def apply_to(scene, painting, bi, k, span, k_end=None, side="out"):
    ps = [p for p in scene.get("paintings", [])
          if not (p["block"] == bi and p["edge"] == k and p.get("side", "out") == side)]
    ps.append({"painting": painting, "block": bi, "edge": k, "edge_end": k if k_end is None else k_end,
               "span": [round(span[0], 2), round(span[1], 2)], "side": side})
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
    apply_to(scene, args.painting, bi, k0, args.span or [0.0, L], k1, args.side)
    update_doors(scene)
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
        p.add_argument("--side", choices=["out", "in"], default="out", help="facade, or the wall's inside (ground storey)")
        if name == "add":
            p.add_argument("--photo", required=True)
            p.add_argument("--corners", required=True, help='"x,y x,y x,y x,y" = TL TR BR BL in the photo')
            p.add_argument("--name")
            p.add_argument("--credit", default="")
        else:
            p.add_argument("--painting", required=True)
    sub.add_parser("list")
    dp = sub.add_parser("doors", help="recompute doorways of painted buildings")
    dp.add_argument("--scene", default=str(DEFAULT_SCENE))
    args = ap.parse_args()
    if args.cmd in ("add", "apply") and args.facing is None and args.edge is None and args.near is None:
        raise SystemExit("say which wall: --facing, --edge or --near")
    if args.cmd == "doors":
        sp = pathlib.Path(args.scene)
        scene = json.loads(sp.read_text())
        print(f"{update_doors(scene)} doorways")
        sp.write_text(json.dumps(scene))
        return
    {"add": add, "apply": apply, "list": list_paintings}[args.cmd](args)


if __name__ == "__main__":
    main()
