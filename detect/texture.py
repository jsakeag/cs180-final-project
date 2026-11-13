"""Photo textures: the photos painted back onto the scene.

Floor and walls: every texel of the floor (top-down, metric) and of each wall (along x
height, metric) is projected into each registered photo with its recovered camera,
and sampled where the surface is actually visible: in frame, in front of the camera,
and not hidden behind furniture (the surface must be no farther than the depth map
says at that pixel). With several photos, the closest view wins (most pixels per m).
This is the homography rectification of proj4a run per texel, with a depth test.
Parts no photo saw are filled by tiling a clean patch taken from what was seen (mirror
tiling, so it's seamless), feathered into the seen part.

Furniture: each object is cut out of its detection box (GrabCut). Its silhouette goes
into a per-photo label image; the engine projects the photo onto the model from the
photo camera, only on surfaces that camera sees and only inside that silhouette (the
real pattern, e.g. a quilt). Everywhere else the model's own textures are shifted to
the object's colors (Reinhard color transfer in Lab: match mean and spread).

Writes engine/textures/<scene>/*.jpg, engine/models/photo/<scene>/<id>.glb and
scene["photo_textures"]; the engine shows them unless turned off in the Esc menu.

usage:
  python detect/texture.py engine/scenes/<scene>.json --detections <detections.json> --reg <registration.json>
"""
import argparse, io, json, math, pathlib, struct, sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
ENGINE = REPO / "engine"
sys.path.insert(0, str(ROOT))
from place import View  # noqa: E402

PX_PER_M = 80          # texture resolution
MAX_SIDE = 2048
PATCH_M = 0.5          # size of the clean patch used to fill unseen parts
FEATHER_M = 0.15
DEPTH_TOL = (0.08, 0.12)   # visible if surface depth <= map depth * (1 + a) + b
MIN_COS = 0.2          # skip surfaces seen at grazing angles (one pixel smears over meters)
STATIONARY_DIFF = 14   # a fill patch is tiled only if its quadrants agree this well (Lab-ish units)


# ------------------------------------------------------------------ projection

def project(view, P3):
    """(N, 3) points as (plan x, plan z, height) -> pixels (N, 2), camera depth (N,)."""
    W = np.c_[P3[:, 0], P3[:, 1], P3[:, 2] * view.pose["up_sign"]]
    Xc = W @ view.pose["R"].T + view.pose["t"]
    z = Xc[:, 2]
    uv = Xc @ view.K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def bilinear(img, uv):
    x, y = uv[:, 0], uv[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    I = img.astype(np.float32)
    return ((1 - fx) * (1 - fy) * I[y0, x0] + fx * (1 - fy) * I[y0, x0 + 1]
            + (1 - fx) * fy * I[y0 + 1, x0] + fx * fy * I[y0 + 1, x0 + 1])


def sample_views(views, images, P3, normal):
    """Color + visibility of 3D points from the best photo that sees each one.
    normal: (3,) surface normal as (plan x, plan z, up), either side."""
    n = len(P3)
    out = np.zeros((n, 3), np.float32)
    best = np.full(n, np.inf)
    for view, img in zip(views, images):
        uv, z = project(view, P3)
        h, w = img.shape[:2]
        C = np.array([view.cam_plan[0], view.cam_plan[1], view.pose["height"]])
        ray = C - P3
        cos = np.abs(ray @ np.asarray(normal, float)) / (np.linalg.norm(ray, axis=1) + 1e-9)
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1) & (cos > MIN_COS)
        ui = np.clip(uv[:, 0], 0, w - 1).astype(int)
        vi = np.clip(uv[:, 1], 0, h - 1).astype(int)
        if getattr(view, "furniture_mask", None) is not None:
            ok &= ~view.furniture_mask[vi, ui]      # detected furniture is not floor or wall
        if view.D is not None:
            Dh, Dw = view.D.shape
            dz = view.D[np.clip(vi * Dh // h, 0, Dh - 1), np.clip(ui * Dw // w, 0, Dw - 1)]
            ok &= z <= dz * (1 + DEPTH_TOL[0]) + DEPTH_TOL[1]
        take = ok & (z < best)
        if take.any():
            out[take] = bilinear(img, uv[take])
            best[take] = z[take]
    return out, np.isfinite(best)


# ------------------------------------------------------------------ fill unseen

def fill_unseen(tex, seen, patch_px):
    """Fill unseen texels by mirror-tiling the most typical fully-seen square patch,
    feathered into the seen part. Returns the texture and how much was seen."""
    H, W = seen.shape
    frac = float(seen.mean())
    if frac > 0.995:
        return tex
    if frac < 0.01:
        return np.full_like(tex, 160)
    p = int(min(patch_px, H, W))
    patch = None
    if p >= 8:
        integ = cv2.integral(seen.astype(np.uint8))
        full = (integ[p:, p:] - integ[:-p, p:] - integ[p:, :-p] + integ[:-p, :-p]) == p * p
        ys, xs = np.nonzero(full)
        if len(ys):
            med = np.median(tex[seen], 0)
            step = max(1, len(ys) // 400)
            cands = [(np.abs(tex[y:y + p, x:x + p].reshape(-1, 3).mean(0) - med).sum(), y, x)
                     for y, x in zip(ys[::step], xs[::step])]
            _, y, x = min(cands)
            patch = tex[y:y + p, x:x + p]
            q = p // 2
            quads = [patch[a:a + q, b:b + q].reshape(-1, 3).mean(0) for a in (0, q) for b in (0, q)]
            if max(np.abs(qa - qb).max() for qa in quads for qb in quads) > STATIONARY_DIFF:
                patch = None              # a rug / poster / furniture edge: tiling it looks awful
    if patch is None:                     # nothing big enough: flat median color
        fill = np.empty_like(tex)
        fill[:] = np.median(tex[seen], 0)
    else:
        tile = np.concatenate([patch, patch[:, ::-1]], 1)
        tile = np.concatenate([tile, tile[::-1]], 0)
        reps = (math.ceil(H / tile.shape[0]), math.ceil(W / tile.shape[1]), 1)
        fill = np.tile(tile, reps)[:H, :W]
    dist = cv2.distanceTransform(seen.astype(np.uint8), cv2.DIST_L2, 3)
    a = np.clip(dist / max(1.0, FEATHER_M * PX_PER_M), 0, 1)[..., None]
    return (a * tex + (1 - a) * fill).astype(np.float32)


# ------------------------------------------------------------------ surfaces

def floor_texture(scene, views, images):
    f = scene["schematic"]["floor"]
    (x0, z0), (x1, z1) = f["min"], f["max"]
    s = min(PX_PER_M, MAX_SIDE / max(x1 - x0, z1 - z0))
    W, H = max(2, int((x1 - x0) * s)), max(2, int((z1 - z0) * s))
    gx, gz = np.meshgrid(x0 + (np.arange(W) + 0.5) / s, z0 + (np.arange(H) + 0.5) / s)
    P3 = np.c_[gx.ravel(), gz.ravel(), np.zeros(gx.size)]
    col, seen = sample_views(views, images, P3, (0, 0, 1))
    tex, seen = col.reshape(H, W, 3), seen.reshape(H, W)
    tex = fill_unseen(tex, seen, PATCH_M * s)
    return tex, float(seen.mean()), [x0, z0], [x1, z1]


def wall_texture(wall, height, views, images):
    a, b = np.array(wall["from"], float), np.array(wall["to"], float)
    L = np.linalg.norm(b - a)
    t = (b - a) / L
    n = np.array([-t[1], t[0]]) * wall.get("thickness", 0.12) / 2
    s = min(PX_PER_M, MAX_SIDE / max(L, height))
    W, H = max(2, int(L * s)), max(2, int(height * s))
    gs, gy = np.meshgrid((np.arange(W) + 0.5) / s, height - (np.arange(H) + 0.5) / s)
    best = None
    for side in (n, -n):                 # sample the face that a camera actually sees
        P = a + side + gs.reshape(-1, 1) * t
        P3 = np.c_[P, gy.ravel()]
        col, seen = sample_views(views, images, P3, (side[0], side[1], 0) / np.linalg.norm(side))
        if best is None or seen.sum() > best[1].sum():
            best = (col, seen)
    tex, seen = best[0].reshape(H, W, 3), best[1].reshape(H, W)
    return tex, seen


# ------------------------------------------------------------------ furniture colors

def object_mask(img, box):
    """The object's silhouette inside its detection box (GrabCut), full-image bool mask."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    fg = np.zeros((h, w), bool)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return fg
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        pass
    if fg.sum() < 0.1 * (x2 - x1) * (y2 - y1):     # GrabCut gave up: use the box
        fg[:] = False
        fg[y1:y2, x1:x2] = True
    return fg


def object_lab_stats(img, fg):
    """Mean / std in Lab of the object's pixels. Lightness is exposure-normalized (photo
    median L -> 128): photos are often under-exposed, and the engine lights the model
    again, so raw photo lightness would make it doubly dark."""
    if fg.sum() < 50:
        return None
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    gain = float(np.clip(128 / max(1.0, np.median(LAB[..., 0])), 1.0, 2.5))
    lab = LAB[fg]
    lab[:, 0] *= gain
    return lab.mean(0), lab.std(0) + 1e-3


def _glb_parts(path):
    b = pathlib.Path(path).read_bytes()
    jlen = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + jlen])
    blen = struct.unpack_from("<I", b, 20 + jlen)[0]
    binb = b[20 + jlen + 8:20 + jlen + 8 + blen]
    views = [binb[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]] for v in j["bufferViews"]]
    return j, views


def _glb_write(path, j, views):
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
    pathlib.Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON")
                                   + js + struct.pack("<I4s", len(newbin), b"BIN\0") + bytes(newbin))


def recolor_glb(src, dst, target):
    """Reinhard color transfer of a .glb's base-color textures (and plain base-color
    factors) toward target = (mean, std) in Lab. The spread ratio is limited so a
    flat photo crop doesn't wash out the model's detail."""
    from PIL import Image
    j, views = _glb_parts(src)
    mats = j.get("materials", [])
    tex_imgs = {}
    for m in mats:
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bct is not None:
            src_i = j["textures"][bct["index"]].get("source")
            if src_i is not None and "bufferView" in j["images"][src_i]:
                tex_imgs[src_i] = None
    # model's own color statistics over all base-color textures
    decoded = {}
    for i in tex_imgs:
        im = Image.open(io.BytesIO(views[j["images"][i]["bufferView"]]))
        im.load()
        decoded[i] = im
    pix = [cv2.cvtColor(np.asarray(im.convert("RGB"))[..., ::-1], cv2.COLOR_BGR2LAB).reshape(-1, 3)[::7]
           for im in decoded.values()]
    for m in mats:
        f = m.get("pbrMetallicRoughness", {}).get("baseColorFactor")
        if f and m.get("pbrMetallicRoughness", {}).get("baseColorTexture") is None:
            rgb = np.uint8(np.clip(np.array(f[:3]) ** (1 / 2.2) * 255, 0, 255))[None, None, ::-1]
            pix.append(np.repeat(cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).reshape(-1, 3), 2000, 0))
    if not pix:
        return False
    allp = np.concatenate(pix).astype(np.float32)
    mu_m, sd_m = allp.mean(0), allp.std(0) + 1e-3
    mu_t, sd_t = target
    k = np.clip(sd_t / sd_m, 0.5, 1.5)
    k[0] = 1.0                            # keep the model's lightness detail
    mu = mu_t.copy()
    mu[0] = 0.5 * mu_m[0] + 0.5 * mu_t[0]   # color fully, lightness halfway (photo light is unreliable)

    def transfer(lab):
        out = (lab - mu_m) * k + mu
        return np.clip(out, 0, 255).astype(np.uint8)

    for i, im in decoded.items():
        alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
        bgr = np.asarray(im.convert("RGB"))[..., ::-1]
        lab = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
        rgb = cv2.cvtColor(transfer(lab), cv2.COLOR_LAB2BGR)[..., ::-1]
        out_im = Image.fromarray(np.ascontiguousarray(rgb))
        buf = io.BytesIO()
        if alpha is not None:
            out_im.putalpha(alpha)
            out_im.save(buf, "PNG")
            j["images"][i]["mimeType"] = "image/png"
        else:
            out_im.save(buf, "JPEG", quality=88)
            j["images"][i]["mimeType"] = "image/jpeg"
        views[j["images"][i]["bufferView"]] = buf.getvalue()
    for m in mats:
        pbr = m.get("pbrMetallicRoughness", {})
        f = pbr.get("baseColorFactor")
        if f and pbr.get("baseColorTexture") is None:
            rgb = np.uint8(np.clip(np.array(f[:3]) ** (1 / 2.2) * 255, 0, 255))[None, None, ::-1]
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
            new = cv2.cvtColor(transfer(lab), cv2.COLOR_LAB2BGR)[0, 0, ::-1] / 255.0
            pbr["baseColorFactor"] = [round(float(c) ** 2.2, 4) for c in new] + list(f[3:])
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _glb_write(dst, j, views)
    return True


# ------------------------------------------------------------------ main

def texture_scene(scene_path, detections, reg):
    scene_path = pathlib.Path(scene_path)
    scene = json.loads(scene_path.read_text())
    stem = scene_path.stem
    dets_by_img = {str(pathlib.Path(r["image"]).resolve()): r for r in json.loads(pathlib.Path(detections).read_text())}
    views, images = [], []
    for r in json.loads(pathlib.Path(reg).read_text()):
        key = str(pathlib.Path(r["image"]).resolve())
        if key in dets_by_img:
            v = View(r, dets_by_img[key])
            img = cv2.imread(str(REPO / r["image"]))
            v.furniture_mask = np.zeros(img.shape[:2], bool)
            v.labels = np.zeros(img.shape[:2], np.uint8)
            views.append(v)
            images.append(img)
    if not views:
        print("no registered photos: nothing to texture")
        return
    out_dir = ENGINE / "textures" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(ENGINE))  # noqa: E731
    pt = {}

    # each seen object's silhouette: masks it out of floor / walls, gives its colors, and
    # labels which photo pixels the engine may project onto its model
    view_of = {str((REPO / v.reg["image"]).resolve()): i for i, v in enumerate(views)}
    masks = {}
    label = 0
    for o in scene["objects"]:
        src = o.get("source", {})
        if src.get("placed_by") == "predicted" or not src.get("box") or not src.get("image"):
            continue
        vi = view_of.get(str((REPO / src["image"]).resolve()))
        if vi is None:
            continue
        fg = object_mask(images[vi], src["box"])
        masks[o["id"]] = (vi, fg)
        views[vi].furniture_mask |= cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if label < 254:
            label += 1
            views[vi].labels[fg & (views[vi].labels == 0)] = label
            o["photo_proj"] = {"camera": vi, "label": label}

    tex, frac, fmin, fmax = floor_texture(scene, views, images)
    cv2.imwrite(str(out_dir / "floor.jpg"), np.clip(tex, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    pt["floor"] = {"image": rel(out_dir / "floor.jpg"), "min": fmin, "max": fmax}
    print(f"floor: {frac:.0%} seen in photos")

    # walls: sample each, then fill unseen walls from all the wall paint that was seen
    height = scene["schematic"].get("wall_height", 2.6)
    raw = [wall_texture(w, height, views, images) for w in scene["schematic"].get("walls", [])]
    seen_pix = np.concatenate([t[s] for t, s in raw if s.any()] or [np.full((1, 3), 200.0)])
    paint = np.median(seen_pix, 0)
    pt["walls"] = {}
    for i, (t, s) in enumerate(raw):
        if s.mean() < 0.02:                # never seen: the room's paint color (not a copy of a
            # seen wall: that would duplicate its pictures and doorways)
            t = np.empty_like(t)
            t[:] = paint
        else:
            t = fill_unseen(t, s, PATCH_M * PX_PER_M)
        p = out_dir / f"wall_{i}.jpg"
        cv2.imwrite(str(p), np.clip(t, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        pt["walls"][str(i)] = rel(p)
    print(f"walls: {sum(s.mean() >= 0.02 for _, s in raw)}/{len(raw)} seen in photos")

    # photos + intrinsics + label images for projecting the photo onto the models
    cams = scene.get("cameras", [])
    for i, v in enumerate(views):
        photo, lab = out_dir / f"photo_{i}.jpg", out_dir / f"labels_{i}.png"
        cv2.imwrite(str(photo), images[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(lab), v.labels)
        for c in cams:
            if str((REPO / c.get("image", "")).resolve()) == str((REPO / v.reg["image"]).resolve()):
                c["K"] = [float(v.K[0, 0]), float(v.K[1, 1]), float(v.K[0, 2]), float(v.K[1, 2])]
                c["size"] = [int(images[i].shape[1]), int(images[i].shape[0])]
                c["photo"], c["labels"] = rel(photo), rel(lab)
                c["view_index"] = i
    for o in scene["objects"]:      # camera index in the photo_proj refers to scene cameras
        pp = o.get("photo_proj")
        if pp:
            idx = next((k for k, c in enumerate(cams) if c.get("view_index") == pp["camera"]), None)
            if idx is None:
                o.pop("photo_proj")
            else:
                pp["camera"] = idx

    # furniture: model colors -> photo colors
    n = 0
    for o in scene["objects"]:
        if not o.get("model", "").endswith(".glb") or o["id"] not in masks:
            continue
        vi, fg = masks[o["id"]]
        stats = object_lab_stats(images[vi], fg)
        if stats is None:
            continue
        dst = ENGINE / "models" / "photo" / stem / f"{o['id']}.glb"
        if recolor_glb(ENGINE / o["model"], dst, stats):
            o["model_photo"] = rel(dst)
            n += 1
    print(f"furniture recolored from the photo: {n}")
    scene["photo_textures"] = pt
    scene_path.write_text(json.dumps(scene, indent=2))
    print(f"updated {scene_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--reg", required=True)
    args = ap.parse_args()
    texture_scene(args.scene, args.detections, args.reg)


if __name__ == "__main__":
    main()
