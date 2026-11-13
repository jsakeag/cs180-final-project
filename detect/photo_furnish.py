"""Furnish from a photo, no clicks: a photo of a real room is turned into furniture
placed around where you stand in the game.

1. detect.py --depth gives the furniture boxes and metric depth (Depth Anything V2 indoor).
2. Floor plane by RANSAC on the depth: pixels in the lower part of the image outside
   furniture boxes are back-projected (X = z K^-1 [u v 1]) and a plane is fitted. Its
   normal gives the camera's tilt, its distance the camera height. (Fallback: level
   camera at 1.4 m.)
3. Each object's floor contact = its box's bottom-center ray intersected with that plane,
   the same geometry as the floor homography but from depth instead of clicks. If the
   bottom is cut off by the frame, its depth at the box center is used instead. Width is
   measured (box width x distance / f), clamped around the class prior; it faces the camera.
4. The photo is assumed taken from where the player stands, facing where they look, so
   (right, forward) on the floor map into the world with the player's position and yaw.
   Objects outside the building are dropped; interior furniture they land on is replaced.

usage:
  python detect/photo_furnish.py --detections D.json --scene S.json --block 6 --at X Z --yaw RAD
"""
import argparse, json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from priors import SIZE_PRIORS, FLOOR_CLASSES, CLAMP  # noqa: E402
from room_prior import in_poly  # noqa: E402


def fit_floor(D, f, cx, cy, boxes, iters=400, seed=0):
    """RANSAC floor plane in camera coords (x right, y down, z forward).
    Returns unit normal n (pointing up, toward the camera side) and height h (m)."""
    H, W = D.shape
    vs, us = np.mgrid[int(0.5 * H):H:4, 0:W:4]
    keep = np.ones(vs.shape, bool)
    for x1, y1, x2, y2 in boxes:
        keep &= ~((us >= x1) & (us <= x2) & (vs >= y1) & (vs <= y2))
    us, vs = us[keep], vs[keep]
    z = D[vs, us]
    ok = z > 0.2
    P = np.c_[(us[ok] - cx) / f * z[ok], (vs[ok] - cy) / f * z[ok], z[ok]]
    if len(P) < 50:
        return np.array([0, -1.0, 0]), 1.4, 0.0
    rng = np.random.default_rng(seed)
    thr = 0.03 * float(np.median(P[:, 2]))
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.7:                      # a floor is roughly horizontal in the camera
            continue
        d = -n @ a
        inl = np.abs(P @ n + d) < thr
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None or best[0] < 0.15 * len(P):
        return np.array([0, -1.0, 0]), 1.4, 0.0
    Q = P[best[1]]                               # refit on inliers (least squares via SVD)
    c0 = Q.mean(0)
    n = np.linalg.svd(Q - c0)[2][-1]
    if n[1] > 0:
        n = -n                                   # point up (camera y is down)
    h = float(-(n @ c0))                         # camera's height above the plane
    return n, h, float(best[0] / len(P))


def place(entry, n, h, f, cx, cy, D, min_conf=0.3):
    """Detections -> (right, forward) floor positions in the photo's frame + sizes."""
    up = n
    fwd = np.array([0, 0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    W, H = entry["width"], entry["height"]
    out = []
    for d in entry["detections"]:
        cls = d["class"]
        if cls not in FLOOR_CLASSES or d["conf"] < min_conf:
            continue
        x1, y1, x2, y2 = d["box"]
        if y2 < H - 3:                           # floor contact visible: ray x floor plane
            ray = np.array([((x1 + x2) / 2 - cx) / f, (y2 - cy) / f, 1.0])
            denom = n @ ray
            if denom >= -1e-6:
                continue                         # ray never reaches the floor (above the horizon)
            t = h / -denom
            X = t * ray
            how = "floor plane"
        else:                                    # bottom cut off: depth at the box center
            zc = float(np.median(D[int(y1 + 0.25 * (y2 - y1)):int(y2 - 0.25 * (y2 - y1)) + 1,
                                   int(x1 + 0.25 * (x2 - x1)):int(x2 - 0.25 * (x2 - x1)) + 1]))
            X = np.array([((x1 + x2) / 2 - cx) / f * zc, ((y1 + y2) / 2 - cy) / f * zc, zc])
            X = X - up * (X @ up + h)            # drop onto the floor
            how = "depth"
        a, b = float(X @ right), float(X @ fwd)
        if b < 0.3 or b > 12:
            continue
        dist = math.hypot(a, b)
        pw, ph, pd = SIZE_PRIORS[cls]
        w = float(np.clip((x2 - x1) * dist / f, pw * CLAMP[0], pw * CLAMP[1]))
        k = w / pw
        size = [round(w, 2), round(ph * float(np.clip(k, *CLAMP)), 2), round(pd * float(np.clip(k, *CLAMP)), 2)]
        # center: contact point pushed away from the camera by half the depth
        dirn = np.array([a, b]) / max(dist, 1e-6)
        c = np.array([a, b]) + dirn * size[2] / 2
        out.append({"class": cls, "right": float(c[0]), "forward": float(c[1]), "face": (-dirn).tolist(),
                    "size": size, "conf": d["conf"], "box": d["box"], "placed_from": how})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--at", type=float, nargs=2, required=True, metavar=("X", "Z"))
    ap.add_argument("--yaw", type=float, required=True, help="player yaw (rad, three.js)")
    args = ap.parse_args()
    entry = json.loads(pathlib.Path(args.detections).read_text())[0]
    D = np.load(entry["depth_map"])
    if D.shape != (entry["height"], entry["width"]):
        import cv2
        D = cv2.resize(D, (entry["width"], entry["height"]))
    f, cx, cy = entry["focal_px"], entry["width"] / 2, entry["height"] / 2
    boxes = [d["box"] for d in entry["detections"] if d["class"] in FLOOR_CLASSES]
    n, h, frac = fit_floor(D, f, cx, cy, boxes)
    tilt = math.degrees(math.asin(np.clip(n[2], -1, 1)))
    print(f"floor plane: camera {h:.2f} m up, tilted {tilt:+.0f} deg ({frac:.0%} of floor samples fit)")
    objs = place(entry, n, h, f, cx, cy, D)

    sp = pathlib.Path(args.scene)
    scene = json.loads(sp.read_text())
    block = scene["schematic"]["blocks"][args.block]
    poly = np.asarray(block["polygon"], float)
    F = np.array([-math.sin(args.yaw), -math.cos(args.yaw)])     # where the player looks
    R = np.array([math.cos(args.yaw), -math.sin(args.yaw)])
    at = np.array(args.at)
    new = []
    for i, o in enumerate(objs):
        c = at + R * o["right"] + F * o["forward"]
        if not in_poly(c, poly)[0]:
            print(f"  {o['class']}: lands outside the building, skipped")
            continue
        face = R * o["face"][0] + F * o["face"][1]
        new.append({"id": f"photo{args.block}_{pathlib.Path(entry['image']).stem}_{i}", "class": o["class"],
                    "position": [round(float(c[0]), 3), 0.0, round(float(c[1]), 3)], "size": o["size"],
                    "rotation_y": round(math.degrees(math.atan2(face[0], face[1])), 1), "collide": True, "model": None,
                    "source": {"placed_by": "interior", "from_photo": True, "block": args.block, "building": block.get("name"),
                               "image": str(pathlib.Path(entry["image"]).resolve().relative_to(ROOT.parent)),
                               "box": o["box"], "confidence": o["conf"], "placed_from": o["placed_from"]}})
        print(f"  {o['class']:<12} {o['right']:+.1f} m right, {o['forward']:.1f} m ahead ({o['placed_from']})")
    # what the photo shows replaces interior furniture standing in the same spots
    def clash(a, b):
        return math.dist(a["position"][::2], b["position"][::2]) < (max(a["size"][0], a["size"][2]) + max(b["size"][0], b["size"][2])) / 2
    scene["objects"] = [o for o in scene["objects"]
                        if not (o.get("source", {}).get("placed_by") == "interior" and o["source"].get("block") == args.block
                                and any(clash(o, q) for q in new))] + new
    sp.write_text(json.dumps(scene))
    print(f"{len(new)} pieces placed in {block.get('name')} from the photo")


if __name__ == "__main__":
    main()
