"""Stage 2: detections + floor registration -> objects in the scene JSON.

usage:
  python detect/place.py --detections detect/out/detections.json \
      --reg detect/scenes/bedroom/registration.json \
      --scene detect/scenes/bedroom/schematic.json \
      -o engine/scenes/bedroom.json [--viz]

registration.json: one entry per photo, >= 4 floor points clicked in the photo and the
same points on the floor plan (meters):
  [{"image": "detect/samples/a.jpg",
    "image_points": [[u, v], ...],
    "plan_points":  [[x, z], ...]}]

Two ways to place a floor-standing detection:
  floor  bottom-center of the box (where it touches the floor) -> H -> plan (x, z),
         then pushed back by half the object's depth. Precise when the contact is visible.
  depth  box center + metric depth -> 3D via the camera pose recovered from H
         (camera.py). Used when the contact is hidden: cut off by the image edge,
         covered by a nearer object, or when the two methods disagree badly.
Width comes from the box's bottom edge on the plan (floor) or box width x depth / f
(depth); height from depth if available. Both are clamped around the class prior.
Facing = box bottom edge direction on the plan, snapped to the walls.
"""
import argparse, json, math, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from homography import compute_h, apply_h, reprojection_error  # noqa: E402
from camera import (K_matrix, estimate_focal, pose_from_floor_h, pixel_to_world,  # noqa: E402
                    plan_point_depth, box_depth)
from priors import SIZE_PRIORS, FLOOR_CLASSES, CLAMP  # noqa: E402

SNAP_DEG = 20.0
ROUND_ITEMS = ("chair", "stool", "plant", "lamp", "basket")  # depth doesn't scale with width


def clamp_to_prior(measured, prior):
    if measured is None or not np.isfinite(measured) or measured <= 0:
        return prior
    return float(np.clip(measured, prior * CLAMP[0], prior * CLAMP[1]))


def wall_yaws(schematic):
    """Wall directions as three.js yaw angles (deg, mod 90). Default: axis-aligned."""
    out = []
    for w in schematic.get("walls", []):
        (x0, z0), (x1, z1) = w["from"], w["to"]
        out.append((-math.degrees(math.atan2(z1 - z0, x1 - x0))) % 90)
    return out or [0.0]


def snap_yaw(yaw, walls):
    best, best_d = yaw, SNAP_DEG
    for w in walls:
        for k in range(-4, 5):
            cand = w + 90 * k
            d = abs((yaw - cand + 180) % 360 - 180)
            if d < best_d:
                best, best_d = cand, d
    return ((best + 180) % 360) - 180


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class View:
    """Everything known about one registered photo."""

    def __init__(self, reg, det_entry):
        self.reg, self.entry = reg, det_entry
        self.w, self.h = det_entry["width"], det_entry["height"]
        ip, pp = np.asarray(reg["image_points"], float), np.asarray(reg["plan_points"], float)
        self.H = compute_h(ip, pp)                     # image -> plan
        self.H_err = reprojection_error(self.H, ip, pp)
        self.w_sign = np.sign((np.c_[ip, np.ones(len(ip))] @ self.H.T)[0, 2])

        # focal length, best source first: calibration > EXIF > from H > assumed FOV.
        # (H pins f down poorly when the camera is nearly level.)
        H_p2i = np.linalg.inv(self.H)
        cx, cy = self.w / 2, self.h / 2
        f_h = estimate_focal(H_p2i, cx, cy)
        if reg.get("K"):
            Kc = np.array(reg["K"], float)
            self.f, self.f_src = float((Kc[0, 0] + Kc[1, 1]) / 2), "calibrated"
            cx, cy = float(Kc[0, 2]), float(Kc[1, 2])
        elif str(det_entry.get("focal_source", "")).startswith("exif"):
            self.f, self.f_src = det_entry["focal_px"], det_entry["focal_source"]
        elif f_h and 0.3 * self.w < f_h < 3 * self.w:
            self.f, self.f_src = f_h, "from homography"
        else:
            self.f, self.f_src = det_entry["focal_px"], det_entry.get("focal_source", "assumed")
        self.K = K_matrix(self.f, cx, cy)
        self.wall_height = 2.7
        self.pose = pose_from_floor_h(H_p2i, self.K, pp[0])
        self.cam_plan = self.pose["C"][:2]

        # depth map, rescaled so the clicked floor points sit at their true distance
        self.D, self.depth_scale = None, None
        if det_entry.get("depth_map") and pathlib.Path(det_entry["depth_map"]).exists():
            D = np.load(det_entry["depth_map"])
            ratios = []
            vis = reg.get("visible") or [True] * len(ip)
            for (u, v), p, ok in zip(ip, pp, vis):
                if not ok or not (0 <= u < self.w and 0 <= v < self.h):
                    continue  # hidden floor point: the depth there is of whatever covers it
                ui, vi = int(u), int(v)
                if D[vi, ui] > 0:
                    ratios.append(plan_point_depth(self.pose, p) / D[vi, ui])
            # no visible floor point: trust the metric model as is
            self.depth_scale = float(np.clip(np.median(ratios), 1 / 3, 3)) if ratios else 1.0
            self.D = D * self.depth_scale

    def floor_ok(self, pts):
        ph = np.c_[pts, np.ones(len(pts))] @ self.H.T
        return bool((np.sign(ph[:, 2]) == self.w_sign).all())


def covering(d, others):
    """Another floor object whose box covers d's bottom-center and reaches lower in the
    image. In 2D that's either "d is behind it" or "d sits on top of it"; depth decides."""
    x1, _, x2, y2 = d["box"]
    bx = (x1 + x2) / 2
    for o in others:
        if o is d:
            continue
        ox1, oy1, ox2, oy2 = o["box"]
        if ox1 < bx < ox2 and oy1 < y2 - 3 and oy2 > y2 + 10:
            return o
    return None


def place_detection(d, view, walls, floor_dets):
    cls = d["class"]
    x1, y1, x2, y2 = d["box"]
    cx = (x1 + x2) / 2
    reasons = []

    z = box_depth(view.D, d["box"]) if view.D is not None else None
    z = z if z and z > 0 else None
    # tall "cabinet" is almost always a bookshelf (open-vocab detectors mix them up)
    if cls == "cabinet" and z and (y2 - y1) * z / view.f > 1.4:
        cls = "bookshelf"
        reasons.append("tall cabinet -> bookshelf")
    pw, ph, pd = SIZE_PRIORS[cls]

    # --- floor-contact estimate
    floor_est = None
    probe = np.array([[x1, y2], [x2, y2], [cx, y2], [cx, y2 - 0.1 * (y2 - y1) - 2]])
    if view.floor_ok(probe):
        pL, pR, pB, pFar = apply_h(view.H, probe)
        width = clamp_to_prior(np.linalg.norm(pR - pL), pw)
        floor_est = dict(base=pB, away=unit(pFar - pB), width=width, axis=unit(pR - pL))
    else:
        reasons.append("base above horizon")

    # --- depth estimate
    depth_est = None
    if z:
        if True:
            P = pixel_to_world(view.pose, view.K, cx, (y1 + y2) / 2, z)
            pL = pixel_to_world(view.pose, view.K, x1, (y1 + y2) / 2, z)
            pR = pixel_to_world(view.pose, view.K, x2, (y1 + y2) / 2, z)
            depth_est = dict(front=P[:2], away=unit(P[:2] - view.cam_plan),
                             width=clamp_to_prior(np.linalg.norm(pR[:2] - pL[:2]), pw),
                             height=clamp_to_prior((y2 - y1) * z / view.f, ph),
                             axis=unit(pR[:2] - pL[:2]))

    clipped = y2 >= view.h - 2
    cover = covering(d, floor_dets)
    occ, resting_on = None, None
    if cover is not None:
        z_cov = box_depth(view.D, cover["box"]) if view.D is not None else None
        if z and z_cov and abs(z - z_cov) < max(0.4, 0.15 * z):
            resting_on = cover["class"]
        else:
            occ = cover["class"]
    if clipped:
        reasons.append("base cut off by image edge")
    if occ:
        reasons.append(f"base hidden by {occ}")

    # elevation: bottom of the box lifted with depth. Only trusted when the box bottom is
    # the object's real bottom (not cut off, not hidden behind something).
    elev = 0.0
    if depth_est and not clipped and not occ and (resting_on or floor_est is None):
        elev = max(0.0, float(pixel_to_world(view.pose, view.K, cx, y2, z)[2]))
        wall_h = view.wall_height
        elev = min(elev, max(0.0, wall_h - SIZE_PRIORS[cls][1]))
        if elev > 0.15:
            reasons.append(f"resting on {resting_on} at {elev:.2f} m" if resting_on
                           else f"elevated {elev:.2f} m")
        else:
            elev = 0.0

    use_depth = depth_est is not None and (floor_est is None or clipped or occ or elev > 0)
    if depth_est is not None and floor_est is not None and not use_depth:
        # same rough point from both: floor base vs depth front surface
        gap = np.linalg.norm(floor_est["base"] - depth_est["front"])
        rng = np.linalg.norm(floor_est["base"] - view.cam_plan)
        if gap > max(0.6, 0.35 * rng):
            use_depth = True
            reasons.append(f"floor/depth disagree by {gap:.1f} m")
    if floor_est is None and depth_est is None:
        return None, reasons

    est = depth_est if use_depth else floor_est
    width = est["width"]
    height = depth_est["height"] if depth_est else ph
    if clipped or occ:
        # only part of the object is visible: the measurement is a lower bound
        height = max(height, ph)
    depth = pd if cls in ROUND_ITEMS else float(np.clip(pd * width / pw, pd * CLAMP[0], pd * CLAMP[1]))
    front = est["front"] if use_depth else est["base"]
    center = front + est["away"] * depth / 2
    yaw = snap_yaw(math.degrees(math.atan2(-est["axis"][1], est["axis"][0])), walls)

    return {
        "class": cls,
        "position": [round(float(center[0]), 3), round(elev, 3), round(float(center[1]), 3)],
        "size": [round(width, 3), round(height, 3), round(depth, 3)],
        "rotation_y": round(yaw, 1),
        "model": None,
        "source": {"image": view.reg["image"], "confidence": d["conf"], "box": d["box"],
                   "placed_by": "depth" if use_depth else "floor", "notes": reasons},
    }, reasons


def view_camera(view):
    """Photo camera in engine terms (three.js yaw/pitch, vertical FOV) so the engine can
    jump to the exact viewpoint and the render can be compared with the photo."""
    R, up = view.pose["R"], view.pose["up_sign"]
    fwd = R.T @ np.array([0, 0, 1.0])        # optical axis in (plan x, plan z, n)
    fx, fz, fv = fwd[0], fwd[1], fwd[2] * up
    return {
        "image": view.reg["image"],
        "position": [round(float(view.cam_plan[0]), 3), round(view.pose["height"], 3), round(float(view.cam_plan[1]), 3)],
        "yaw_deg": round(math.degrees(math.atan2(-fx, -fz)), 2),
        "pitch_deg": round(math.degrees(math.asin(np.clip(fv, -1, 1))), 2),
        "fov_v_deg": round(math.degrees(2 * math.atan(view.h / 2 / view.f)), 2),
        "aspect": round(view.w / view.h, 4),
    }


def inside_floor(obj, schematic, margin=0.75):
    f = schematic.get("floor")
    if not f:
        return True
    x, _, z = obj["position"]
    (x0, z0), (x1, z1) = f["min"], f["max"]
    return x0 - margin <= x <= x1 + margin and z0 - margin <= z <= z1 + margin


def clamp_into_floor(obj, schematic):
    f = schematic.get("floor")
    if not f:
        return
    (x0, z0), (x1, z1) = f["min"], f["max"]
    r = min(obj["size"][0], obj["size"][2]) / 2
    obj["position"][0] = round(float(np.clip(obj["position"][0], x0 + r, x1 - r)), 3)
    obj["position"][2] = round(float(np.clip(obj["position"][2], z0 + r, z1 - r)), 3)


def merge_duplicates(objs):
    """Same class, centers closer than ~ half its footprint -> one object (conf-weighted)."""
    merged = []
    for o in sorted(objs, key=lambda o: -o["source"]["confidence"]):
        for m in merged:
            if m["class"] != o["class"]:
                continue
            if math.dist(m["position"][::2], o["position"][::2]) < 0.6 * max(m["size"][0], m["size"][2]):
                wa, wb = m["source"]["confidence"], o["source"]["confidence"]
                for i in (0, 2):
                    m["position"][i] = round((m["position"][i] * wa + o["position"][i] * wb) / (wa + wb), 3)
                m.setdefault("also_seen_in", []).append(o["source"]["image"])
                break
        else:
            merged.append(o)
    return merged


def draw_plan(scene, views, path, px_per_m=80, pad=0.5):
    import cv2
    f = scene["schematic"].get("floor", {"min": [0, 0], "max": [6, 6]})
    (x0, z0), (x1, z1) = f["min"], f["max"]
    W, Hh = int((x1 - x0 + 2 * pad) * px_per_m), int((z1 - z0 + 2 * pad) * px_per_m)
    img = np.full((Hh, W, 3), 245, np.uint8)
    P = lambda x, z: (int((x - x0 + pad) * px_per_m), int((z - z0 + pad) * px_per_m))  # noqa: E731
    cv2.rectangle(img, P(x0, z0), P(x1, z1), (225, 215, 200), -1)
    for w in scene["schematic"].get("walls", []):
        cv2.line(img, P(*w["from"]), P(*w["to"]), (60, 60, 60), 3)
    for v in views:
        for i, (x, z) in enumerate(v.reg["plan_points"]):
            cv2.circle(img, P(x, z), 5, (0, 140, 255), -1)
            cv2.putText(img, str(i + 1), P(x + 0.08, z - 0.08), 0, 0.45, (0, 100, 200), 1)
        c = v.cam_plan
        cv2.drawMarker(img, P(*c), (200, 0, 200), cv2.MARKER_TRIANGLE_UP, 14, 2)
        cv2.putText(img, f"cam h={v.pose['height']:.2f}m", P(c[0] + 0.1, c[1]), 0, 0.4, (200, 0, 200), 1)
    for o in scene["objects"]:
        x, _, z = o["position"]
        w, _, d = o["size"]
        a = -math.radians(o["rotation_y"])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        corners = (R @ np.array([[-w, -d], [w, -d], [w, d], [-w, d]]).T / 2).T + [x, z]
        color = (40, 140, 40) if o["source"].get("placed_by") == "depth" else (180, 60, 40)
        cv2.polylines(img, [np.array([P(*c) for c in corners])], True, color, 2)
        cv2.putText(img, o["class"], P(x - w / 2, z), 0, 0.45, (40, 40, 40), 1)
    cv2.putText(img, "blue=floor contact  green=depth", (8, Hh - 8), 0, 0.45, (80, 80, 80), 1)
    cv2.imwrite(str(path), img)


def draw_photo(view, placed, path):
    import cv2
    img = cv2.imread(view.reg["image"])
    for i, (u, v) in enumerate(view.reg["image_points"]):
        cv2.circle(img, (int(u), int(v)), 6, (0, 140, 255), -1)
        cv2.putText(img, str(i + 1), (int(u) + 6, int(v) - 6), 0, 0.6, (0, 140, 255), 2)
    for o in placed:
        x1, y1, x2, y2 = map(int, o["source"]["box"])
        by_depth = o["source"]["placed_by"] == "depth"
        color = (40, 140, 40) if by_depth else (180, 60, 40)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if by_depth:
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)
        else:
            cv2.circle(img, ((x1 + x2) // 2, y2), 5, color, -1)
        cv2.putText(img, o["class"], (x1, y1 - 4), 0, 0.5, color, 2)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--reg", required=True)
    ap.add_argument("--scene", required=True, help="base scene JSON with the schematic")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--append", action="store_true", help="keep objects already in --scene")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    scene = json.loads(pathlib.Path(args.scene).read_text())
    det_list = json.loads(pathlib.Path(args.detections).read_text())
    dets_by_img = {str(pathlib.Path(r["image"]).resolve()): r for r in det_list}
    regs = json.loads(pathlib.Path(args.reg).read_text())
    walls = wall_yaws(scene["schematic"])

    placed, views, per_view = [], [], []
    for reg in regs:
        name = pathlib.Path(reg["image"]).name
        key = str(pathlib.Path(reg["image"]).resolve())
        if key not in dets_by_img:
            print(f"skip {reg['image']}: no detections")
            continue
        view = View(reg, dets_by_img[key])
        view.wall_height = scene["schematic"].get("wall_height", 2.7)
        views.append(view)
        floor_dets = [d for d in view.entry["detections"]
                      if d["class"] in FLOOR_CLASSES and d["conf"] >= args.min_conf]
        print(f"{name}: H fit {view.H_err * 100:.1f} cm | f {view.f:.0f}px ({view.f_src}) | "
              f"camera at ({view.cam_plan[0]:.2f}, {view.cam_plan[1]:.2f}) h {view.pose['height']:.2f} m | "
              f"depth scale {view.depth_scale if view.depth_scale is None else round(view.depth_scale, 2)}")
        mine = []
        for d in floor_dets:
            o, reasons = place_detection(d, view, walls, floor_dets)
            if o is None:
                print(f"  skip {d['class']} ({d['conf']:.2f}): {'; '.join(reasons)}")
                continue
            if not inside_floor(o, scene["schematic"]):
                print(f"  drop {d['class']} ({d['conf']:.2f}) by {o['source']['placed_by']}: outside the floor "
                      f"at ({o['position'][0]:.2f}, {o['position'][2]:.2f})")
                continue
            clamp_into_floor(o, scene["schematic"])
            mine.append(o)
            note = f" [{'; '.join(reasons)}]" if reasons else ""
            print(f"  {o['class']:<10} by {o['source']['placed_by']:<5} at "
                  f"({o['position'][0]:.2f}, {o['position'][2]:.2f}) size {o['size']}{note}")
        placed += mine
        per_view.append((view, mine))

    objs = merge_duplicates(placed)
    counts = {}
    final = []
    for o in objs:
        counts[o["class"]] = counts.get(o["class"], 0) + 1
        final.append({"id": f"{o['class']}_{counts[o['class']] - 1}", **o})
    scene["objects"] = (scene.get("objects", []) if args.append else []) + final
    scene["cameras"] = [view_camera(v) for v in views]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2))
    print(f"wrote {out} with {len(final)} objects")

    if args.viz:
        vdir = pathlib.Path(args.detections).parent
        draw_plan(scene, views, vdir / f"{out.stem}_plan.png")
        for view, mine in per_view:
            ip = pathlib.Path(view.reg["image"])
            stem = f"{ip.parent.name}_{ip.stem}" if ip.stem == "image" else ip.stem
            draw_photo(view, mine, vdir / f"{stem}_placed.jpg")
        print(f"viz in {vdir}")


if __name__ == "__main__":
    main()
