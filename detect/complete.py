"""Complete a placed scene: the whole room, not just what the camera saw.

1. Room type: from the dataset label, else CLIP zero-shot on the photos (bedroom /
   living room / office / dining room; outdoor or other -> nothing is completed).
2. Room shell: a rectangle aligned with the walls the camera saw. Sides backed by an
   observed wall stay where they are; unseen sides (always the one behind the camera)
   are pushed out until the room reaches a typical size for its type.
3. Unseen contents: floor the camera could not see gets furniture from the room prior
   (room_prior.py, learned from SUN RGB-D): expected count per class = learned rate
   per m2 x unseen floor area, capped by how many such things a room usually has, plus
   anything nearly every room of that type has (a bedroom's bed). Each one is put where
   the prior likes it best: against a wall or not, near the furniture it usually sits
   next to, facing the usual way, never where the camera would have seen it, never
   overlapping anything.

Predicted objects carry source.placed_by = "predicted".

usage:
  python detect/complete.py engine/scenes/<scene>.json [--viz DIR] [--no-predict]
"""
import argparse, json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from priors import MATCH_GROUPS  # noqa: E402
from room_prior import FACE_TOWARD, FRONT_ON_SHORT_SIDE, PRIOR, face_toward, local_offset, rel_angle, real_walls  # noqa: E402

# typical room footprints (short, long) in meters; used only for sides nobody saw
ROOM_SIZE = {"bedroom": (3.4, 4.0), "living_room": (4.2, 5.4), "office": (3.8, 4.8), "dining_area": (3.6, 4.5)}
DEFAULT_SIZE = (3.8, 4.6)
SCENE_PROMPTS = {
    "bedroom": "a photo of a bedroom", "living_room": "a photo of a living room",
    "office": "a photo of an office", "dining_area": "a photo of a dining room",
    "outdoor": "a photo taken outdoors", "other": "a photo of a kitchen, bathroom, or hallway",
}
GRID = 0.1             # floor sampling for the seen / unseen split
WALL = 0.12
MAX_PREDICTED = 10
SIGMA = 0.3            # relation kernel width (m)
FACE_RANGE_PRED = 1.6
# table lamps stand on furniture (SUN "lamp" median height 0.68 m); never on the floor
ON_TOP = {"lamp": ("nightstand", "dresser", "desk", "table", "cabinet")}
TUCK_UNDER = {("chair", "table"), ("chair", "desk"), ("stool", "table"), ("stool", "desk"),
              ("chair", "dining_table")}


# ------------------------------------------------------------------ scene type

def scene_type(scene):
    ds = (scene.get("dataset") or {}).get("scene")
    if ds:
        return ds, {"source": "dataset label"}
    imgs = [c["image"] for c in scene.get("cameras", []) if c.get("image")]
    if not imgs:
        return None, {"source": "none"}
    from PIL import Image
    from models_catalog import Clip
    clip = Clip()
    im = clip.embed([Image.open(REPO / p).convert("RGB") for p in imgs]).mean(0)
    im /= np.linalg.norm(im)
    keys = list(SCENE_PROMPTS)
    T = clip.embed_text([SCENE_PROMPTS[k] for k in keys])
    p = np.exp(100 * (T @ im))
    p /= p.sum()
    best = keys[int(p.argmax())]
    probs = {k: round(float(v), 3) for k, v in zip(keys, p)}
    return (best if best in ROOM_SIZE else None), {"source": "CLIP zero-shot", "probs": probs}


# ------------------------------------------------------------------ cameras

class Cam:
    """Photo camera from scene["cameras"] (three.js yaw/pitch, vertical FOV)."""

    def __init__(self, c):
        y, p = math.radians(c["yaw_deg"]), math.radians(c["pitch_deg"])
        self.C = np.array(c["position"], float)
        self.f = np.array([-math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])
        self.r = np.array([math.cos(y), 0.0, -math.sin(y)])
        self.u = np.cross(self.r, self.f)
        self.tv = math.tan(math.radians(c["fov_v_deg"]) / 2)
        self.th = self.tv * c.get("aspect", 4 / 3)

    def sees(self, P):
        d = np.atleast_2d(P) - self.C
        z, x, y = d @ self.f, d @ self.r, d @ self.u
        return (z > 0.2) & (np.abs(x) <= self.th * z) & (np.abs(y) <= self.tv * z)


def seen(cams, P):
    P = np.atleast_2d(P)
    out = np.zeros(len(P), bool)
    for c in cams:
        out |= c.sees(P)
    return out


# ------------------------------------------------------------------ boxes

def obj_axes(o):
    """Plan axes of an engine object: local +x (width) and local +z (front)."""
    t = math.radians(o["rotation_y"])
    return np.array([math.cos(t), -math.sin(t)]), np.array([math.sin(t), math.cos(t)])


def corners(c, ux, uz, w, d):
    c = np.asarray(c)
    return np.array([c + sx * ux * w / 2 + sz * uz * d / 2 for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def overlap(A, B):
    """Separating-axis test for two convex quads (4x2 corner arrays)."""
    for P in (A, B):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]])
            a, b = A @ n, B @ n
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def yaw_from_front(f):
    return math.degrees(math.atan2(f[0], f[1]))


# ------------------------------------------------------------------ room shell

def complete_room(scene, stype, cams):
    sch = scene["schematic"]
    if (sch.get("room") or {}).get("given"):
        return sch["room"]
    cam_xz = [c.C[[0, 2]] for c in cams] or [np.array(scene.get("spawn", {}).get("position", [0, 0, 0]))[[0, 2]]]
    seen_walls = sch.get("observed_outline") or sch.get("walls", [])     # re-runs start from what was seen
    segs = [(np.array(w["from"], float), np.array(w["to"], float)) for w in seen_walls]
    walls = [s for s in segs if all(real_walls([s[0], s[1]], c) for c in cam_xz)]   # not view-cone edges
    # dominant wall direction (mod 90 deg), length weighted
    if walls:
        z = sum(np.linalg.norm(b - a) * np.exp(4j * math.atan2(*(b - a)[::-1])) for a, b in walls)
        phi = np.angle(z) / 4
    else:
        phi = 0.0
    u, v = np.array([math.cos(phi), math.sin(phi)]), np.array([-math.sin(phi), math.cos(phi)])
    to_uv = lambda p: np.array([np.dot(p, u), np.dot(p, v)])  # noqa: E731

    # Each side of the room sits on the longest wall the camera saw on that side. Outline
    # beyond it (a hallway or the next room seen through a doorway) is an opening, not room.
    cam_uv = np.mean([to_uv(c) for c in cam_xz], 0)
    best = {}                                   # side -> (wall length, coordinate)
    for a, b in walls:
        qa, qb = to_uv(a), to_uv(b)
        L = np.linalg.norm(qb - qa)
        if L < 0.8:
            continue
        d = (qb - qa) / L
        for ax in (0, 1):                       # wall runs across axis `ax` (constant coord on it)
            if abs(d[ax]) > math.sin(math.radians(15)):
                continue
            coord = (qa[ax] + qb[ax]) / 2
            side = "uv"[ax] + ("-" if coord < cam_uv[ax] else "+")
            if L > best.get(side, (0, 0))[0]:
                best[side] = (L, coord)
    observed = {k: k in best for k in ("u-", "u+", "v-", "v+")}

    # must contain: detected furniture and the cameras; unseen sides also take the outline
    must = [q for o in scene["objects"]
            for q in map(to_uv, corners(o["position"][::2], *obj_axes(o), o["size"][0], o["size"][2]))]
    must += [cam_uv + [0.5, 0.5], cam_uv - [0.5, 0.5]]
    outline = [to_uv(p) for sg in segs for p in sg]
    lo, hi = np.zeros(2), np.zeros(2)
    for ax in (0, 1):
        k = "uv"[ax]
        m = np.array([q[ax] for q in must])
        o_ = np.array([q[ax] for q in outline] or [cam_uv[ax]])
        lo[ax] = min(best[k + "-"][1], m.min()) if observed[k + "-"] else min(o_.min(), m.min())
        hi[ax] = max(best[k + "+"][1], m.max()) if observed[k + "+"] else max(o_.max(), m.max())

    short, long_ = ROOM_SIZE.get(stype, DEFAULT_SIZE)
    span = hi - lo
    target = np.array([long_, short]) if span[0] >= span[1] else np.array([short, long_])
    for ax in (0, 1):
        deficit = target[ax] - (hi[ax] - lo[ax])
        if deficit <= 0:
            continue
        key = "uv"[ax]
        lo_obs, hi_obs = observed[key + "-"], observed[key + "+"]
        if lo_obs and not hi_obs:
            hi[ax] += deficit
        elif hi_obs and not lo_obs:
            lo[ax] -= deficit
        elif not lo_obs and not hi_obs:
            lo[ax] -= deficit / 2
            hi[ax] += deficit / 2
    to_plan = lambda q: (q[0] * u + q[1] * v).round(4).tolist()  # noqa: E731
    rect_uv = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]
    rect = [to_plan(np.asarray(q)) for q in rect_uv]

    # Seen floor beyond an observed wall (a hallway, the next room through a doorway) is
    # kept as an attached space: its own walls, joined to the room by a gap in that wall.
    exts = []
    if len(seen_walls) >= 3:
        poly = np.array([to_uv(np.array(w["from"], float)) for w in seen_walls])
        from room_prior import in_poly
        g_lo, g_hi = poly.min(0), poly.max(0)
        G = np.stack(np.meshgrid(np.arange(g_lo[0], g_hi[0], GRID), np.arange(g_lo[1], g_hi[1], GRID)), -1).reshape(-1, 2)
        G = G[in_poly(G, poly)]
        for side in ("u-", "u+", "v-", "v+"):
            if not observed[side]:
                continue                       # unseen sides were already pushed past the outline
            ax, sgn = "uv".index(side[0]), 1 if side[1] == "+" else -1
            edge = hi[ax] if sgn > 0 else lo[ax]
            o = 1 - ax
            P = G[(sgn * (G[:, ax] - edge) > 0.3) & (G[:, o] > lo[o]) & (G[:, o] < hi[o])]
            if len(P) * GRID ** 2 < 1.0:
                continue
            a0, a1 = P[:, o].min(), P[:, o].max()
            far = P[:, ax].max() if sgn > 0 else P[:, ax].min()
            if a1 - a0 < 0.7:
                continue
            q = np.zeros((4, 2))
            q[:, ax] = [edge, far, far, edge]
            q[:, o] = [a0, a0, a1, a1]
            exts.append({"side": side, "corners": [to_plan(x) for x in q],
                         "gap": [round(float(a0), 3), round(float(a1), 3)],
                         "size": [round(float(abs(far - edge)), 2), round(float(a1 - a0), 2)]})
    return {"corners": rect, "angle_deg": round(math.degrees(phi), 2),
            "size": [round(float(hi[0] - lo[0]), 2), round(float(hi[1] - lo[1]), 2)],
            "observed_sides": observed, "scene_type": stype, "extensions": exts,
            "_uv": {"u": u.tolist(), "v": v.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}}


def room_walls(room):
    """Main rectangle's walls with a gap where each extension attaches, plus each
    extension's three outer walls."""
    uv = room["_uv"]
    u, v, lo, hi = (np.array(uv[k]) for k in ("u", "v", "lo", "hi"))
    P = lambda a, b: (a * u + b * v).round(4).tolist()  # noqa: E731
    W = lambda a, b: {"from": a, "to": b, "thickness": WALL}  # noqa: E731
    walls = []
    for side in ("v-", "u+", "v+", "u-"):
        ax, sgn = "uv".index(side[0]), side[1] == "+"
        o = 1 - ax
        edge = (hi if sgn else lo)[ax]
        gaps = sorted(e["gap"] for e in room.get("extensions", []) if e["side"] == side)
        t, pieces = lo[o], []
        for g0, g1 in gaps:
            pieces.append((t, g0))
            t = g1
        pieces.append((t, hi[o]))
        for t0, t1 in pieces:
            if t1 - t0 > 0.05:
                a, b = np.zeros(2), np.zeros(2)
                a[ax], a[o], b[ax], b[o] = edge, t0, edge, t1
                walls.append(W(P(*a), P(*b)))
    for e in room.get("extensions", []):
        c = e["corners"]                 # edge-a0, far-a0, far-a1, edge-a1
        walls += [W(c[0], c[1]), W(c[1], c[2]), W(c[2], c[3])]
    return walls


def apply_room(scene, room):
    sch = scene["schematic"]
    if not room.get("given"):
        rect = room["corners"]
        sch["observed_outline"] = sch.get("observed_outline") or sch.get("walls", [])
        sch["walls"] = room_walls(room)
        polys = [rect] + [e["corners"] for e in room.get("extensions", [])]
        xs, zs = [p[0] for q in polys for p in q], [p[1] for q in polys for p in q]
        sch["floor"] = {"min": [min(xs), min(zs)], "max": [max(xs), max(zs)], "polygon": rect, "polygons": polys}
    sch["room"] = room


# ------------------------------------------------------------------ facing of what was seen

def orient_observed(scene, polys, cams):
    """place.py fixes each object's axis but can't tell front from back. With the room
    known: against a wall -> faces away from it (a bed: its foot end); a seat near its
    table / a sofa near its coffee table -> faces it; else keeps facing the camera."""
    cam_xz = [c.C[[0, 2]] for c in cams]
    boxes, objs = [], []
    for o in scene["objects"]:
        if o.get("source", {}).get("placed_by") == "predicted" or o["position"][1] > 0.3:
            continue
        f = anchor_front(o, polys, cam_xz)
        boxes.append({"cls": o["class"], "c": np.array(o["position"][::2], float), "front": f,
                      "w": o["size"][0], "d": o["size"][2], "facing_from": "wall/camera"})
        objs.append(o)
    face_toward(boxes)
    for o, b in zip(objs, boxes):
        ux, uz = obj_axes(o)
        f = b["front"]
        w, d = o["size"][0], o["size"][2]
        if abs(f @ ux) > 0.5 and max(w, d) > 1.3 * min(w, d):
            # the photo fit chose this axis; context may only pick front vs back
            cam = cam_xz[0] if cam_xz else b["c"]
            f = uz if uz @ (cam - b["c"]) >= 0 else -uz
            for n in (uz, -uz):          # ...unless a wall is right behind one of them
                if not inside_any(b["c"] + n * (d / 2 + 0.35), polys):
                    f = -n
        if abs(f @ ux) > 0.5:            # front along the old width axis: turn 90, swap w/d
            o["size"][0], o["size"][2] = o["size"][2], o["size"][0]
        new = round(yaw_from_front(f), 1)
        if abs((new - o["rotation_y"] + 180) % 360 - 180) > 1:
            o.setdefault("source", {})["facing"] = b["facing_from"]
        o["rotation_y"] = new


# ------------------------------------------------------------------ prediction

def inside_any(p, polys):
    return any(point_in_poly(p, q) for q in polys)


def anchor_front(o, room_rect, cam_xz):
    """Facing used for relations, defined the same way room_prior.py learned it: away from
    the wall the object stands against (longest such side), else the long side facing
    the camera's side of the room."""
    ux, uz = obj_axes(o)
    c = np.array(o["position"][::2], float)
    w, d = o["size"][0], o["size"][2]
    best = None
    long_x = w >= d
    for n, half, length in ((uz, d / 2, w), (-uz, d / 2, w), (ux, w / 2, d), (-ux, w / 2, d)):
        short_side = (abs(n @ ux) > 0.5) == long_x       # normal along the long axis
        if o["class"] in FRONT_ON_SHORT_SIDE and not short_side:
            continue
        polys = room_rect if isinstance(room_rect, list) and np.ndim(room_rect[0]) == 2 else [room_rect]
        if not inside_any(c + n * (half + 0.35), polys) and (best is None or length > best[1]):
            best = (-n, length)
    if best:
        return best[0]
    f = uz if long_x != (o["class"] in FRONT_ON_SHORT_SIDE) else ux
    cam = cam_xz[0] if cam_xz else c
    return f if np.dot(f, cam - c) >= 0 else -f


def point_in_poly(p, poly):
    c, n = False, len(poly)
    for i in range(n):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
        if (z1 > p[1]) != (z2 > p[1]) and p[0] < (x2 - x1) * (p[1] - z1) / (z2 - z1) + x1:
            c = not c
    return c


def rel_term(samples, dx, dz, dyaw, pair_range=3.5):
    S = np.asarray(samples)
    k = np.exp(-((S[:, 0] - dx) ** 2 + (S[:, 1] - dz) ** 2) / (2 * SIGMA ** 2))
    p_off = k.mean() / (2 * math.pi * SIGMA ** 2)                 # density per m2
    da = (S[:, 2] - dyaw + 180) % 360 - 180
    ang = float((k * np.exp(-da ** 2 / (2 * 30 ** 2))).sum() / (k.sum() + 1e-9))
    # reward only: 0 unless the spot is likelier than uniform within the relation's range
    # (being elsewhere in the room is fine), ~+2-4 at the usual spot; the facing factor
    # only matters where the offset is typical
    ratio = p_off * math.pi * pair_range ** 2
    if ratio <= 1:
        return 0.0
    return math.log(ratio) + 0.5 * math.log((ang + 0.1) / 1.1)


def predict(scene, room, cams, prior):
    stype = room.get("scene_type")
    P = prior.get(stype)
    if not P:
        return [], {"reason": f"no prior for scene type {stype}"}
    rect = np.array(room["corners"], float)
    u = (rect[1] - rect[0]) / np.linalg.norm(rect[1] - rect[0])
    v = (rect[3] - rect[0]) / np.linalg.norm(rect[3] - rect[0])
    Lu, Lv = np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[3] - rect[0])

    # seen / unseen floor
    gu, gv = np.meshgrid(np.arange(GRID / 2, Lu, GRID), np.arange(GRID / 2, Lv, GRID))
    floor = rect[0] + gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v
    vis = seen(cams, np.c_[floor[:, 0], np.zeros(len(floor)), floor[:, 1]])
    # unseen area for the rates is measured like room_prior.py measured seen area: floor
    # outside every camera's horizontal field of view
    in_wedge = np.zeros(len(floor), bool)
    for c in cams:
        fh = c.f[[0, 2]] / np.linalg.norm(c.f[[0, 2]])
        d = floor - c.C[[0, 2]]
        along = d @ fh
        side = np.abs(d @ np.array([fh[1], -fh[0]]))
        in_wedge |= (along > 0.1) & (side <= c.th * along)
    unseen_area = float((~in_wedge).sum() * GRID ** 2)

    observed = [o for o in scene["objects"] if o.get("source", {}).get("placed_by") != "predicted"]
    placed = []
    for o in observed:
        ux, uz = obj_axes(o)
        f = anchor_front(o, rect, [c.C[[0, 2]] for c in cams])
        placed.append({"cls": o["class"], "c": np.array(o["position"][::2], float), "front": f,
                       "w": o["size"][0], "d": o["size"][2], "top": o["position"][1] + o["size"][1],
                       "quad": corners(o["position"][::2], ux, uz, o["size"][0], o["size"][2])})
    face_toward(placed)

    # how many of each class to add
    want = []
    for cls, info in P["classes"].items():
        n_obs = sum(o["class"] == cls for o in observed)
        expected = info["rate_per_m2"] * unseen_area
        n = min(max(1, info["max_count"]) - n_obs, int(expected + 0.5))
        if info["frames_present"] >= 0.5 and n_obs == 0:     # most such rooms have one
            n = max(n, 1)
        for _ in range(max(0, n)):
            want.append((cls, info, expected))
    want.sort(key=lambda t: -t[1]["size"][0] * t[1]["size"][2])   # big anchors first
    want = want[:MAX_PREDICTED]

    out, log = [], []
    cam_xz = [c.C[[0, 2]] for c in cams]
    for cls, info, expected in want:
        w, h, d = info["size"]
        if cls in ON_TOP:
            o = place_on_top(cls, info, expected, placed, cams, out, stype)
            log.append(f"{cls}: " + (f"on the {o['source']['near']}" if o else "nothing unseen to stand on"))
            if o:
                out.append(o)
                placed.append({"cls": cls, "c": np.array(o["position"][::2]), "front": np.array([0.0, 1.0]),
                               "quad": np.zeros((4, 2)) + o["position"][::2], "w": w, "d": d, "top": o["position"][1] + h})
            continue
        pw = float(np.clip(info["p_wall"], 0.05, 0.95))
        cands = []
        # against each wall, facing into the room
        for a, b, n_in in ((rect[0], rect[1], v), (rect[1], rect[2], -u), (rect[2], rect[3], -v), (rect[3], rect[0], u)):
            L = np.linalg.norm(b - a)
            t = (b - a) / L
            for s in np.arange(w / 2 + 0.05, L - w / 2 - 0.05 + 1e-6, 0.15):
                cands.append((a + t * s + n_in * (WALL / 2 + d / 2 + 0.03), t, n_in, True))
        # free standing, axis aligned with the room, all 4 facings
        m = max(w, d) / 2 + 0.3
        for su in np.arange(m, Lu - m + 1e-6, 0.25):
            for sv in np.arange(m, Lv - m + 1e-6, 0.25):
                c = rect[0] + su * u + sv * v
                for f in (u, -u, v, -v):
                    cands.append((c, np.array([f[1], -f[0]]), f, False))
        targets = [p for p in placed if p["cls"] in FACE_TOWARD.get(cls, ())]

        best = None
        for c, ux, f, at_wall in cands:
            if targets and not at_wall:
                t = min(targets, key=lambda p: np.linalg.norm(p["c"] - c))
                dvec = t["c"] - c
                if np.linalg.norm(dvec) < FACE_RANGE_PRED and np.dot(f, dvec) < 0.7 * np.linalg.norm(dvec):
                    continue                      # a seat next to its table faces it
            quad = corners(c, ux, f, w, d)
            if any(np.linalg.norm(c - q) < 0.7 for q in cam_xz):
                continue
            # the camera would have seen it: sample the footprint at floor and top height
            samp = [c + a * ux * w / 2.2 + b * f * d / 2.2 for a in (-1, 0, 1) for b in (-1, 0, 1)]
            S3 = np.array([[p[0], y, p[1]] for p in samp for y in (0.02, 0.9 * h)])
            if seen(cams, S3).any():
                continue
            clash = False
            for p in placed:
                if (cls, p["cls"]) in TUCK_UNDER or (p["cls"], cls) in TUCK_UNDER:
                    continue
                grown = corners(c, ux, f, w + 0.2, d + 0.2)
                if overlap(grown, p["quad"]):
                    clash = True
                    break
            if clash:
                continue
            score = math.log(pw if at_wall else 1 - pw)
            if placed:   # mild preference for open floor, so unrelated pieces spread out
                gap = min(np.linalg.norm(p["c"] - c) for p in placed)
                score += 0.5 * min(gap, 2.5) / 2.5
            near = None
            for acls in {p["cls"] for p in placed}:
                terms = []
                key_fwd, key_rev = f"{acls}|{cls}", f"{cls}|{acls}"
                for p in placed:
                    if p["cls"] != acls:
                        continue
                    t = []
                    if key_fwd in P["pairs"]:
                        dx, dz = local_offset(p, c)
                        t.append(rel_term(P["pairs"][key_fwd], dx, dz, rel_angle(p["front"], f), P.get("pair_range", 3.5)))
                    if key_rev in P["pairs"]:
                        dx, dz = local_offset({"c": c, "front": f}, p["c"])
                        t.append(rel_term(P["pairs"][key_rev], dx, dz, rel_angle(f, p["front"]), P.get("pair_range", 3.5)))
                    if t:
                        terms.append(max(t))
                if terms:
                    tb = max(terms)
                    score += tb
                    if tb > 1.0 and (near is None or tb > near[1]):
                        near = (acls, tb)
            if best is None or score > best[0]:
                best = (score, c, ux, f, at_wall, quad, near)
        if best is None:
            log.append(f"{cls}: no free unseen spot")
            continue
        score, c, ux, f, at_wall, quad, near = best
        i = sum(o["class"] == cls for o in out)
        out.append({
            "id": f"pred_{cls}_{i}", "class": cls,
            "position": [round(float(c[0]), 3), 0.0, round(float(c[1]), 3)],
            "size": [w, h, d], "rotation_y": round(yaw_from_front(f), 1), "collide": True, "model": None,
            "source": {"placed_by": "predicted", "scene_type": stype, "expected_in_unseen": round(expected, 2),
                       "score": round(float(score), 2), "against_wall": bool(at_wall),
                       "near": near[0] if near else None},
        })
        placed.append({"cls": cls, "c": c, "front": f, "quad": quad, "w": w, "d": d, "top": h})
        log.append(f"{cls}: score {score:.2f}" + (f", near {near[0]}" if near else "") + (", against wall" if at_wall else ""))
    return out, {"unseen_area_m2": round(unseen_area, 1), "seen_area_m2": round(float(vis.sum() * GRID ** 2), 1),
                 "log": log, "floor_samples": floor, "visible": vis}


def place_on_top(cls, info, expected, placed, cams, out, stype):
    """Put a table lamp on an unseen piece of furniture that has nothing on it yet, toward
    its back corner. Furniture the camera saw is skipped: a lamp there would have shown."""
    w, h, d = info["size"]
    hosts = []
    for p in placed:
        if p["cls"] not in ON_TOP[cls]:
            continue
        if any(q["cls"] == cls and np.linalg.norm(q["c"] - p["c"]) < 0.5 for q in placed):
            continue
        right = np.array([p["front"][1], -p["front"][0]])
        side = 1 if sum(q["cls"] == cls for q in placed) % 2 == 0 else -1
        c = p["c"] - p["front"] * max(0.0, p["d"] / 2 - d / 2 - 0.03) + right * side * max(0.0, p["w"] / 2 - w / 2 - 0.05)
        y = p["top"]
        S3 = np.array([[c[0], y + 0.02, c[1]], [c[0], y + h, c[1]]])
        if seen(cams, S3).any():
            continue
        hosts.append((ON_TOP[cls].index(p["cls"]), c, y, p))
    if not hosts:
        return None
    _, c, y, p = min(hosts, key=lambda t: t[0])
    i = sum(o["class"] == cls for o in out)
    return {"id": f"pred_{cls}_{i}", "class": cls,
            "position": [round(float(c[0]), 3), round(float(y), 3), round(float(c[1]), 3)],
            "size": [w, h, d], "rotation_y": round(yaw_from_front(p["front"]), 1), "collide": True, "model": None,
            "source": {"placed_by": "predicted", "scene_type": stype, "expected_in_unseen": round(expected, 2),
                       "against_wall": False, "near": p["cls"], "on_top_of": p["cls"]}}


# ------------------------------------------------------------------ viz

def draw(scene, room, info, path):
    import cv2
    rect = np.array(room["corners"])
    allp = np.vstack([rect] + [np.array(e["corners"]) for e in room.get("extensions", [])]
                     + [np.array(c["position"])[[0, 2]][None] for c in scene.get("cameras", [])])
    x0, z0 = allp.min(0) - 0.5
    x1, z1 = allp.max(0) + 0.5
    s = 700 / max(x1 - x0, z1 - z0)
    W, H = int((x1 - x0) * s), int((z1 - z0) * s) + 30
    img = np.full((H, W, 3), 255, np.uint8)
    P = lambda p: (int((p[0] - x0) * s), int((p[1] - z0) * s))  # noqa: E731
    for q in [rect] + [np.array(e["corners"]) for e in room.get("extensions", [])]:
        cv2.fillPoly(img, [np.array([P(p) for p in q])], (225, 225, 225))
    if "floor_samples" in info:
        for p, vv in zip(info["floor_samples"], info["visible"]):
            if vv:
                cv2.circle(img, P(p), 1, (120, 200, 230), -1)
    for w in scene["schematic"].get("observed_outline", []):
        cv2.line(img, P(w["from"]), P(w["to"]), (170, 170, 170), 1)
    for w in scene["schematic"]["walls"]:
        cv2.line(img, P(w["from"]), P(w["to"]), (50, 50, 50), 3)
    for o in scene["objects"]:
        ux, uz = obj_axes(o)
        q = corners(o["position"][::2], ux, uz, o["size"][0], o["size"][2])
        pred = o.get("source", {}).get("placed_by") == "predicted"
        col = (0, 140, 255) if pred else (180, 60, 40)
        cv2.polylines(img, [np.array([P(p) for p in q])], True, col, 2)
        c = np.array(o["position"][::2])
        cv2.arrowedLine(img, P(c), P(c + uz * 0.35), col, 1, tipLength=0.3)
        cv2.putText(img, o["class"], P(c + [0.05, 0.05]), 0, 0.42, (30, 30, 30), 1)
    for c in scene.get("cameras", []):
        cv2.drawMarker(img, P(np.array(c["position"])[[0, 2]]), (200, 0, 200), cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(img, f"{room.get('scene_type')}  blue=seen floor  dark=detected  orange=predicted  arrow=front",
                (6, H - 10), 0, 0.42, (60, 60, 60), 1)
    cv2.imwrite(str(path), img)


# ------------------------------------------------------------------ main

def complete(scene_path, viz=None, do_predict=True):
    scene_path = pathlib.Path(scene_path)
    scene = json.loads(scene_path.read_text())
    scene["objects"] = [o for o in scene["objects"] if o.get("source", {}).get("placed_by") != "predicted"]
    cams = [Cam(c) for c in scene.get("cameras", [])]
    stype, how = scene_type(scene)
    print(f"room type: {stype} ({how['source']}" + (f", {how['probs']}" if "probs" in how else "") + ")")
    if stype is None and not (scene["schematic"].get("room") or {}).get("given"):
        print("not a room type we know: leaving the scene as is")
        scene_path.write_text(json.dumps(scene, indent=2))
        return
    room = complete_room(scene, stype, cams)
    room["scene_type"] = room.get("scene_type") or stype
    room["scene_type_from"] = how
    apply_room(scene, room)
    if "size" in room:
        print(f"room {room['size'][0]} x {room['size'][1]} m, walls seen: "
              f"{[k for k, v in room.get('observed_sides', {}).items() if v] or 'none'}"
              + "".join(f", + {e['size'][0]} x {e['size'][1]} m space past the {e['side']} wall"
                        for e in room.get("extensions", [])))
    if room.get("corners"):
        orient_observed(scene, [room["corners"]] + [e["corners"] for e in room.get("extensions", [])], cams)
    info = {}
    if do_predict and PRIOR.exists() and room.get("corners"):
        preds, info = predict(scene, room, cams, json.loads(PRIOR.read_text()))
        print(f"unseen floor {info.get('unseen_area_m2')} m2 (seen {info.get('seen_area_m2')} m2) -> "
              f"{len(preds)} predicted")
        for line in info.get("log", []):
            print("  " + line)
        scene["objects"] += preds
        room["predicted"] = len(preds)
    scene_path.write_text(json.dumps(scene, indent=2))
    if viz:
        pathlib.Path(viz).mkdir(parents=True, exist_ok=True)
        draw(scene, room, info, pathlib.Path(viz) / f"{scene_path.stem}_room.png")
    print(f"updated {scene_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--viz")
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()
    complete(args.scene, args.viz, not args.no_predict)


if __name__ == "__main__":
    main()
