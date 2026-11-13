"""Room prior learned from SUN RGB-D 3D annotations: what a room of a given type contains
(per m2 of floor), how big things are, whether they stand against a wall, which way they
face, and where they sit relative to each other (nightstand beside the bed, coffee table
in front of the sofa). complete.py uses it to fill the parts of a room the photo missed.

usage:
  python detect/room_prior.py fetch --per-type 400    # annotations only (~10 KB/frame)
  python detect/room_prior.py learn                   # -> detect/data/room_prior.json

Frames used for evaluation (selected.json) are excluded.
"""
import argparse, json, math, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import sunrgbd  # noqa: E402
from priors import SIZE_PRIORS  # noqa: E402

ANN = sunrgbd.DATA / "ann"
PRIOR = ROOT / "data" / "room_prior.json"
TYPES = ["bedroom", "living_room", "office", "dining_area"]
WALL_GAP = 0.35        # a side this close to a wall (and parallel) is "against the wall"
PAIR_RANGE = 3.5       # relations are only learned between objects this close
MAX_PAIR_SAMPLES = 400
MIN_PEAK = 5.0         # keep a relation only if its offsets cluster this much above uniform
KDE_SIGMA = 0.3
HALF_HFOV = 31.0      # SUN cameras (Kinect / Xtion / RealSense) see about +-31 deg sideways
FRONT_ON_SHORT_SIDE = {"bed"}
# seats face the thing they're used with when one is close (a chair at a desk faces it)
FACE_TOWARD = {"chair": ("desk", "table"), "stool": ("table", "desk"), "armchair": ("coffee_table", "tv_stand"),
               "sofa": ("coffee_table", "tv_stand")}
FACE_RANGE = 1.3   # a bed's front is its foot end even when its long side hugs a wall


# ------------------------------------------------------------------ fetch

def fetch(per_type, seed=0):
    names = set((sunrgbd.DATA / "namelist.txt").read_text().split("\n"))
    types = json.loads((sunrgbd.DATA / "scene_types.json").read_text())
    held_out = set(json.loads((sunrgbd.DATA / "selected.json").read_text()))
    ANN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    todo = []
    for t in TYPES:
        pool = [p for p, tt in types.items() if tt == t and sunrgbd.frame_id(p) not in held_out
                and p + "annotation3Dfinal/index.json" in names and p + "annotation3Dlayout/index.json" in names]
        rng.shuffle(pool)
        todo += pool[:per_type]
        print(f"{t}: {len(pool)} annotated frames, using {min(per_type, len(pool))}")

    def one(prefix):
        out = ANN / f"{sunrgbd.frame_id(prefix)}.json"
        if out.exists():
            return True
        try:
            ann = json.loads(sunrgbd._read(prefix + "annotation3Dfinal/index.json"))
            lay = json.loads(sunrgbd._read(prefix + "annotation3Dlayout/index.json"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {prefix}: {e}")
            return False
        lay = [o for o in lay.get("objects", []) if o and o.get("polygon")]
        out.write_text(json.dumps({"scene": types[prefix], "prefix": prefix,
                                   "objects": [o for o in ann.get("objects", []) if o],
                                   "layout": lay[0]["polygon"][0] if lay else None}))
        return True

    done = 0
    with ThreadPoolExecutor(6) as ex:
        for ok in ex.map(one, todo):
            done += ok
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"{done} frames in {ANN}")


# ------------------------------------------------------------------ geometry shared with complete.py

def polygon_area(P):
    P = np.asarray(P, float)
    return 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))


def real_walls(poly, cam, min_dist=0.4, min_len=0.3):
    """Edges of a SUN layout polygon that are walls. The polygon is the visible part of the
    room, so some edges are just the edges of the view cone; those run through the camera."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % n], float)
        e = b - a
        L = np.linalg.norm(e)
        if L < min_len:
            continue
        dist = abs(e[0] * (cam[1] - a[1]) - e[1] * (cam[0] - a[0])) / L
        if dist > min_dist:
            out.append((a, b))
    return out


def box_sides(c, ux, uz, w, d):
    """4 sides of a footprint: (midpoint, outward normal, length)."""
    return [(c + uz * d / 2, uz, w), (c - uz * d / 2, -uz, w), (c + ux * w / 2, ux, d), (c - ux * w / 2, -ux, d)]


def side_against_wall(mid, normal, walls, gap=WALL_GAP):
    """True if a wall runs parallel to this side just outside it."""
    for a, b in walls:
        e = b - a
        L = np.linalg.norm(e)
        t = e / L
        if abs(np.dot(t, normal)) > math.sin(math.radians(20)):
            continue
        s = np.dot(mid - a, t)
        if s < -0.3 or s > L + 0.3:
            continue
        d_out = np.dot(a + s * t - mid, normal)     # how far outside this side the wall is
        if -0.15 < d_out < gap:
            return True
    return False


def footprint(corners, h, cls, walls, cam):
    """SUN 4-corner footprint -> center, size, and facing. The front is the side away from
    a wall when the object stands against one; otherwise the long side that faces the
    camera's side of the room (people photograph the fronts of things)."""
    P = np.asarray(corners, float)
    c = P.mean(0)
    e1, e2 = P[1] - P[0], P[2] - P[1]
    sides = [(e1, np.linalg.norm(e1)), (e2, np.linalg.norm(e2))]
    against = None
    for e, L in sides:
        if cls in FRONT_ON_SHORT_SIDE and L < max(s[1] for s in sides):
            continue                      # front must be a short side: width = the short edge
        ux = e / L
        uz = np.array([-ux[1], ux[0]])
        other = sides[1][1] if e is sides[0][0] else sides[0][1]
        for sgn in (1, -1):
            if side_against_wall(c - sgn * uz * other / 2, -sgn * uz, walls):
                cand = (L, ux * 1, sgn * uz, other)
                if against is None or cand[0] > against[0]:
                    against = cand
    if against:
        w, ux, front, d = against
        facing_from = "wall"
    else:
        (e, w), (_, d) = sorted(sides, key=lambda s: s[1] if cls in FRONT_ON_SHORT_SIDE else -s[1])
        ux = e / w
        front = np.array([-ux[1], ux[0]])
        if np.dot(front, np.asarray(cam) - c) < 0:
            front = -front
        facing_from = "camera"
    return {"cls": cls, "c": c, "w": float(w), "d": float(d), "h": float(h),
            "front": front, "against_wall": against is not None, "facing_from": facing_from}


def face_toward(boxes):
    """Re-aim seats at their table (in place). Facing snaps to the box's own axes."""
    for b in boxes:
        targets = [a for a in boxes if a["cls"] in FACE_TOWARD.get(b["cls"], ())]
        if not targets:
            continue
        a = min(targets, key=lambda a: np.linalg.norm(a["c"] - b["c"]))
        d = a["c"] - b["c"]
        if np.linalg.norm(d) > FACE_RANGE + max(a["w"], a["d"]) / 2:
            continue
        f = b["front"]
        axes = [f, -f, np.array([f[1], -f[0]]), np.array([-f[1], f[0]])]
        if max(b["w"], b["d"]) > 1.3 * min(b["w"], b["d"]):
            axes = axes[:2]           # long things (sofa) only flip; a side table mustn't turn them
        b["front"] = max(axes, key=lambda x: np.dot(x, d))
        b["facing_from"] = "faces " + a["cls"]


def local_offset(anchor, p):
    """p in the anchor's frame: x to the anchor's right, z toward its front."""
    f = anchor["front"]
    right = np.array([f[1], -f[0]])
    v = np.asarray(p) - anchor["c"]
    return float(np.dot(v, right)), float(np.dot(v, f))


def rel_angle(f_anchor, f_obj):
    """Facing of obj relative to anchor, degrees in (-180, 180]."""
    a = math.atan2(f_obj[1], f_obj[0]) - math.atan2(f_anchor[1], f_anchor[0])
    return (math.degrees(a) + 180) % 360 - 180


# ------------------------------------------------------------------ learn

def in_poly(P, poly):
    """Vectorized point-in-polygon (even-odd)."""
    P, poly = np.atleast_2d(P), np.asarray(poly, float)
    x, z = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    for (x1, z1), (x2, z2) in zip(poly, np.roll(poly, -1, 0)):
        cross = (z1 > z) != (z2 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (x2 - x1) * (z - z1) / (z2 - z1) + x1
        inside ^= cross & (x < xi)
    return inside


def in_view(P, slack=0.0):
    """Inside the camera's horizontal field of view (SUN frame: camera at the origin
    looking along plan -z)."""
    P = np.atleast_2d(P)
    return (-P[:, 1] > 0.1) & (np.degrees(np.abs(np.arctan2(P[:, 0], -P[:, 1]))) < HALF_HFOV + slack)


def seen_area(poly, step=0.1):
    """Floor area of the room layout that falls inside the view: objects were annotated
    only here, so per-m2 rates are counted over this area, not the whole layout."""
    P = np.asarray(poly, float)
    lo, hi = P.min(0), P.max(0)
    g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)), -1).reshape(-1, 2)
    return float((in_poly(g, P) & in_view(g)).sum() * step * step)


def frame_boxes(fr):
    poly = [[x, -z] for x, z in zip(fr["layout"]["X"], fr["layout"]["Z"])]
    cam = np.zeros(2)
    walls = real_walls(poly, cam)
    boxes = []
    for o in fr["objects"]:
        cls = sunrgbd.SUN_TO_CLASS.get(o.get("name", "").split(":")[0].strip().lower())
        if cls not in SIZE_PRIORS:
            continue
        if not o.get("polygon"):
            continue
        pg = o["polygon"][0]
        if len(pg.get("X", [])) < 3:
            continue
        corners = [[x, -z] for x, z in zip(pg["X"], pg["Z"])]
        b = footprint(corners, abs(pg["Ymax"] - pg["Ymin"]), cls, walls, cam)
        if 0.05 < b["w"] < 5 and 0.05 < b["d"] < 5:
            boxes.append(b)
    face_toward(boxes)
    return poly, walls, boxes


def peakedness(S):
    """How clustered a relation's offsets are: mean leave-one-out kernel density at the
    samples, relative to uniform over the PAIR_RANGE disc. ~1-3 = objects merely share a
    room (sofa / chair); 10-50 = a real arrangement (lamp on nightstand, nightstand by bed)."""
    S = np.asarray(S)[:, :2]
    d = ((S[:, None] - S[None]) ** 2).sum(-1)
    k = np.exp(-d / (2 * KDE_SIGMA ** 2))
    np.fill_diagonal(k, 0)
    dens = (k.sum(1) / (len(S) - 1)).mean() / (2 * math.pi * KDE_SIGMA ** 2)
    return float(dens * math.pi * PAIR_RANGE ** 2)


def learn():
    rng = np.random.default_rng(0)
    stats = {}
    for f in sorted(ANN.glob("*.json")):
        fr = json.loads(f.read_text())
        if not fr.get("layout") or fr["scene"] not in TYPES:
            continue
        poly, walls, boxes = frame_boxes(fr)
        area = seen_area(poly)
        if not 1.0 < area < 60:
            continue
        S = stats.setdefault(fr["scene"], {"frames": 0, "area": 0.0, "cls": {}, "pairs": {}})
        S["frames"] += 1
        S["area"] += area
        counts = {}
        for b in boxes:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
            C = S["cls"].setdefault(b["cls"], {"n": 0, "n_view": 0, "frames": 0, "counts": [], "wall": 0, "sizes": []})
            C["n"] += 1
            C["n_view"] += bool(in_view(b["c"], slack=5)[0])
            C["wall"] += b["against_wall"]
            C["sizes"].append([b["w"], b["h"], b["d"]])
        for cls, k in counts.items():
            S["cls"][cls]["frames"] += 1
            S["cls"][cls]["counts"].append(k)
        # relations: each object relative to the nearest instance of every other class
        for b in boxes:
            for acls in {a["cls"] for a in boxes}:
                if acls == b["cls"]:
                    continue
                anchors = [a for a in boxes if a["cls"] == acls]
                a = min(anchors, key=lambda a: np.linalg.norm(a["c"] - b["c"]))
                if np.linalg.norm(a["c"] - b["c"]) > PAIR_RANGE:
                    continue
                dx, dz = local_offset(a, b["c"])
                key = f"{acls}|{b['cls']}"
                S["pairs"].setdefault(key, []).append([round(dx, 2), round(dz, 2), round(rel_angle(a["front"], b["front"]), 1)])

    out = {}
    for t, S in stats.items():
        cls_out = {}
        for c, C in S["cls"].items():
            if C["n"] < 5:
                continue
            cls_out[c] = {
                "rate_per_m2": round(C["n_view"] / S["area"], 4),    # Poisson rate over seen floor
                "frames_present": round(C["frames"] / S["frames"], 3),
                # how many a whole room holds: a view rarely shows all of them, so +1 unless
                # the class is almost always alone (one bed per bedroom)
                "max_count": int(np.percentile(C["counts"], 90)) + int(np.mean(np.array(C["counts"]) == 1) < 0.92),
                "p_wall": round(C["wall"] / C["n"], 3),
                "size": [round(float(v), 2) for v in np.median(C["sizes"], 0)],
                "n": C["n"],
            }
        pairs, strength = {}, {}
        for k, v in S["pairs"].items():
            a, b = k.split("|")
            if a in cls_out and b in cls_out and len(v) >= 15:
                v = np.array(v)
                if len(v) > MAX_PAIR_SAMPLES:
                    v = v[rng.choice(len(v), MAX_PAIR_SAMPLES, replace=False)]
                pk = peakedness(v)
                if pk >= MIN_PEAK:
                    pairs[k] = v.tolist()
                    strength[k] = round(pk, 1)
        out[t] = {"frames": S["frames"], "seen_area_m2": round(S["area"], 1), "classes": cls_out, "pairs": pairs,
                  "pair_strength": strength, "pair_range": PAIR_RANGE}
        print(f"{t:<12} {S['frames']:4d} frames  {S['area'] / S['frames']:.1f} m2 seen/frame  "
              f"{len(cls_out)} classes  {len(pairs)} relations: "
              + ", ".join(f"{k} {v}" for k, v in sorted(strength.items(), key=lambda kv: -kv[1])[:6]))
        for c, v in sorted(cls_out.items(), key=lambda kv: -kv[1]["rate_per_m2"])[:8]:
            print(f"   {c:<13} {v['rate_per_m2']:.3f}/m2  in {v['frames_present']:.0%} of views  "
                  f"wall {v['p_wall']:.0%}  max {v['max_count']}  size {v['size']}")
    PRIOR.write_text(json.dumps(out))
    print(f"-> {PRIOR}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--per-type", type=int, default=400)
    sub.add_parser("learn")
    args = ap.parse_args()
    fetch(args.per_type) if args.cmd == "fetch" else learn()


if __name__ == "__main__":
    main()
