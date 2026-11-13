"""Furnish the ground floor of enterable campus buildings with the learned room prior.

The prior (room_prior.py, from SUN RGB-D annotations) knows per room type how common
each class is per m2, which classes hug walls, and where classes sit relative to each
other. Here that becomes a layout:

  units     an anchor (table, desk, sofa, bed) with its satellites (chairs, stools,
            armchairs, nightstands, coffee table) at the *modes* of the learned relative
            offsets: the densest spots in the training samples, not hand-placed seats.
            Satellites per anchor = ratio of their learned rates (chairs per table).
  rows      units are packed in rows along the building's main axis, facing away from the
            main entrance (lecture-hall style), clear of walls and doorways.
  walls     classes that stand against walls in >= 70% of examples (bookshelves,
            cabinets) line the walls, facing in, skipping doorways.

Room type comes from the building's name (library -> library, else classroom) unless
given. Objects get source.placed_by = "interior".

usage:
  python detect/interior.py furnish --building "Doe Memorial Library"
  python detect/interior.py furnish --new          # every enterable building without an interior
"""
import argparse, json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from room_prior import PRIOR, in_poly, polygon_area  # noqa: E402

DEFAULT_SCENE = ROOT.parent / "engine" / "scenes" / "campus_berkeley_core.json"
ANCHORS = ("table", "desk", "sofa", "bed")
SATELLITES = ("chair", "stool", "armchair", "nightstand", "coffee_table")
WALL_P = 0.5              # classes against a wall this often line the walls (their wall share of them)
AISLE = (1.0, 1.3)        # m between units: along a row, between rows
DOOR_CLEAR = 3.0          # m kept free inside each doorway
WALL_BAND = 1.2           # m along the walls kept for wall furniture
MAX_OBJECTS = 140
WALL_CLASSES = ("bookshelf", "cabinet", "dresser", "wardrobe", "tv_stand")


def room_type(block, prior):
    name = (block.get("name") or "").lower()
    if "library" in name and "library" in prior:
        return "library"
    return "classroom" if "classroom" in prior else "office"


def density_peaks(samples, k, sigma=0.3, min_sep=0.6):
    """The k densest, mutually separated sample positions (dx, dz) and their typical
    relative facing."""
    S = np.asarray(samples, float)
    d2 = ((S[:, None, :2] - S[None, :, :2]) ** 2).sum(-1)
    dens = np.exp(-d2 / (2 * sigma ** 2)).sum(1)
    peaks = []
    for i in np.argsort(-dens):
        if all(np.hypot(*(S[i, :2] - S[j, :2])) > min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) == k:
            break
    out = []
    for i in peaks:
        near = S[np.hypot(S[:, 0] - S[i, 0], S[:, 1] - S[i, 1]) < 0.4]
        a = np.radians(near[:, 2])
        out.append((float(S[i, 0]), float(S[i, 1]), float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))))
    return out


def unit_templates(P):
    """anchor -> list of (class, dx, dz, dyaw) satellites, from the learned relations."""
    out = {}
    for anchor in ANCHORS:
        if anchor not in P["classes"]:
            continue
        sats = []
        for key, samples in P["pairs"].items():
            a, b = key.split("|")
            if a != anchor or b not in SATELLITES or P["pair_strength"].get(key, 0) < 6:
                continue
            ratio = P["classes"][b]["rate_per_m2"] / max(1e-6, P["classes"][anchor]["rate_per_m2"])
            n = int(np.clip(round(ratio), 1, 6)) if b in ("chair", "stool") else 1
            if anchor == "bed" and b == "nightstand":
                n = 2
            sats += [(b, *pk) for pk in density_peaks(samples, n)]
        out[anchor] = sats
    return out


def rot(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def yaw_from_front(f):
    return round(math.degrees(math.atan2(f[0], f[1])), 1)


def edge_distance(q, P):
    """Distance from point(s) q (N, 2) to the polygon outline."""
    q = np.atleast_2d(q)
    best = np.full(len(q), np.inf)
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        ab = b - a
        t = np.clip(((q - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
    return best


def furnish(scene, bi, rtype=None, max_objects=MAX_OBJECTS):
    prior_all = json.loads(PRIOR.read_text())
    block = scene["schematic"]["blocks"][bi]
    rtype = rtype or room_type(block, prior_all)
    P = prior_all[rtype]
    poly = np.asarray(block["polygon"], float)
    area = polygon_area(poly)
    scene["objects"] = [o for o in scene["objects"]
                        if not (o.get("source", {}).get("placed_by") == "interior" and o["source"].get("block") == bi)]
    doors = [((np.array(d["from"]) + np.array(d["to"])) / 2) for d in block.get("doors", [])]
    # axes = dominant wall direction (length-weighted, mod 90 deg); units face away from
    # the main (first) doorway
    c0 = poly.mean(0)
    z = sum(np.linalg.norm(poly[(k + 1) % len(poly)] - poly[k]) * np.exp(4j * math.atan2(*(poly[(k + 1) % len(poly)] - poly[k])[::-1]))
            for k in range(len(poly)))
    phi = np.angle(z) / 4
    u, v = np.array([math.cos(phi), math.sin(phi)]), np.array([-math.sin(phi), math.cos(phi)])
    main = doors[0] if doors else c0 + v
    front = -v if (main - c0) @ v > 0 else v
    if abs((main - c0) @ u) > abs((main - c0) @ v):
        front = -u if (main - c0) @ u > 0 else u
    right = np.array([front[1], -front[0]])
    near_door = lambda c: min((np.linalg.norm(c - d) for d in doors), default=0.0)  # noqa: E731

    placed = []    # (center, radius) of everything placed, for spacing
    objs = []

    def free(c, r, band=0.6):
        if not in_poly(c, poly)[0] or edge_distance(c, poly)[0] < r + band:
            return False
        if any(np.linalg.norm(c - d) < DOOR_CLEAR + r for d in doors):
            return False
        return all(np.linalg.norm(c - pc) >= r + pr + 0.3 for pc, pr in placed)

    def add(cls, c, f, info):
        wdt, hgt, dpt = info["size"]
        objs.append({"id": f"int{bi}_{cls}_{sum(o['class'] == cls for o in objs)}", "class": cls,
                     "position": [round(float(c[0]), 3), 0.0, round(float(c[1]), 3)], "size": [wdt, hgt, dpt],
                     "rotation_y": yaw_from_front(f), "collide": True, "model": None,
                     "source": {"placed_by": "interior", "block": bi, "building": block.get("name"), "room_type": rtype}})

    # --- wall furniture
    # wall furniture, nearest the entrance first (the part of a big hall you walk into)
    budget_wall = max_objects // 3
    wall_cls = [c for c in WALL_CLASSES if c in P["classes"] and P["classes"][c]["p_wall"] >= WALL_P]
    want = {c: int(min(budget_wall, round(P["classes"][c]["rate_per_m2"] * P["classes"][c]["p_wall"] * area))) for c in wall_cls}
    total = sum(want.values())
    if total > budget_wall:
        want = {c: int(round(n * budget_wall / total)) for c, n in want.items()}
    queue = [c for c in sorted(want, key=lambda c: -want[c]) for _ in range(want[c])]
    n_edges = len(poly)
    area_sign = np.sign(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    slots = []                                   # (distance to door, a, t, n_in, s) every 0.5 m of wall
    for k in range(n_edges):
        a, b = poly[k], poly[(k + 1) % n_edges]
        L = np.linalg.norm(b - a)
        t = (b - a) / L
        n_in = -area_sign * np.array([t[1], -t[0]])
        for s in np.arange(0.8, L - 0.8, 0.5):
            slots.append((near_door(a + t * s), k, a, t, n_in, s, L))
    slots.sort(key=lambda x: x[0])
    for _, k, a, t, n_in, s, L in slots:
        if not queue:
            break
        cls = queue[0]
        info = P["classes"][cls]
        wdt, _, dpt = info["size"]
        if s + wdt > L - 0.5:
            continue
        c = a + t * (s + wdt / 2) + n_in * (0.2 + 0.25 + dpt / 2)   # half the wall + a gap
        if (in_poly(c, poly)[0] and all(np.linalg.norm(c - d) > DOOR_CLEAR for d in doors)
                and all(np.linalg.norm(c - pc) >= 0.15 + pr for pc, pr in placed)):
            add(cls, c, n_in, info)
            placed.append((c, max(wdt, dpt) / 2))
            queue.pop(0)

    # --- units in rows
    templates = unit_templates(P)
    anchors = [a for a in templates if a != "bed" or rtype == "bedroom"]
    weight = {a: P["classes"][a]["rate_per_m2"] for a in anchors}
    # the share of shelf-like classes that does NOT stand against walls stands free, in rows
    # (library stacks): they join the units with no satellites
    for c in WALL_CLASSES:
        info = P["classes"].get(c)
        if info and info["p_wall"] < 0.9 and info["rate_per_m2"] * (1 - info["p_wall"]) * area >= 1:
            templates[c] = []
            anchors.append(c)
            weight[c] = info["rate_per_m2"] * (1 - info["p_wall"])
    if anchors:
        weights = np.array([weight[a] for a in anchors])
        weights = weights / weights.sum()
        # footprint radius of a unit (anchor + satellites)
        radius = {}
        for a in anchors:
            wdt, _, dpt = P["classes"][a]["size"]
            r = math.hypot(wdt, dpt) / 2
            for cls, dx, dz, _ in templates[a]:
                sw, _, sd = P["classes"][cls]["size"]
                r = max(r, math.hypot(dx, dz) + max(sw, sd) / 2)
            radius[a] = r
        pitch_u = 2 * max(radius.values()) + AISLE[0]
        pitch_v = 2 * max(radius.values()) + AISLE[1]
        proj_u, proj_v = (poly - c0) @ right, (poly - c0) @ front
        rng = np.random.default_rng(bi)
        order = []
        for jv in np.arange(proj_v.min(), proj_v.max(), pitch_v):
            for ju in np.arange(proj_u.min(), proj_u.max(), pitch_u):
                order.append(c0 + right * (ju + pitch_u / 2) + front * (jv + pitch_v / 2))
        order.sort(key=near_door)                # furnish outward from the entrance
        for c in order:
            if len(objs) >= max_objects:
                break
            a = anchors[int(rng.choice(len(anchors), p=weights))]
            if not free(c, radius[a], band=WALL_BAND):
                continue
            # anchors face front, so seats (facing the anchor) end up facing the back.
            # Lecture-hall feel: turn the unit so its satellites face the front instead.
            f = -front if a in ("table", "desk") else front
            info = P["classes"][a]
            add(a, c, f, info)
            r_f = np.array([f[1], -f[0]])
            for cls, dx, dz, dyaw in templates[a]:
                if len(objs) >= max_objects:
                    break
                sc = c + r_f * dx + f * dz
                add(cls, sc, rot(f, dyaw), P["classes"][cls])
            placed.append((c, radius[a]))
    scene["objects"] += objs
    block["interior"] = {"type": rtype, "objects": len(objs)}
    by = {}
    for o in objs:
        by[o["class"]] = by.get(o["class"], 0) + 1
    print(f"{block.get('name')}: {rtype}, {area:.0f} m2 -> {len(objs)} objects {by}")
    return len(objs)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("furnish")
    f.add_argument("--scene", default=str(DEFAULT_SCENE))
    f.add_argument("--building", help="name substring or block index")
    f.add_argument("--new", action="store_true", help="every enterable building without an interior")
    f.add_argument("--type", help="room type override (classroom, library, office, ...)")
    args = ap.parse_args()
    sp = pathlib.Path(args.scene)
    scene = json.loads(sp.read_text())
    blocks = scene["schematic"]["blocks"]
    if args.building is not None:
        from paint import find_block
        targets = [find_block(scene, args.building)[0]]
    else:
        targets = [i for i, b in enumerate(blocks) if b.get("doors") and (not args.new or "interior" not in b)]
    for bi in targets:
        furnish(scene, bi, args.type)
    sp.write_text(json.dumps(scene))


if __name__ == "__main__":
    main()
