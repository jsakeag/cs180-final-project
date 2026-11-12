"""SUN RGB-D adapter: a calibrated indoor photo -> schematic + automatic floor
registration + ground-truth objects, so the pipeline runs without clicking and can be
scored.

Data: https://rgbd.cs.princeton.edu (Song, Lichtenberg, Xiao, CVPR 2015). The 6.9 GB
zip is read remotely with HTTP range requests; only the files we need for selected
frames are downloaded (~100 KB per frame).

SUN frame (annotations): gravity-aligned camera frame, X right, Y DOWN, Z forward,
camera at the origin. Ymin of the room layout = floor (so camera height = Ymin).
Engine frame: x right, y up, z toward the viewer. Mapping (a 180 deg rotation about X,
so nothing is mirrored):  x = X,  y = floorY - Y,  z = -Z.

usage:
  python detect/sunrgbd.py select --n 60 --types bedroom living_room
  python detect/sunrgbd.py prepare <frame_id>      # -> detect/scenes/sun_<frame_id>/
"""
import argparse, json, pathlib, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data" / "sunrgbd"
FRAMES = DATA / "frames"
SCENES = ROOT / "scenes"
ZIP_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

# SUN label (before ':') -> our class
SUN_TO_CLASS = {
    "bed": "bed", "chair": "chair", "sofa": "sofa", "sofa_chair": "armchair", "armchair": "armchair",
    "table": "table", "dining_table": "table", "coffee_table": "coffee_table", "endtable": "nightstand",
    "end_table": "nightstand", "side_table": "nightstand", "night_stand": "nightstand", "nightstand": "nightstand",
    "desk": "desk", "dresser": "dresser", "bookshelf": "bookshelf", "bookcase": "bookshelf", "shelf": "bookshelf",
    "lamp": "lamp", "cabinet": "cabinet", "kitchen_cabinet": "cabinet", "file_cabinet": "cabinet",
    "plant": "plant", "stool": "stool", "ottoman": "stool", "bench": "bench", "tv_stand": "tv_stand",
    "basket": "basket", "toilet": "toilet", "bathtub": "bathtub", "fridge": "fridge", "refrigerator": "fridge",
    "wardrobe": "wardrobe", "closet": "wardrobe",
}

_local = threading.local()


def y_bottom(pg):
    """Y points down, but files disagree on which of Ymin/Ymax is the bottom."""
    return max(pg["Ymin"], pg["Ymax"])


def y_top(pg):
    return min(pg["Ymin"], pg["Ymax"])


def _zip():
    if not hasattr(_local, "z"):
        from remotezip import RemoteZip
        _local.z = RemoteZip(ZIP_URL)
    return _local.z


def _read(member, tries=3):
    for i in range(tries):
        try:
            return _zip().read(member)
        except Exception:  # noqa: BLE001
            time.sleep(1 + i)
    raise RuntimeError(f"could not read {member}")


def frame_id(prefix):
    return prefix.strip("/").removeprefix("SUNRGBD/").replace("/", "_")


# ------------------------------------------------------------------ download

def select(n, types, seed=0):
    names = (DATA / "namelist.txt").read_text().split("\n")
    scene_types = json.loads((DATA / "scene_types.json").read_text())
    by_prefix = {}
    for nm in names:
        for key in ("image/", "extrinsics/"):
            i = nm.find("/" + key)
            if i > 0 and not nm.endswith("/"):
                by_prefix.setdefault(nm[:i + 1], {}).setdefault(key, []).append(nm)
    pool = [f for f, t in scene_types.items() if t in types and f in by_prefix]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    print(f"{len(pool)} frames of {types}; downloading {n}")

    def fetch(prefix):
        try:
            return _fetch_frame(prefix)
        except Exception as e:  # noqa: BLE001  (flaky server: skip the frame)
            import shutil
            shutil.rmtree(FRAMES / frame_id(prefix), ignore_errors=True)
            print(f"  skip {frame_id(prefix)}: {e}")
            return None

    def _fetch_frame(prefix):
        out = FRAMES / frame_id(prefix)
        if (out / "meta.json").exists():
            return frame_id(prefix)
        out.mkdir(parents=True, exist_ok=True)
        files = by_prefix[prefix]
        img = sorted(files["image/"])[0]
        ext = sorted(files["extrinsics/"])[-1]
        (out / "image.jpg").write_bytes(_read(img))
        (out / "annotation3Dfinal.json").write_bytes(_read(prefix + "annotation3Dfinal/index.json"))
        (out / "annotation3Dlayout.json").write_bytes(_read(prefix + "annotation3Dlayout/index.json"))
        K = np.array(_read(prefix + "intrinsics.txt").decode().split(), float).reshape(3, 3)
        R = np.array(_read(ext).decode().split(), float).reshape(3, 4)[:, :3]
        meta = {"id": frame_id(prefix), "zip_prefix": prefix, "scene": scene_types[prefix],
                "K": K.tolist(), "Rtilt": R.tolist()}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta["id"]

    ids, i = [], 0
    with ThreadPoolExecutor(8) as ex:
        while len(ids) < n and i < len(pool):
            batch = pool[i:i + (n - len(ids))]
            i += len(batch)
            ids += [r for r in ex.map(fetch, batch) if r]
    (DATA / "selected.json").write_text(json.dumps(sorted(set(ids) | set(_existing_ids())), indent=2))
    print(f"frames in {FRAMES}")
    return ids


def _existing_ids():
    return [p.name for p in FRAMES.iterdir() if (p / "meta.json").exists()] if FRAMES.exists() else []


# ------------------------------------------------------------------ geometry

class Frame:
    def __init__(self, fid):
        self.dir = FRAMES / fid
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.id = fid
        self.K = np.array(self.meta["K"])
        self.R = np.array(self.meta["Rtilt"])
        lay = [o for o in json.loads((self.dir / "annotation3Dlayout.json").read_text())["objects"] if o]
        self.layout = lay[0]["polygon"][0] if lay else None
        self.ann = [o for o in json.loads((self.dir / "annotation3Dfinal.json").read_text())["objects"] if o]
        # Floor height below the camera. The layout's Ymin can be ~10 cm off; the bottoms of
        # floor-standing furniture are measured from the depth data, so they win if lower.
        bottoms = [y_bottom(o["polygon"][0]) for o in self.ann
                   if SUN_TO_CLASS.get(o["name"].split(":")[0].strip().lower()) or o["name"].startswith("floor")]
        lay_y = y_bottom(self.layout) if self.layout else None
        obj_y = float(np.percentile(bottoms, 75)) if bottoms else None
        self.floorY = max(v for v in (lay_y, obj_y, 0.5) if v is not None) if (lay_y or obj_y) else 1.3
        self.ceilY = y_top(self.layout) if self.layout else self.floorY - 2.6
        from PIL import Image
        with Image.open(self.dir / "image.jpg") as im:
            self.w, self.h = im.size

    def project(self, X, Y, Z):
        """SUN world point -> pixel (None if behind the camera)."""
        ctb = self.R.T @ np.array([X, Z, -Y])          # undo tilt; toolbox axes [x, z, -y]
        xc, yc, zc = ctb[0], -ctb[2], ctb[1]
        if zc <= 0.1:
            return None
        u, v, w = self.K @ np.array([xc, yc, zc])
        return np.array([u / w, v / w])

    @staticmethod
    def to_plan(X, Z):
        return [round(float(X), 4), round(float(-Z), 4)]

    def _in_room(self, X, Z, margin=0.2):
        """Point-in-layout-polygon (SUN X, Z), shrunk by a margin via sampling the 4
        neighbours. Without a layout everything counts as inside."""
        if not self.layout:
            return True
        xs, zs = self.layout["X"], self.layout["Z"]

        def inside(x, z):
            c, n = False, len(xs)
            for i in range(n):
                x1, z1, x2, z2 = xs[i], zs[i], xs[(i + 1) % n], zs[(i + 1) % n]
                if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1) + x1:
                    c = not c
            return c
        return all(inside(X + dx, Z + dz) for dx, dz in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)))

    def _occluders(self):
        """Screen rectangle + nearest depth of every annotated object (not walls/floor)."""
        out = []
        for o in self.ann:
            label = o["name"].split(":")[0].strip().lower()
            if label in ("wall", "floor", "ceiling", "window", "door", "room"):
                continue
            pg = o["polygon"][0]
            pts, zs = [], []
            for X, Z in zip(pg["X"], pg["Z"]):
                for Y in (pg["Ymin"], pg["Ymax"]):
                    q = self.project(X, Y, Z)
                    if q is not None:
                        pts.append(q)
                        zs.append(Z)
            if len(pts) >= 2:
                P = np.array(pts)
                out.append((P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max(), min(zs)))
        return out

    def registration(self):
        """Floor points visible in the image, as if clicked perfectly. Points hidden behind
        annotated furniture are dropped: a person clicks visible floor, and place.py
        calibrates the depth model's scale at these pixels."""
        occ = self._occluders()
        cand = []
        for Z in np.linspace(0.8, 8.0, 16):
            for X in np.linspace(-5.0, 5.0, 21):
                p = self.project(X, self.floorY, Z)
                if p is None:
                    continue
                inside = 5 < p[0] < self.w - 5 and 5 < p[1] < self.h - 5
                near_frame = -self.w < p[0] < 2 * self.w and -self.h < p[1] < 2 * self.h
                hidden = (not self._in_room(X, Z)          # beyond a wall: the wall is what's seen
                          or any(x1 <= p[0] <= x2 and y1 <= p[1] <= y2 and z_near < Z - 0.1
                                 for x1, y1, x2, y2, z_near in occ))
                if inside or near_frame:
                    cand.append((inside, inside and not hidden, p, X, Z))
        # H only needs the math to be right, so off-image or hidden points are fine for it;
        # "visible" marks the ones a person could click (used to calibrate depth).
        inside = [c for c in cand if c[0]]
        use = inside if len(inside) >= 6 else cand
        img_pts = [[round(float(p[0]), 2), round(float(p[1]), 2)] for _, _, p, _, _ in use]
        plan_pts = [self.to_plan(X, Z) for _, _, _, X, Z in use]
        visible = [bool(v) for _, v, _, _, _ in use]
        return {"image": str((self.dir / "image.jpg").relative_to(ROOT.parent)),
                "note": "SUN RGB-D: floor points generated from the calibrated camera (no clicks)",
                "K": self.K.tolist(),
                "image_points": img_pts, "plan_points": plan_pts, "visible": visible}

    def schematic(self):
        if self.layout:
            xs, zs = self.layout["X"], self.layout["Z"]
        else:  # no layout: a box around the annotated objects
            pts = [(x, z) for o in self.ann for x, z in zip(o["polygon"][0]["X"], o["polygon"][0]["Z"])]
            xs, zs = [min(p[0] for p in pts) - .5, max(p[0] for p in pts) + .5], [-.3, max(p[1] for p in pts) + .5]
            xs, zs = [xs[0], xs[1], xs[1], xs[0]], [zs[0], zs[0], zs[1], zs[1]]
        plan = [self.to_plan(x, z) for x, z in zip(xs, zs)]
        walls = [{"from": plan[i], "to": plan[(i + 1) % len(plan)], "thickness": 0.12} for i in range(len(plan))]
        px, pz = [p[0] for p in plan], [p[1] for p in plan]
        return {
            "name": f"sun_{self.id}",
            "units": "meters",
            "spawn": {"position": [0, 0, 0], "yaw_deg": 0},     # camera position, facing -z = forward
            "schematic": {"floor": {"min": [min(px), min(pz)], "max": [max(px), max(pz)]},
                          "wall_height": round(float(self.floorY - self.ceilY), 3),
                          "walls": walls},
            "objects": [],
            "dataset": {"name": "SUN RGB-D", "frame": self.meta["zip_prefix"], "scene": self.meta["scene"]},
        }

    def gt_objects(self):
        out = []
        for o in self.ann:
            label = o["name"].split(":")[0].strip().lower()
            cls = SUN_TO_CLASS.get(label)
            if not cls:
                continue
            pg = o["polygon"][0]
            P = np.array([self.to_plan(x, z) for x, z in zip(pg["X"], pg["Z"])])
            if len(P) < 3:
                continue
            c = P.mean(0)
            e1, e2 = P[1] - P[0], P[2] - P[1]
            w, d = np.linalg.norm(e1), np.linalg.norm(e2)
            if d > w:
                (w, d), e1 = (d, w), e2
            yaw = np.degrees(np.arctan2(-e1[1], e1[0]))
            bottom, top = self.floorY - y_bottom(pg), self.floorY - y_top(pg)
            out.append({"class": cls, "sun_label": label,
                        "position": [round(float(c[0]), 3), round(max(0.0, float(bottom)), 3), round(float(c[1]), 3)],
                        "size": [round(float(w), 3), round(float(top - bottom), 3), round(float(d), 3)],
                        "rotation_y": round(float(yaw), 1)})
        return out


def prepare(fid):
    fr = Frame(fid)
    out = SCENES / f"sun_{fid}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "schematic.json").write_text(json.dumps(fr.schematic(), indent=2))
    (out / "registration.json").write_text(json.dumps([fr.registration()], indent=2))
    (out / "gt.json").write_text(json.dumps(fr.gt_objects(), indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--types", nargs="+", default=["bedroom", "living_room"])
    p = sub.add_parser("prepare")
    p.add_argument("frame_id")
    args = ap.parse_args()
    if args.cmd == "select":
        select(args.n, args.types)
    else:
        print(prepare(args.frame_id))


if __name__ == "__main__":
    sys.exit(main())
