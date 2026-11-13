"""Outdoor schematic of a campus from OpenStreetMap: the outdoor counterpart of a room
layout. Buildings become extruded blocks (height from OSM height / levels), paths and
lawns become ground surfaces, and mapped street furniture (trees, benches, lamps, bins,
bike racks, fountains) becomes placed objects. Photos of the area can later be
registered to it like rooms are (ground-plane homography).

Coordinates: local meters around the bbox center; engine x = east, z = -north (so the
default facing, -z, is north).

usage:
  python detect/campus.py fetch berkeley_core 37.8700 -122.2620 37.8745 -122.2560
  python detect/campus.py build berkeley_core       # -> engine/scenes/campus_berkeley_core.json

Map data (c) OpenStreetMap contributors, ODbL.
"""
import argparse, json, math, pathlib, sys, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OSM = ROOT / "data" / "osm"
ENGINE = ROOT.parent / "engine"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

LEVEL_H = 3.6           # m per storey (campus buildings run tall)
DEFAULT_H = 10.0
PATH_W = {"footway": 2.0, "path": 1.5, "pedestrian": 4.0, "steps": 2.5, "service": 4.5}
# OSM street furniture -> engine class, [w, h, d]
FURNITURE = {
    ("natural", "tree"): ("tree", [0.5, 8.0, 0.5]),          # collider is the trunk
    ("amenity", "bench"): ("bench", [1.8, 0.85, 0.6]),
    ("highway", "street_lamp"): ("lamp_post", [0.3, 4.5, 0.3]),
    ("amenity", "waste_basket"): ("trash_can", [0.5, 1.0, 0.5]),
    ("amenity", "bicycle_parking"): ("bike_rack", [2.0, 0.9, 0.6]),
    ("amenity", "drinking_water"): ("fountain", [0.5, 1.0, 0.5]),
}


def fetch(name, s, w, n, e):
    bb = f"({s},{w},{n},{e})"
    q = f"""[out:json][timeout:90];
(
  way["building"]{bb};
  relation["building"]{bb};
  way["highway"~"footway|path|pedestrian|service|steps"]{bb};
  way["landuse"="grass"]{bb};
  way["leisure"~"park|garden|pitch"]{bb};
  node["natural"="tree"]{bb};
  node["amenity"~"bench|waste_basket|bicycle_parking|drinking_water"]{bb};
  node["highway"="street_lamp"]{bb};
);
out body; >; out skel qt;"""
    import time
    data, err = None, None
    for attempt in range(6):          # public Overpass servers are busy: retry, alternate mirrors
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "cs180-scene-builder/0.1"})
            data = json.load(urllib.request.urlopen(req, timeout=120))
            break
        except Exception as ex:  # noqa: BLE001
            err = ex
            print(f"  {url}: {ex}; retrying")
            time.sleep(5 * (attempt + 1))
    if data is None:
        raise SystemExit(f"Overpass unavailable: {err}")
    OSM.mkdir(parents=True, exist_ok=True)
    data["bbox"] = [s, w, n, e]
    (OSM / f"{name}.json").write_text(json.dumps(data))
    print(f"{len(data['elements'])} elements -> {OSM / f'{name}.json'}")


def building_height(t):
    for k in ("height", "building:height"):
        if k in t:
            try:
                return float(str(t[k]).split()[0].replace("m", ""))
            except ValueError:
                pass
    if "building:levels" in t:
        try:
            return float(t["building:levels"]) * LEVEL_H + 1.0
        except ValueError:
            pass
    return DEFAULT_H


def ring_from_ways(way_ids, ways):
    """Join a relation's outer ways into closed rings."""
    segs = [list(ways[i]) for i in way_ids if i in ways]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for k, sg in enumerate(segs):
                if sg[0] == ring[-1]:
                    ring += sg[1:]
                elif sg[-1] == ring[-1]:
                    ring += sg[::-1][1:]
                elif sg[-1] == ring[0]:
                    ring = sg[:-1] + ring
                elif sg[0] == ring[0]:
                    ring = sg[::-1][:-1] + ring
                else:
                    continue
                segs.pop(k)
                changed = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def build(name):
    data = json.loads((OSM / f"{name}.json").read_text())
    s, w, n, e = data["bbox"]
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    kx = 111320 * math.cos(math.radians(lat0))
    to_xz = lambda lat, lon: [round((lon - lon0) * kx, 2), round(-(lat - lat0) * 110540, 2)]  # noqa: E731
    nodes = {el["id"]: el for el in data["elements"] if el["type"] == "node"}
    ways = {el["id"]: el.get("nodes", []) for el in data["elements"] if el["type"] == "way"}
    pts = lambda ids: [to_xz(nodes[i]["lat"], nodes[i]["lon"]) for i in ids if i in nodes]  # noqa: E731

    blocks, surfaces, objects = [], {"grass": [], "path": [], "steps": [], "plaza": []}, []
    used_ways = set()
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "relation" and "building" in t:
            outer = [m["ref"] for m in el.get("members", []) if m["type"] == "way" and m.get("role") == "outer"]
            used_ways |= set(outer)
            for ring in ring_from_ways(outer, ways):
                blocks.append({"name": t.get("name"), "polygon": pts(ring[:-1]), "height": building_height(t)})
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "way":
            ids = el.get("nodes", [])
            closed = len(ids) >= 4 and ids[0] == ids[-1]
            if "building" in t and closed and el["id"] not in used_ways:
                blocks.append({"name": t.get("name"), "polygon": pts(ids[:-1]), "height": building_height(t)})
            elif t.get("landuse") == "grass" or t.get("leisure") in ("park", "garden", "pitch"):
                if closed:
                    surfaces["grass"].append(pts(ids[:-1]))
            elif t.get("highway") in PATH_W:
                hw = t["highway"]
                if closed and (t.get("area") == "yes" or hw == "pedestrian"):
                    surfaces["plaza"].append(pts(ids[:-1]))
                else:
                    kind = "steps" if hw == "steps" else "path"
                    surfaces[kind].append({"line": pts(ids), "width": PATH_W[hw]})
        elif el["type"] == "node" and t:
            for (k, v), (cls, size) in FURNITURE.items():
                if t.get(k) == v:
                    objects.append({"id": f"{cls}_{len([o for o in objects if o['class'] == cls])}", "class": cls,
                                    "position": [*to_xz(el["lat"], el["lon"])[:1], 0.0, to_xz(el["lat"], el["lon"])[1]],
                                    "size": size, "rotation_y": 0.0, "collide": True, "model": None,
                                    "source": {"placed_by": "osm", "osm_id": el["id"]}})

    half = [(e - w) * kx / 2, (n - s) * 110540 / 2]
    tower = next((b for b in blocks if b["name"] and "Campanile" in b["name"]), None)
    if tower:   # stand on the esplanade south of the Campanile, facing it
        cx = sum(p[0] for p in tower["polygon"]) / len(tower["polygon"])
        cz = sum(p[1] for p in tower["polygon"]) / len(tower["polygon"])
        spawn = {"position": [round(cx, 2), 0, round(cz + 30, 2)], "yaw_deg": 0}
    else:
        spawn = {"position": [0, 0, 0], "yaw_deg": 0}
    scene = {
        "name": f"campus_{name}", "units": "meters", "outdoor": True, "spawn": spawn,
        "schematic": {"floor": {"min": [-half[0], -half[1]], "max": [half[0], half[1]]},
                      "wall_height": 0, "walls": [], "blocks": blocks, "surfaces": surfaces},
        "objects": objects,
        "dataset": {"name": "OpenStreetMap", "bbox": data["bbox"], "origin_latlon": [lat0, lon0]},
        "credits": ["Map data (c) OpenStreetMap contributors, ODbL 1.0"],
    }
    out = ENGINE / "scenes" / f"campus_{name}.json"
    out.write_text(json.dumps(scene))
    by = {}
    for o in objects:
        by[o["class"]] = by.get(o["class"], 0) + 1
    print(f"{len(blocks)} buildings ({sum(1 for b in blocks if b['name'])} named), "
          f"{len(surfaces['path'])} paths, {len(surfaces['steps'])} steps, {len(surfaces['grass'])} lawns, "
          f"{len(surfaces['plaza'])} plazas, objects {by}; area {2 * half[0]:.0f} x {2 * half[1]:.0f} m -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    b = sub.add_parser("build")
    b.add_argument("name")
    args = ap.parse_args()
    fetch(args.name, *args.bbox) if args.cmd == "fetch" else build(args.name)


if __name__ == "__main__":
    sys.exit(main())
