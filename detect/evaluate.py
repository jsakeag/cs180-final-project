"""Score placed objects against SUN RGB-D ground truth.

usage:
  python detect/evaluate.py one engine/scenes/sun_<id>.json detect/scenes/sun_<id>/gt.json
  python detect/evaluate.py batch [--n 40] [--skip-detect]     # all selected frames

Matching: a prediction and a GT object match if their classes are in the same group
(priors.MATCH_GROUPS) and their floor centers are within MATCH_DIST; greedy by distance.
Reported:
  precision / recall    over floor-standing classes we place
  center error          floor-plane distance between matched centers (m)
  size error            mean |log(pred / gt)| over w, h, d (0.1 ~ 10% off)
  yaw error             facing difference mod 180 (front/back is ambiguous for boxes)

Registration comes from the calibrated camera, i.e. as if floor points were clicked
perfectly, so this measures detection + placement, not clicking.
"""
import argparse, json, math, pathlib, subprocess, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from priors import MATCH_GROUPS, FLOOR_CLASSES  # noqa: E402

MATCH_DIST = 1.0


def group_of(cls):
    return next((g for g in MATCH_GROUPS if cls in g), (cls,))


def score(pred, gt):
    gt = [g for g in gt if g["class"] in FLOOR_CLASSES]
    pred = [p for p in pred if p["class"] in FLOOR_CLASSES]
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if g["class"] in group_of(p["class"]):
                d = math.dist(p["position"][::2], g["position"][::2])
                if d < MATCH_DIST:
                    pairs.append((d, i, j))
    used_p, used_g, matches = set(), set(), []
    for d, i, j in sorted(pairs):
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        p, g = pred[i], gt[j]
        size_err = float(np.mean(np.abs(np.log(np.maximum(p["size"], 1e-3)) - np.log(np.maximum(g["size"], 1e-3)))))
        # footprint axes may be swapped between pred and gt; take the better ordering
        ps = np.array(p["size"])[[2, 1, 0]]
        size_err = min(size_err, float(np.mean(np.abs(np.log(np.maximum(ps, 1e-3)) - np.log(np.maximum(g["size"], 1e-3))))))
        dyaw = abs((p["rotation_y"] - g["rotation_y"] + 90) % 180 - 90)
        dyaw = min(dyaw, 90 - dyaw)  # width/depth may be swapped between pred and gt: 90 deg ~ 0
        matches.append({"pred": p.get("id", p["class"]), "gt": g["sun_label"], "class": g["class"],
                        "center_err": round(d, 3), "size_err": round(size_err, 3), "yaw_err": round(dyaw, 1),
                        "placed_by": p.get("source", {}).get("placed_by")})
    return {"n_pred": len(pred), "n_gt": len(gt), "n_match": len(matches), "matches": matches,
            "missed": [g["sun_label"] for j, g in enumerate(gt) if j not in used_g],
            "false_pos": [p["class"] for i, p in enumerate(pred) if i not in used_p]}


def summarize(results):
    allm = [m for r in results for m in r["matches"]]
    tp, npred, ngt = len(allm), sum(r["n_pred"] for r in results), sum(r["n_gt"] for r in results)
    out = {"frames": len(results), "gt": ngt, "pred": npred, "matched": tp,
           "precision": round(tp / npred, 3) if npred else None,
           "recall": round(tp / ngt, 3) if ngt else None}
    if allm:
        ce = np.array([m["center_err"] for m in allm])
        out |= {"center_err_median_m": round(float(np.median(ce)), 3),
                "center_err_mean_m": round(float(ce.mean()), 3),
                "size_err_mean_log": round(float(np.mean([m["size_err"] for m in allm])), 3),
                "yaw_err_median_deg": round(float(np.median([m["yaw_err"] for m in allm])), 1)}
        for how in ("floor", "depth"):
            sub = [m["center_err"] for m in allm if m["placed_by"] == how]
            if sub:
                out[f"center_err_median_{how}"] = round(float(np.median(sub)), 3)
                out[f"n_{how}"] = len(sub)
        per = {}
        for m in allm:
            per.setdefault(m["class"], []).append(m["center_err"])
        out["per_class_center_err_median"] = {k: [len(v), round(float(np.median(v)), 3)] for k, v in sorted(per.items())}
    return out


def run(cmd):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def batch(n, skip_detect):
    import sunrgbd
    py = str(REPO / ".venv/bin/python")
    ids = json.loads((sunrgbd.DATA / "selected.json").read_text())[:n]
    out = ROOT / "out" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for fid in ids:
        sunrgbd.prepare(fid)
    dets = out / "detections.json"
    if not skip_detect or not dets.exists():
        imgs = [str(sunrgbd.FRAMES / fid / "image.jpg") for fid in ids]
        print(f"detecting on {len(imgs)} frames ...")
        run([py, "detect/detect.py", *imgs, "-o", str(dets), "--depth"])
    results = []
    for fid in ids:
        sd = ROOT / "scenes" / f"sun_{fid}"
        scene_out = out / f"sun_{fid}.json"
        run([py, "detect/place.py", "--detections", str(dets), "--reg", str(sd / "registration.json"),
             "--scene", str(sd / "schematic.json"), "-o", str(scene_out)])
        pred = json.loads(scene_out.read_text())["objects"]
        r = score(pred, json.loads((sd / "gt.json").read_text()))
        r["frame"] = fid
        results.append(r)
        print(f"{fid:<55} gt {r['n_gt']:2d} pred {r['n_pred']:2d} match {r['n_match']:2d}")
    summary = summarize(results)
    (out / "report.json").write_text(json.dumps({"summary": summary, "frames": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"report: {out / 'report.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("scene")
    o.add_argument("gt")
    b = sub.add_parser("batch")
    b.add_argument("--n", type=int, default=1000)
    b.add_argument("--skip-detect", action="store_true")
    args = ap.parse_args()
    if args.cmd == "one":
        r = score(json.loads(pathlib.Path(args.scene).read_text())["objects"], json.loads(pathlib.Path(args.gt).read_text()))
        print(json.dumps(r, indent=2))
        print(json.dumps(summarize([r]), indent=2))
    else:
        batch(args.n, args.skip_detect)


if __name__ == "__main__":
    main()
