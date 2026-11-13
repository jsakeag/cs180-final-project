"""Scene builder: pick a SUN RGB-D photo or upload your own, run the pipeline, walk it.

usage (from the repo root):
  .venv/bin/python builder/server.py        # http://localhost:5180

Serves the builder page at /, the engine at /engine/, and runs
  [prepare | write schematic + registration] -> detect.py --depth -> place.py
  -> complete.py (full room + predicted furniture) -> models_catalog.py match
  -> texture.py (photo textures) -> (SUN only) evaluate
as a background job per request.
"""
import json, pathlib, subprocess, sys, threading, time, uuid

from flask import Flask, abort, jsonify, request, send_from_directory

REPO = pathlib.Path(__file__).resolve().parents[1]
DETECT = REPO / "detect"
ENGINE = REPO / "engine"
JOBS = REPO / "builder" / "jobs"
PY = str(REPO / ".venv" / "bin" / "python")
sys.path.insert(0, str(DETECT))
import sunrgbd  # noqa: E402
import evaluate  # noqa: E402

app = Flask(__name__, static_folder=None)
jobs: dict[str, dict] = {}
MAX_SIDE = 1600   # uploaded photos are downscaled to this (speed); focal scales with it


# ------------------------------------------------------------------ static

@app.get("/")
def index():
    return send_from_directory(REPO / "builder", "index.html")


@app.get("/engine/<path:p>")
def engine_files(p):
    return send_from_directory(ENGINE, p)


@app.get("/img/sun/<fid>")
def sun_image(fid):
    d = sunrgbd.FRAMES / fid
    if not (d / "image.jpg").exists():
        abort(404)
    return send_from_directory(d, "image.jpg")


@app.get("/img/job/<job>/<name>")
def job_image(job, name):
    return send_from_directory(JOBS / job / "images", name)


@app.get("/work/<job>/<name>")
def work_file(job, name):
    return send_from_directory(JOBS / job, name)


# ------------------------------------------------------------------ gallery / upload

@app.get("/api/gallery")
def gallery():
    out = []
    for fid in sorted(p.name for p in sunrgbd.FRAMES.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((sunrgbd.FRAMES / fid / "meta.json").read_text())
        gt_path = DETECT / "scenes" / f"sun_{fid}" / "gt.json"
        out.append({"id": fid, "scene": meta["scene"], "url": f"/img/sun/{fid}",
                    "n_gt": len(json.loads(gt_path.read_text())) if gt_path.exists() else None,
                    "built": (ENGINE / "scenes" / f"sun_{fid}.json").exists()})
    return jsonify(out)


@app.post("/api/upload")
def upload():
    from PIL import Image, ImageOps
    files = request.files.getlist("images")
    if not files:
        abort(400, "no images")
    job = "upload_" + uuid.uuid4().hex[:8]
    d = JOBS / job / "images"
    d.mkdir(parents=True)
    out = []
    for i, f in enumerate(files):
        im = Image.open(f.stream)
        exif = im.info.get("exif")
        im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos: apply rotation
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        name = f"photo_{i:02d}.jpg"
        im.save(d / name, quality=92, **({"exif": exif} if exif else {}))
        out.append({"name": name, "url": f"/img/job/{job}/{name}", "width": im.width, "height": im.height})
    return jsonify({"job": job, "images": out})


# ------------------------------------------------------------------ jobs

def _run_job(job_id, steps, scene_rel, gt_path, work):
    j = jobs[job_id]
    try:
        for label, cmd in steps:
            j["stage"] = label
            j["log"].append(f"== {label}")
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                line = line.rstrip()
                if line and not any(s in line for s in ("Warning", "warn(", "Loading weights", "Redirects")):
                    j["log"].append(line)
            if p.wait():
                raise RuntimeError(f"{label} failed (exit {p.returncode})")
        if gt_path and gt_path.exists():
            j["stage"] = "scoring vs ground truth"
            pred = json.loads((ENGINE / scene_rel).read_text())["objects"]
            r = evaluate.score(pred, json.loads(gt_path.read_text()))
            j["metrics"] = {"summary": evaluate.summarize([r]), "missed": r["missed"], "false_pos": r["false_pos"]}
        j["viz"] = [f"/work/{work.name}/{f.name}" for f in sorted(work.glob("*_placed.jpg")) + sorted(work.glob("*_plan.png")) + sorted(work.glob("*_room.png"))]
        j["stage"], j["done"] = "done", True
        j["scene_url"] = f"/engine/index.html?scene={scene_rel}"
    except Exception as e:  # noqa: BLE001
        j["stage"], j["error"], j["done"] = "error", str(e), True


def _pipeline_steps(images, reg_path, schematic_path, scene_out, work, match_models):
    steps = [
        ("detecting objects + depth", [PY, "detect/detect.py", *map(str, images), "-o", str(work / "detections.json"), "--depth", "--viz"]),
        ("placing on the floor plan", [PY, "detect/place.py", "--detections", str(work / "detections.json"),
                                       "--reg", str(reg_path), "--scene", str(schematic_path), "-o", str(scene_out), "--viz"]),
        ("completing the room (shell + unseen furniture)", [PY, "detect/complete.py", str(scene_out), "--viz", str(work)]),
    ]
    if match_models:
        steps.append(("matching real 3D models", [PY, "detect/models_catalog.py", "match", str(scene_out)]))
    steps.append(("painting photo textures", [PY, "detect/texture.py", str(scene_out), "--detections",
                                              str(work / "detections.json"), "--reg", str(reg_path)]))
    return steps


def _start(job_id, steps, scene_rel, work, gt_path=None):
    jobs[job_id] = {"id": job_id, "stage": "queued", "log": [], "done": False, "started": time.time()}
    threading.Thread(target=_run_job, args=(job_id, steps, scene_rel, gt_path, work), daemon=True).start()
    return jsonify({"job": job_id})


@app.post("/api/run/sun")
def run_sun():
    body = request.get_json()
    fid = body["frame_id"]
    sd = sunrgbd.prepare(fid)
    work = JOBS / f"sun_{fid}"
    work.mkdir(parents=True, exist_ok=True)
    scene_rel = f"scenes/sun_{fid}.json"
    steps = _pipeline_steps([sunrgbd.FRAMES / fid / "image.jpg"], sd / "registration.json",
                            sd / "schematic.json", ENGINE / scene_rel, work, body.get("models", True))
    return _start(f"sun_{fid}_{uuid.uuid4().hex[:4]}", steps, scene_rel, work, sd / "gt.json")


@app.post("/api/run/upload")
def run_upload():
    """body: {job, room: {width, depth, wall_height}, registrations: [{name, image_points, plan_points}]}"""
    body = request.get_json()
    job = body["job"]
    work = JOBS / job
    if not work.exists():
        abort(404)
    W, D = float(body["room"]["width"]), float(body["room"]["depth"])
    H = float(body["room"].get("wall_height", 2.6))
    corners = [[0, 0], [W, 0], [W, D], [0, D]]
    schematic = {
        "name": job, "units": "meters",
        "spawn": {"position": [W / 2, 0, D - 0.5], "yaw_deg": 0},
        "schematic": {"floor": {"min": [0, 0], "max": [W, D]}, "wall_height": H,
                      "room": {"given": True, "corners": corners, "size": [W, D]},   # measured: no guessing
                      "walls": [{"from": corners[i], "to": corners[(i + 1) % 4], "thickness": 0.12} for i in range(4)]},
        "objects": [],
    }
    regs = []
    for r in body["registrations"]:
        if len(r["image_points"]) < 4 or len(r["image_points"]) != len(r["plan_points"]):
            abort(400, f"{r['name']}: need >= 4 matched point pairs")
        regs.append({"image": str((work / "images" / r["name"]).relative_to(REPO)),
                     "image_points": r["image_points"], "plan_points": r["plan_points"]})
    (work / "schematic.json").write_text(json.dumps(schematic, indent=2))
    (work / "registration.json").write_text(json.dumps(regs, indent=2))
    images = [REPO / r["image"] for r in regs]
    scene_rel = f"scenes/{job}.json"
    steps = _pipeline_steps(images, work / "registration.json", work / "schematic.json",
                            ENGINE / scene_rel, work, body.get("models", True))
    return _start(f"{job}_{uuid.uuid4().hex[:4]}", steps, scene_rel, work)


# ------------------------------------------------------------------ Paint Berkeley

CAMPUS = "scenes/campus_berkeley_core.json"
# a painted building becomes enterable: furnish any new interior and give it models
FURNISH_STEPS = [
    ("furnishing new interiors", [PY, "detect/interior.py", "furnish", "--new", "--scene", str(ENGINE / CAMPUS)]),
    ("matching interior models", [PY, "detect/models_catalog.py", "match", str(ENGINE / CAMPUS), "--missing"]),
]


@app.get("/api/paintings")
def paintings():
    out = []
    for p in sorted((ENGINE / "paintings").glob("*/painting.json")):
        d = json.loads(p.read_text())
        out.append({"id": d["id"], "building": d["building"], "style": d["style"],
                    "rules": [r["type"] for r in d["rules"]], "texture": f"/engine/{d['texture']}",
                    "elements": f"/engine/paintings/{d['id']}/elements.jpg", "size_m": d["size_m"]})
    return jsonify(out)


SAMPLES = REPO / "builder" / "samples" / "campus"


@app.get("/samples/campus/<name>")
def sample_file(name):
    return send_from_directory(SAMPLES, name)


@app.get("/api/samples")
def samples():
    """Openly licensed campus photos with suggested facade corners (builder/samples/campus)."""
    items = json.loads((SAMPLES / "samples.json").read_text())
    return jsonify([{**s, "url": f"/samples/campus/{s['file']}"} for s in items])


@app.post("/api/samples/use")
def use_sample():
    """Start a paint job from a sample photo, as if it had been uploaded."""
    items = {s["name"]: s for s in json.loads((SAMPLES / "samples.json").read_text())}
    s = items.get(request.get_json().get("name"))
    if not s:
        abort(404)
    job = "upload_" + uuid.uuid4().hex[:8]
    d = JOBS / job / "images"
    d.mkdir(parents=True)
    import shutil
    shutil.copy(SAMPLES / s["file"], d / "photo_00.jpg")
    (JOBS / job / "credit.txt").write_text(s["credit"])
    return jsonify({"job": job, "images": [{"name": "photo_00.jpg", "url": f"/img/job/{job}/photo_00.jpg",
                                            "width": s["width"], "height": s["height"]}]})


@app.post("/api/paint")
def paint():
    """body: {job, name (uploaded photo), corners [[x,y] x4 TL TR BR BL], near [x, z], label}"""
    body = request.get_json()
    work = JOBS / body["job"]
    photo = work / "images" / body["name"]
    if not photo.exists() or len(body.get("corners", [])) != 4:
        abort(400, "need an uploaded photo and 4 corners")
    label = "".join(ch for ch in body.get("label") or "" if ch.isalnum() or ch in "_-")[:40]
    cmd = [PY, "detect/paint.py", "add", "--scene", str(ENGINE / CAMPUS), "--photo", str(photo),
           "--corners", " ".join(f"{x:.1f},{y:.1f}" for x, y in body["corners"]),
           "--near", str(body["near"][0]), str(body["near"][1]),
           "--credit", (work / "credit.txt").read_text() if (work / "credit.txt").exists() else "uploaded",
           "--side", "in" if body.get("side") == "in" else "out"]
    if body.get("block") is not None:          # the game knows which building was aimed at
        cmd += ["--building", str(int(body["block"]))]
    if label:
        cmd += ["--name", label]
    return _start(f"paint_{uuid.uuid4().hex[:6]}", [("painting the facade", cmd)] + FURNISH_STEPS, CAMPUS, work)


@app.post("/api/paint/apply")
def paint_apply():
    """Imagine: put an existing painting on another wall. body: {painting, near [x, z]}"""
    body = request.get_json()
    pid = "".join(ch for ch in body["painting"] if ch.isalnum() or ch in "_-")
    r = subprocess.run([PY, "detect/paint.py", "apply", "--scene", str(ENGINE / CAMPUS), "--painting", pid,
                        "--near", str(body["near"][0]), str(body["near"][1]),
                        "--side", "in" if body.get("side") == "in" else "out"]
                       + (["--building", str(int(body["block"]))] if body.get("block") is not None else []),
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        abort(400, (r.stdout + r.stderr)[-400:])
    for _, cmd in FURNISH_STEPS:      # a newly enterable building gets its interior now
        subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return jsonify({"message": r.stdout.strip()})


@app.post("/api/furnish")
def furnish_from_photo():
    """Furnish a building's interior from a photo of a real room, placed around where the
    player stands. body: {job, name (uploaded photo), block, at [x, z], yaw (rad)}"""
    body = request.get_json()
    work = JOBS / body["job"]
    photo = work / "images" / body["name"]
    if not photo.exists():
        abort(400, "upload a photo first")
    dets = work / "detections.json"
    steps = [
        ("detecting furniture + depth", [PY, "detect/detect.py", str(photo), "-o", str(dets), "--depth", "--viz"]),
        ("fitting the floor and placing", [PY, "detect/photo_furnish.py", "--detections", str(dets), "--scene", str(ENGINE / CAMPUS),
                                          "--block", str(int(body["block"])), "--at", str(body["at"][0]), str(body["at"][1]),
                                          "--yaw", str(body["yaw"])]),
        ("matching models", [PY, "detect/models_catalog.py", "match", str(ENGINE / CAMPUS), "--missing"]),
    ]
    return _start(f"furnish_{uuid.uuid4().hex[:6]}", steps, CAMPUS, work)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    j = jobs.get(job_id)
    if not j:
        abort(404)
    return jsonify({**j, "log": j["log"][-60:], "elapsed": round(time.time() - j["started"], 1)})


if __name__ == "__main__":
    import os
    app.run(port=int(os.environ.get("PORT", 5180)), debug=False, threaded=True)
