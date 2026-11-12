# Detection: photos → objects in the scene JSON

```bash
python3 -m venv .venv && .venv/bin/pip install -r detect/requirements.txt   # once, from repo root

# 1. detect objects (+ metric depth) in each photo
.venv/bin/python detect/detect.py photos/*.jpg -o detect/out/detections.json --depth --viz

# 2. place them on the room's floor plan
.venv/bin/python detect/place.py --detections detect/out/detections.json \
    --reg detect/scenes/bedroom/registration.json \
    --scene detect/scenes/bedroom/schematic.json \
    -o engine/scenes/bedroom.json --viz

# 3. swap boxes for real 3D models (once: build the catalog, ~3.5k models)
.venv/bin/python detect/models_catalog.py build
.venv/bin/python detect/models_catalog.py match engine/scenes/bedroom.json

# 4. walk around it; press P to view from each photo's camera
cd engine && python3 -m http.server 8000   # http://localhost:8000/?scene=scenes/bedroom.json

python3 detect/tests/test_homography.py && python3 detect/tests/test_camera.py
```

## Per room you provide

- `schematic.json`: the room's floor and walls in meters (same format as the engine's scene files, `objects` empty).
- `registration.json`: for each photo, ≥ 4 floor points clicked in the photo (`image_points`, pixels) and the same points on the plan (`plan_points`, meters). Room corners, door frames and rug corners all work. Use ≥ 5 points for least-squares slack.

## How placement works

| step | what |
|---|---|
| floor homography | `homography.py`: clicked pairs → H (image → plan), DLT + Hartley normalization |
| camera pose | `camera.py`: H = K[r1 r2 t] → focal length (if no EXIF), camera position, height, facing |
| depth scale | Depth Anything's metric depth is rescaled so the clicked floor points sit at their true distance |
| per object | visible floor contact → bottom of box through H; contact hidden (cut off by the image edge / behind something) → box center + depth through the camera pose; sitting on something (same depth as the object covering its base) → lifted to that height |
| size | box width/height in meters, clamped to 0.6–1.6× a per-class prior (`priors.py`) |
| facing | bottom edge direction, snapped to the walls within 20° |
| multiple photos | same class within ~half a footprint → merged |

Debug output (`--viz`, in `detect/out/`): `*_det.jpg` (boxes + depth), `*_placed.jpg` (what got placed and how: blue = floor contact, green = depth), `<scene>_plan.png` (top-down plan with the recovered camera).

## Known limits

- `samples/` are COCO photos with no measurements; `scenes/bedroom` uses guessed floor coordinates, so its absolute scale is a guess. Real rooms need measured `plan_points`.
- Loose boxes (e.g. the bed's box covering a plant pot) can wrongly mark a base as hidden; depth placement is then used, which is less precise.
- Only floor-standing classes are placed. Wall items (tv, pictures, windows, doors) are detected but not placed yet.

## 3D models

`models_catalog.py` retrieves a real model per object: ABO (Amazon Berkeley Objects, CC BY 4.0, ~8k real furniture products with metric GLBs) for furniture, Poly Haven (CC0) for plants. CLIP embeds each product's catalog photo; each detected crop is matched against its class group (dresser/cabinet/nightstand etc. share candidates) with a penalty for size mismatch. The model keeps its real proportions, scaled 0.7–1.4× toward the estimate, and the physics box is set to the model's true bounding box. Downloads go to `engine/models/` (git-ignored, re-fetched by `match`); attribution is written to the scene's `credits`.

## SUN RGB-D + evaluation

`sunrgbd.py` reads the 6.9 GB SUN RGB-D zip remotely (HTTP range requests) and downloads only the needed files per frame (~50 KB). Each frame becomes a schematic (room layout polygon), a registration (floor points projected with the calibrated camera; points hidden by furniture or beyond walls are marked not visible so they don't calibrate depth) and `gt.json` (3D boxes in engine coordinates).

```bash
.venv/bin/python detect/sunrgbd.py select --n 60 --types bedroom living_room
.venv/bin/python detect/evaluate.py batch          # -> detect/out/eval/report.json
```

Baseline, 60 bedroom/living-room frames (perfect registration, 169 floor-standing GT objects):

| | |
|---|---|
| recall / precision | 0.65 / 0.53 |
| median center error | 0.41 m |
| ... placed by visible floor contact (26) | **0.22 m** |
| ... placed by depth (84) | 0.46 m |
| size error (mean \|log ratio\|) | 0.26 (~30%) |
| median facing error | 21° |

Coordinate notes (learned the hard way): SUN annotation frame is gravity-aligned, X right, **Y down**, Z forward; `Ymin`/`Ymax` are not consistently bottom/top (bottom = max); the layout's floor height can be ~10 cm off, so the floor is taken from furniture bottoms when lower.
