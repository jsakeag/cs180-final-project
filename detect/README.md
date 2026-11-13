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

Results on 60 bedroom/living-room frames (perfect registration, 169 floor-standing GT objects):

| | |
|---|---|
| recall / precision | 0.65 / 0.53 |
| median center error | 0.41 m |
| ... placed by visible floor contact (25) | **0.22 m** |
| ... placed by depth (84) | 0.45 m |
| size error (mean \|log ratio\|) | 0.26 (~30%) |
| long-axis error, elongated objects (59) | **3.7° median, 66% within 15°** (was 35.8°, 14%) |

### Orientation

A 2D box says nothing direct about which way a sofa runs. The old rule set its facing perpendicular to the camera ray, so everything faced the camera; the median axis error was 35.8°, about chance. Now (`place.py: fit_orientation`):

1. The room's wall direction is taken from the observed walls, ignoring view-cone edges (lines through the camera). Furniture runs along walls, which gives two candidate axes, plus "square to the camera" at a penalty.
2. For each candidate, the class-size 3D box is projected into the photo and scaled to the detection's width. Score = IoU with the 2D box − 0.5·|log scale| − penalty, and the best candidate wins. When the base is hidden or cut off, the box bottom is ignored.
3. Front versus back is symmetric for a box, so `complete.py` decides it once the room is known:
   - against a wall: face away from it (a bed faces with its foot end)
   - seat near its table, or sofa near its coffee table: face it
   - otherwise: face the camera

   Long items only flip front-to-back; context never overrides the photo's axis.

SUN ground truth has no facing direction, so the table measures the axis only. The weights were swept on these frames: the size weight (0.5 best for axis) and the camera-facing penalty (≥ 0.3 means it never wins). Moving the center along with the fitted box hurt floor-contact accuracy (0.22 → 0.26 m), so the center still comes from the measured front point.

Coordinate notes (learned the hard way): SUN annotation frame is gravity-aligned, X right, **Y down**, Z forward; `Ymin`/`Ymax` are not consistently bottom/top (bottom = max); the layout's floor height can be ~10 cm off, so the floor is taken from furniture bottoms when lower.

## Completing the room

A photo shows part of a room. `complete.py` turns a placed scene into a whole room:

1. **Room type**: the dataset label, else CLIP zero-shot on the photos (bedroom / living room / office / dining room). Outdoor or other photos are left alone.
2. **Shell**: a rectangle aligned with the walls the camera saw (SUN layouts are view polygons, so the edges that run through the camera are ignored). Sides backed by an observed wall stay put. Unseen sides, always including the one behind the camera, are pushed out to a typical room size for the type (bedroom 3.4 × 4.0 m, living room 4.2 × 5.4 m). Uploads use the measured room as is.
3. **Unseen furniture**: `room_prior.py` learns a small model from about 1,500 SUN RGB-D annotations (evaluation frames excluded). Per room type it records:
   - per-m² rate of each class over in-view floor
   - how many a room holds
   - P(against a wall)
   - median size
   - where each class sits relative to every other class, as samples in the anchor's frame (offset + relative facing)

   Expected count in the unseen floor = rate × unseen area, capped. A class most rooms of that type have (a bedroom's bed) is always added if missing. Each object goes to the spot that scores best under the wall term and a kernel density of the learned relations. The spot must be out of every camera's view (or the photo would have shown it), must not overlap anything, and must not be on top of the camera. Seats face the table or desk they're at; a bed's front is its foot end.

```bash
.venv/bin/python detect/room_prior.py fetch --per-type 400   # ~10 KB/frame, annotations only
.venv/bin/python detect/room_prior.py learn                  # -> detect/data/room_prior.json
.venv/bin/python detect/complete.py engine/scenes/<scene>.json --viz detect/out
```

Predicted objects have `source.placed_by = "predicted"`. They get models by CLIP text ("a photo of a sofa") plus the average look of the photographed furniture. The engine labels them "(predicted)", and **O** hides or shows them. `evaluate.py` ignores them, because SUN ground truth only covers what the camera saw. Learned relations look right (nightstand about 1 m beside the bed at the headboard; coffee table about 1 m in front of the sofa; chair about 0.5 m in front of the desk). Whole-room predictions can't be scored against SUN, which has no full-room ground truth.

## Photo textures

`texture.py` paints the photos back onto the scene (the builder's last step):

- **Object outlines.** Each seen object is cut out of its detection box with GrabCut. The outline is used three ways:
  - it is masked out of floor and wall sampling (the box would also hide the wall around the object)
  - it supplies the object's colors
  - it goes into a per-photo label image for projection
- **Floor and walls.** Every texel of the floor (top-down, 80 px/m) and of each wall is projected into each photo with the recovered camera. It is sampled only where that surface is really what the photo shows:
  - in frame and in front of the camera
  - not seen at a grazing angle (cos > 0.2)
  - outside every object's outline
  - no farther than the depth map says, so it isn't hidden behind something

  With several photos, the closest view wins. This is proj4a's rectification run per texel, with a visibility test. Unseen floor is filled by mirror-tiling a clean patch of what was seen, but only if the patch is uniform (wood grain, carpet). A rug or a poster would tile into a kaleidoscope, so then the fill is the median color, feathered into the seen part. Unseen walls get the room's paint color.
- **Furniture, pattern: projective texturing (engine).** Each photo becomes a projector built from the camera's real intrinsics K. A depth pre-pass is rendered from the photo camera. A model fragment takes the photo's pixel if:
  - it is the surface that camera sees (depth test)
  - the pixel lies inside that object's outline (label image)

  From the photo viewpoint the bed shows the real quilt. From elsewhere, the pattern stays on the surfaces that faced the camera. Where the model is bigger than the real object's outline, the model keeps its own texture.
- **Furniture, color everywhere else: Reinhard color transfer** in Lab of the model's base-color textures toward the object's outline pixels:
  - color (a, b) moves fully
  - lightness moves halfway, after normalizing the photo's exposure (photo median L → 128). Photos are often dark, and the engine lights the model again.

The engine shows photo surfaces unlit (the photo already has its light). **Esc** opens a menu to turn photo textures off, which gives plain surfaces and original models. The menu also toggles predicted furniture and labels; settings are remembered per browser.

## Outdoor: campus from OpenStreetMap

`campus.py` builds the outdoor equivalent of a room layout from OpenStreetMap (no account needed):
- **Buildings** become extruded blocks, using the height from OSM's `height` or `building:levels` × 3.6 m. Multi-part buildings (relations, such as Doe Library) are included. Every footprint edge is a collider.
- **Ground:** lawns, plazas, paths and steps become flat surfaces.
- **Street furniture:** trees, benches, street lamps, bins, bike racks and fountains become placed objects. Trees and lamps are simple procedural shapes until real models are matched.

```bash
.venv/bin/python detect/campus.py fetch berkeley_core 37.8700 -122.2620 37.8745 -122.2560   # any bbox / any campus
.venv/bin/python detect/campus.py build berkeley_core     # -> engine/scenes/campus_berkeley_core.json
```

Central Berkeley comes out as 527 × 497 m: 54 buildings (47 named; the Campanile is 94 m), 642 paths, 132 steps, 59 lawns, 120 trees, 62 benches and 38 lamps. You spawn south of the Campanile, facing it. The engine draws outdoor scenes with sky and fog, and shadows follow the player.

Not yet:
- **Terrain:** the ground is flat, but Berkeley is hilly. Needs USGS elevation data.
- **Photo registration:** registering campus photos to this map, e.g. Mapillary images (CC BY-SA, needs a free API token) with GPS positions.
- **Real models** for the street furniture.

Map data © OpenStreetMap contributors (ODbL).

## Paint Berkeley

People photograph campus buildings and say roughly where on the map they were. Each photo paints a building and reshapes it. Any facade can also go on any other building ("imagine").

1. **Annotate** (builder → Paint Berkeley tab): upload a photo, click the facade's 4 corners, click near the building on the campus map. Precision isn't needed.
2. **Rectify.** The corners give a homography (proj4a's `compute_h`) that unwarps the facade into a flat texture. Its real width-to-height ratio also comes from that homography: the facade's two axes, seen by the camera as K⁻¹h₁ and K⁻¹h₂, must be at right angles. That constraint gives the focal length, and their length ratio is the aspect. The height comes from the map, and the painting repeats along the wall, since a photo often shows only part of a facade. Consecutive map segments facing the same way count as one wall.
3. **3D from the photo: depth relief.** Depth Anything V2 (metric outdoor) estimates per-pixel depth, which is unwarped with the facade homography. At 20–40 m its absolute error is meters but smooth, while real relief (columns, window recesses, cornices) is under a meter and sharp. So the depth is high-passed (local 2 m Gaussian mean minus depth), and then:
   - vegetation and sky are masked out by color (specks removed first)
   - a 0.5 m median filter cleans it
   - it fades to flat at the wall's ends and is capped at ±1 m

   The result is saved as `relief.png`. On its own, a height field only gives soft "terrain" (it can't make a free-standing column or a sharp ledge), so the relief mainly supplies depths for the next step, plus a fallback mesh for facades where no parts are found.
4. **Architectural parts fitted to the photo** (`fit_parts`, often called inverse procedural modeling). Grounding DINO runs on the sharp 80 px/m rectification in overlapping tiles, since a long facade squeezed to its ~800 px input makes windows a few pixels wide. Then:
   - **Openings** (windows, doors, arches) snap into rows and columns, since facades repeat. Grid cells the detector missed are filled only where the photo is dark like glass and not covered by a tree. Each opening's recess depth = depth around it − depth inside, capped at 15–60 cm.
   - **Columns** are tall, narrow detections. A run fills only gaps that are whole multiples of the median spacing, and extends only while the depth still stands out, the area isn't vegetation and no window is in the way. Protrusion comes from the depth.
   - **Ledges** (cornices, string courses) are rows where the depth sticks out along most of the wall.
   - Colors come from the photo.

   The engine builds these concretely: a wall surface with real holes, recesses with the photo on their back faces, turned columns with capitals, and ledges. The stock colonnade and arcade rules are skipped, and the roof rule still applies. The Esc menu toggles "3D facades".
5. **Read the facade (style).**
   - Grounding DINO (open-vocabulary detector) finds columns, arches, windows, doors, pediments and balconies. YOLO-World found almost nothing on facades.
   - CLIP reads the column order (Doric, Ionic or Corinthian) from capitals cut out of a sharper rectification at 80 px per meter, the overall style, and whether a red tile roof shows above the facade.
6. **Rules** (the engine builds them procedurally; no 3D model files needed):

   | facade shows | building gets |
   |---|---|
   | ≥ 2 tall, narrow columns, at most 8 diameters apart (lamp posts don't count) | a colonnade in front of the wall: turned columns with capitals by order, an entablature, a pediment if one was seen |
   | ≥ 3 similar arches | an arcade: stone arch surrounds stepped out from the wall |
   | red tile roof | a terracotta hip roof, or a low tile cap that follows the footprint if it isn't rectangular |

   Columns found at a regular spacing fill in the gaps the detector missed.
7. **Imagine:** `paint.py apply --painting dwinelle_plaza --building "Wheeler Hall" --facing N`, or click a facade and a wall in the UI. The painting's rules come with it.

```bash
.venv/bin/python detect/paint.py add --photo photo.jpg --corners "x,y x,y x,y x,y" --near X Z    # TL TR BR BL
.venv/bin/python detect/paint.py apply --painting wheeler_south --near X Z
.venv/bin/python detect/paint.py list
```

**In the game:** press **B**, aim at a building (its wall lights up orange) and click. A panel opens:
- apply any painted facade with one click, or
- paint from a photo: pick one of the samples (corners pre-marked) or upload your own and click 4 corners.

The scene reloads in place when done. The Esc menu also has **Exit to Scene Builder**.

**Sample photos:** `builder/samples/campus/` has 9 openly licensed Wikimedia Commons photos with suggested corners in `samples.json`, and each photo's credit carries into its painting:
- Beaux-Arts: Doe, Wheeler, California, Haviland, Hearst Mining (its modern glass addition)
- red-tile roof: Dwinelle
- Second Empire brick: South Hall
- modern concrete: Sproul
- Brutalist: Evans

**Enterable buildings:** a painted building becomes enterable. `paint.py` turns ground-level doors and arches from its paintings into doorways, stored on the building in world coordinates (`update_doors`; `paint.py doors` recomputes them). A painted wall that showed no door gets one in the middle. The engine then draws the building as a hollow shell, with door gaps in both the walls and the colliders, lintels above the doors, an interior floor, a ceiling at 4.5 m and an interior light. The facade's door openings stay open all the way through.

Demo paintings use Wikimedia Commons photos of Wheeler (CC BY-SA 3.0), Doe Library (CC BY-SA 4.0) and Dwinelle (public domain). Credits are in each `engine/paintings/*/painting.json`.
