# Scene builder

```bash
.venv/bin/pip install -r detect/requirements.txt flask remotezip scipy
.venv/bin/python detect/models_catalog.py build          # once: 3D model catalog
.venv/bin/python detect/sunrgbd.py select --n 60          # once: SUN RGB-D gallery (~3 MB)
.venv/bin/python detect/room_prior.py fetch && .venv/bin/python detect/room_prior.py learn   # once: room prior (~15 MB)
.venv/bin/python builder/server.py                        # http://localhost:5180
```

- **Dataset tab**: pick a SUN RGB-D photo → Build. Floor registration and room outline come from the dataset's calibration (no clicking); the result is scored against its ground-truth 3D boxes.
- **Upload tab**: upload photos, enter the room size, then for each photo click ≥ 4 floor points in the photo and the same spots on the plan (room corners are best; clicks snap to walls). Build.
- Each build runs `detect.py --depth` → `place.py` → `complete.py` (full room + predicted furniture for unseen floor) → `models_catalog.py match`, writes `engine/scenes/<name>.json`, and links to the engine (press **P** there to view from the photo's camera).

Generated jobs go to `builder/jobs/` and generated scenes to `engine/scenes/sun_*.json` / `upload_*.json` (all git-ignored).
