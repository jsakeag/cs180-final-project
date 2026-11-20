# cs180-final-project

Turn a reference photo into a reusable, Instagram-style filter.

Pick a photo whose look you like; `filters/` measures its tone and colour, fits that look to any other photo
(JPEG, PNG or RAW), and lets you fine-tune it with sliders. Subjects, near scenery and backgrounds can be matched
separately with masks, and the global look can be exported as a `.cube` LUT.

```bash
cd filters
pip install -r requirements.txt
python3 make_filter.py reference.jpg --name mylook      # -> looks/mylook.json
python3 apply_filter.py looks/mylook.json photo.jpg     # -> out/photo_mylook.jpg
python3 tune.py looks/mylook.json photo.jpg             # interactive sliders
```

See [`filters/README.md`](filters/README.md) for all options and how the method works.

## Layout

| Path | What it is |
|---|---|
| `filters/photofilter.py` | the library: RAW/image loading, fitting, regions, rendering, LUT export |
| `filters/make_filter.py` | reference photo → filter `.json` |
| `filters/apply_filter.py` | apply a filter to photos (JPEG/PNG/16-bit TIFF, optional `.cube`) |
| `filters/tune.py` | slider window with live preview |
| `filters/demo.py` | comparison sheet of several filters on several photos |
| `filters/reference-lab-raw-style-matching.ipynb` | the notebook the method is ported from (outputs cleared) |
