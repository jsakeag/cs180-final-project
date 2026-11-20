# filters: photo → Instagram-style filter

Pick a photo whose look you like. This turns it into a filter you can apply to any other photo (JPEG, PNG or RAW),
and can export the look as a `.cube` LUT for Lightroom, Photoshop, DaVinci Resolve or phone editors.

The method is a port of the *Reference Lab RAW Style Matching* notebook (`reference-lab-raw-style-matching.ipynb`
in this folder), including its subject/foreground/background masks and its sliders. It gives the notebook's output
to within float rounding (max difference < 1e-6 on the notebook's three references, both globally and with masks and
every control set), as a reusable library, command-line tools and a slider window.

```bash
pip install -r requirements.txt        # numpy, scipy, Pillow, matplotlib (+ optional rawpy, tifffile, rembg)
```

## Use it

```bash
cd ~/Downloads/cs180-final-project/filters

# 1. make a filter from a reference photo  -> looks/amber.json
python3 make_filter.py cozy_room.jpg --name amber

# 2. apply it                              -> out/<photo>_amber.jpg
python3 apply_filter.py looks/amber.json photo1.jpg P1060804.dng
python3 apply_filter.py looks/amber.json P1060804.dng --format tif --raw-ev 1   # 16-bit TIFF, RAW +1 stop

# tweak it (same controls as the notebook's sliders)
python3 apply_filter.py looks/amber.json photo.jpg --strength 0.6 --contrast 0.1 --vibrance 0.2 \
        --zones 0 0.02 0 0 -0.03 --hsl green=-10,-0.15,0 --clarity 0.2

# 3. optional: LUT fitted to a photo (global grade only: no regions, clarity or texture)
python3 apply_filter.py looks/amber.json photo.jpg --cube amber.cube

# comparison sheet (rows = filters, columns = photos) -> out/demo.jpg
python3 demo.py --refs a.jpg b.jpg --targets x.jpg y.jpg
```

Controls given to `make_filter.py` are saved as the filter's defaults; `apply_filter.py` flags override them.

### Interactive sliders (`tune.py`)

```bash
python3 tune.py looks/amber.json photo.jpg
python3 tune.py looks/portrait.json me.jpg --subject-mask me_subject.png      # with regions
python3 tune.py looks/amber.json photo.jpg --save-as looks/amber_soft.json   # keep the original filter
```

A window with a live preview. Left: strength, tone, colour and all finishing controls, plus the five tonal zones.
Right: view (after / before / split / masks / reference), a region picker with that region's strength, exposure and
saturation, and an HSL band picker with that band's hue, saturation and lightness.
**Save filter** writes the settings into the filter `.json`; **Save full-size image** renders to `out/`.

### Region masks (subject / foreground / background)

Masks let the subject, near scenery and the rest be matched separately, e.g. a face to the reference's face and the
sky to its background. They need to exist in **both** photos: give the reference's masks to `make_filter.py` and the
photo's to `apply_filter.py` / `tune.py`.

```bash
python3 make_filter.py portrait_ref.jpg --name portrait --subject-mask ref_subject.png
python3 apply_filter.py looks/portrait.json me.jpg --subject-mask me_subject.png \
        --local subject=1,0.15,1 --local background=1,0,0.85     # strength,exposure,saturation
python3 apply_filter.py looks/portrait.json me.jpg --auto-subject # detect it (pip install "rembg[cpu]")
```

- Masks are white-on-black images with the same framing and orientation as their photo (any resolution).
  `--foreground-mask` is near scenery, separate from the subject. Subject wins where they overlap; anything not
  covered is background. Edges are softened slightly.
- **Match meaning, not position:** the reference's subject mask should cover the same kind of thing.
- A region below 1.5% of either image, or missing from either, falls back to the global grade (with a warning).
- `--auto-subject` uses rembg's small u2netp model. It finds a *salient* subject, not semantic classes, so it can be
  wrong on landscapes, clutter, hair, glass or several subjects. Always look at the `*_masks.jpg` preview that
  `make_filter.py` / `apply_filter.py` save (red subject, green foreground, blue background), or the Masks view
  in `tune.py`.
- Masks are computed at fitting size (≤1280 px) and scaled up for the full-size render, as in the notebook, so check
  fine edges at full size.

| Control | Default | Range | Meaning |
|---|---|---|---|
| `strength` | 0.85 | 0–1 | overall mix of the matched look |
| `tone` | 0.85 | 0–1 | how much of the light curve |
| `color` | 0.80 | 0–1 | how much of the colour curves + zone offsets |
| `exposure` | 0 | −2–2 | stops |
| `temperature`, `tint` | 0 | −1–1 | warm/cool and green/magenta bias (not Kelvin) |
| `contrast` | 0 | −0.5–0.5 | |
| `saturation` | 1 | 0–2 | |
| `vibrance` | 0 | −0.5–0.5 | saturation that favours muted colours |
| `zones` | 0×5 | −0.2–0.2 each | lift blacks, shadows, midtones, highlights, whites |
| `hsl` | 0 | | per band `[hue°, saturation, lightness]`; bands red, orange, yellow, green, aqua, blue, purple, magenta |
| `clarity`, `texture` | 0 | −0.5–0.5 | local contrast at coarse / fine scale |

## How it works (`photofilter.py`)

All in display sRGB, values 0–1. Photos are fitted on 40,000 sampled pixels at ≤1280 px, then applied at full size.

1. **Light curve.** Match 25 luma quantiles (1–99%) of the photo to the reference with a monotonic PCHIP curve.
   The ends are pinned (black stays black, white stays white), each point moves ≤ 0.30, and the result is
   85% curve + 15% identity. It's applied as an additive luma shift, so colour differences are kept.
2. **RGB curves.** Same fit per channel (≤ 0.18), luma of the change removed so it only alters colour.
3. **Zone offsets.** Mean RGB difference in five overlapping tonal zones, capped at 0.15, luma removed,
   down-weighted where a zone has few pixels.
4. **Regions.** With masks, steps 1–3 are fitted separately per region, each region gets its own strength, exposure
   and saturation, and the results are blended by the soft masks.
5. **Mix.** colour change = 0.65 × RGB curves + 0.35 × zone offsets, then the finishing controls.

A saved filter stores only the reference's statistics (quantiles, zone means and zone weights, globally and per
region) plus control settings, so the curves are refitted to each new photo, as in the notebook. Once fitted, the
global grade is per-pixel apart from clarity/texture, so it can be baked into a 33³ `.cube` LUT (mean error vs.
direct ≈ 0.004).

Not carried over from the notebook: its Colab-specific parts (embedded files, uploads, Drive) and tiled export.

## Limits and ideas

- It matches whole-image distributions, so content counts as "look": a blue sky in the reference pulls colour toward
  blue. Region masks reduce this; estimating tints from neutral pixels only would protect object colours further, at
  the cost of a weaker look.
- The reference's subject matter isn't recreated: no light rays, fog, blur or grain.
- Next steps: learning a LUT from a before/after pair; comparing against Reinhard (2001) and Pitié (2005) colour
  transfer.
