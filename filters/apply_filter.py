#!/usr/bin/env python3
"""Apply a filter to one or more photos.

    python apply_filter.py looks/golden.json photo1.jpg photo2.dng
    python apply_filter.py looks/golden.json photo.jpg --strength 0.6 --contrast 0.1
    python apply_filter.py looks/golden.json photo.dng --format tif     # 16-bit TIFF
    python apply_filter.py looks/golden.json photo.jpg --cube golden.cube

Regions: if the filter was made with reference masks, give the photo's masks
too and each region is matched separately.

    python apply_filter.py looks/portrait.json me.jpg --subject-mask me_subject.png \\
        --local subject=1,0.15,1 --local background=1,0,0.85
    python apply_filter.py looks/portrait.json a.jpg b.jpg --auto-subject

Outputs go to out/<photo>_<filter>.<format> (plus <photo>_masks.jpg when masks
are used: check it). Each photo gets its own fit.
--cube writes a 3D LUT fitted to the first photo, for Lightroom, Photoshop,
DaVinci Resolve or phone editors. A LUT holds the global grade only (no
regions, clarity or texture).
"""
import argparse
from pathlib import Path

import numpy as np

import photofilter as pf
from make_filter import add_control_args, add_mask_args, controls_from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('filter', help='filter .json made by make_filter.py')
    ap.add_argument('photos', nargs='+')
    ap.add_argument('-o', '--outdir', default=str(Path(__file__).parent / 'out'))
    ap.add_argument('--format', default='jpg', choices=['jpg', 'png', 'tif'])
    ap.add_argument('--max-edge', type=int, default=0, help='resize output long edge (0 = full size)')
    ap.add_argument('--raw-ev', type=float, default=0.0, help='exposure for RAW decoding, in stops')
    ap.add_argument('--cube', help='also save a .cube LUT fitted to the first photo')
    add_mask_args(ap, 'photo (mask files: one photo at a time)')
    add_control_args(ap)
    args = ap.parse_args()

    if (args.subject_mask or args.foreground_mask) and len(args.photos) > 1:
        raise SystemExit('Mask files belong to one photo: run one photo at a time, or use --auto-subject.')

    f = pf.load_filter(args.filter)
    overrides, hsl, local = controls_from_args(args)
    controls = {**f['controls'], **overrides}
    hsl = {**f['hsl'], **hsl}
    local = {k: {**f['local'][k], **local.get(k, {})} for k in pf.REGIONS}
    use_masks = bool(args.subject_mask or args.foreground_mask or args.auto_subject)
    if use_masks and len(f['stats']) == 1:
        print('note: this filter has no reference regions, so masks only apply the per-region --local settings')

    for i, p in enumerate(args.photos):
        photo = pf.load_image(p, args.max_edge, raw_ev=args.raw_ev)
        masks = None
        if use_masks:
            # computed at fitting size, then scaled up at render time (as the notebook does)
            small = pf.downsize(photo, pf.FIT_LONG_EDGE)
            masks = pf.region_masks(small, args.subject_mask, args.foreground_mask, args.auto_subject)
            prev = Path(args.outdir) / f'{Path(p).stem}_masks.jpg'
            pf.save_image(pf.mask_overlay(small, masks), prev)
        models = pf.fit(photo, f, masks)
        out = pf.render(photo, models, controls, hsl, masks, local)
        clipped = 100 * float(np.any((out < 0) | (out > 1), axis=-1).mean())
        dst = Path(args.outdir) / f"{Path(p).stem}_{f['name']}.{args.format}"
        pf.save_image(out, dst)
        regions = pf.fitted_regions(models)
        print(f'{p} -> {dst}  ({clipped:.1f}% pixels clipped'
              + (f'; regions matched: {", ".join(regions)})' if regions else ')'))
        if i == 0 and args.cube:
            pf.save_cube(pf.bake_lut(models['global'], controls, hsl), args.cube, title=f['name'])
            print(f'LUT -> {args.cube}' + ('  (global grade only)' if regions else ''))


if __name__ == '__main__':
    main()
