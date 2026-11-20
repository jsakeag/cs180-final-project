#!/usr/bin/env python3
"""Turn a reference photo into a filter.

    python make_filter.py reference.jpg                 # -> looks/reference.json
    python make_filter.py sunset.jpg --name golden      # -> looks/golden.json
    python make_filter.py sunset.jpg --strength 0.7 --contrast 0.1

Regions (optional): give the reference's masks so the subject, foreground and
background can be matched separately when the photo has masks too.

    python make_filter.py portrait.jpg --subject-mask portrait_subject.png
    python make_filter.py portrait.jpg --auto-subject          # needs rembg

Any control given here is saved as the filter's default; apply_filter.py and
tune.py can still change it.
"""
import argparse
from pathlib import Path

import photofilter as pf

FLOAT_CONTROLS = ['strength', 'tone', 'color', 'exposure', 'temperature', 'tint',
                  'contrast', 'saturation', 'vibrance', 'clarity', 'texture']


def add_control_args(ap):
    g = ap.add_argument_group('controls (defaults: strength .85, tone .85, color .80, others neutral)')
    for k in FLOAT_CONTROLS:
        g.add_argument(f'--{k}', type=float)
    g.add_argument('--zones', type=float, nargs=5, metavar=('BLACKS', 'SHADOWS', 'MIDS', 'HIGHS', 'WHITES'),
                   help='per-zone brightness lift, each about -0.2..0.2')
    g.add_argument('--hsl', action='append', default=[], metavar='BAND=HUE,SAT,LIGHT',
                   help=f'HSL mixer, e.g. green=-10,-0.15,0. Bands: {", ".join(pf.HSL_BANDS)}')
    g.add_argument('--local', action='append', default=[], metavar='REGION=STRENGTH,EXPOSURE,SAT',
                   help='per-region adjustment, e.g. subject=1,0.15,1 or background=1,0,0.85')


def add_mask_args(ap, what):
    g = ap.add_argument_group(f'region masks for the {what} (white = inside, same framing and orientation)')
    g.add_argument('--subject-mask', help='white-on-black image of the main subject')
    g.add_argument('--foreground-mask', help='white-on-black image of near scenery (not the subject)')
    g.add_argument('--auto-subject', action='store_true',
                   help='detect the subject automatically (pip install "rembg[cpu]")')


def controls_from_args(args):
    c = {k: getattr(args, k) for k in FLOAT_CONTROLS if getattr(args, k) is not None}
    if args.zones:
        c['zones'] = list(args.zones)
    hsl = {}
    for item in args.hsl:
        band, vals = item.split('=')
        if band not in pf.HSL_BANDS:
            raise SystemExit(f'unknown HSL band {band!r}')
        hsl[band] = [float(v) for v in vals.split(',')]
    local = {}
    for item in args.local:
        region, vals = item.split('=')
        if region not in pf.REGIONS:
            raise SystemExit(f'unknown region {region!r}; use {", ".join(pf.REGIONS)}')
        s, e, sat = (float(v) for v in vals.split(','))
        local[region] = dict(strength=s, exposure=e, saturation=sat)
    return c, hsl, local


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('reference', help='photo whose look you want to copy (RAW works with rawpy)')
    ap.add_argument('--name', help="filter name (default: the photo's file name)")
    ap.add_argument('-o', '--out', help='output .json (default: looks/<name>.json)')
    add_mask_args(ap, 'reference')
    add_control_args(ap)
    args = ap.parse_args()

    controls, hsl, local = controls_from_args(args)
    f = pf.make_filter(args.reference, name=args.name, controls=controls, hsl=hsl, local=local,
                       subject_mask=args.subject_mask, foreground_mask=args.foreground_mask,
                       auto_subject=args.auto_subject)
    out = Path(args.out) if args.out else Path(__file__).parent / 'looks' / f"{f['name']}.json"
    pf.save_filter(f, out)

    print(f"Saved filter '{f['name']}' -> {out}")
    if f['region_coverage']:
        cov = ', '.join(f'{k} {100 * v:.0f}%' for k, v in f['region_coverage'].items())
        stored = [k for k in f['stats'] if k != 'global']
        print(f'  regions: {cov}; stored: {", ".join(stored) or "none (all too small)"}')
        ref = pf.load_image(args.reference, pf.FIT_LONG_EDGE, raw_half=True)
        masks = pf.region_masks(ref, args.subject_mask, args.foreground_mask, args.auto_subject)
        prev = out.with_name(out.stem + '_masks.jpg')
        pf.save_image(pf.mask_overlay(ref, masks), prev)
        print(f'  check the masks: {prev}  (red subject / green foreground / blue background)')
    print('  zone        mean R   mean G   mean B')
    for name, m in zip(pf.ZONE_NAMES, f['stats']['global']['zone_mean']):
        print(f'  {name:<10} ' + ' '.join(f'{v:7.3f}' for v in m))


if __name__ == '__main__':
    main()
