#!/usr/bin/env python3
"""Make a comparison sheet: each row is a filter (reference photo on the left),
each column a target photo, before on top of after.

    python demo.py --refs sunset.jpg forest.jpg --targets a.jpg b.jpg c.jpg
    -> out/demo.jpg
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import photofilter as pf

TILE = 320


def tile(rgb, label=""):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.copy()
    img.thumbnail((TILE, TILE), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE), (24, 24, 24))
    canvas.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    if label:
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, TILE, 18], fill=(0, 0, 0))
        d.text((6, 3), label, fill=(255, 255, 255))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--strength", type=float, help="override the default strength (0.85)")
    ap.add_argument("-o", "--out", default=str(Path(__file__).parent / "out" / "demo.jpg"))
    args = ap.parse_args()
    overrides = {} if args.strength is None else {"strength": args.strength}

    targets = [(Path(t).stem, pf.load_image(t, 1600)) for t in args.targets]
    cols = 1 + len(targets)
    rows = [[tile(np.full((8, 8, 3), 0.1))] + [tile(t, f"original: {n}") for n, t in targets]]
    for r in args.refs:
        f = pf.make_filter(r)
        name = f["name"]
        row = [tile(pf.load_image(r, 1600), f"reference: {name}")]
        for n, t in targets:
            row.append(tile(pf.apply_filter(t, f, **overrides), f"{n} + {name}"))
        rows.append(row)

    sheet = Image.new("RGB", (cols * TILE, len(rows) * TILE), (24, 24, 24))
    for y, row in enumerate(rows):
        for x, t in enumerate(row):
            sheet.paste(t, (x * TILE, y * TILE))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
