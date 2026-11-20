#!/usr/bin/env python3
"""Interactive sliders: tune a filter on a photo with a live preview.

    python tune.py looks/amber.json photo.jpg
    python tune.py looks/amber.json photo.jpg --subject-mask photo_subject.png
    python tune.py looks/amber.json photo.dng --auto-subject --raw-ev 1

Left column: the look (strength, tone, color) and finishing controls, plus
the five tonal zones. Right column: pick a region and set its strength,
exposure and saturation; pick an HSL band and set its hue, saturation and
lightness. View: after, before, split, masks, or the reference photo.

Save filter writes your settings into the filter .json (or --save-as).
Save image renders the full-resolution photo to out/.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

import photofilter as pf
from make_filter import add_mask_args

PREVIEW_EDGE = 1000
SLIDERS = [  # name, min, max
    ('strength', 0, 1), ('tone', 0, 1), ('color', 0, 1), ('exposure', -2, 2),
    ('temperature', -1, 1), ('tint', -1, 1), ('contrast', -.5, .5), ('saturation', 0, 2),
    ('vibrance', -.5, .5), ('clarity', -.5, .5), ('texture', -.5, .5)]
LOCAL_SLIDERS = [('strength', 0, 1), ('exposure', -1, 1), ('saturation', 0, 2)]
HSL_SLIDERS = [('hue', -30, 30), ('saturation', -1, 1), ('lightness', -.3, .3)]


class Tuner:
    def __init__(self, args):
        self.args = args
        self.f = pf.load_filter(args.filter)
        self.photo = pf.load_image(args.photo, raw_ev=args.raw_ev)
        small = pf.downsize(self.photo, pf.FIT_LONG_EDGE)
        use_masks = bool(args.subject_mask or args.foreground_mask or args.auto_subject)
        self.masks = (pf.region_masks(small, args.subject_mask, args.foreground_mask, args.auto_subject)
                      if use_masks else None)
        self.models = pf.fit(self.photo, self.f, self.masks)
        self.preview = pf.downsize(self.photo, PREVIEW_EDGE)
        self.preview_masks = (None if self.masks is None else
                              {k: pf.resize_float(v, self.preview.shape[:2]) for k, v in self.masks.items()})
        self.reference = None
        ref_path = args.reference or self.f.get('source')
        if ref_path and Path(ref_path).exists():
            self.reference = pf.load_image(ref_path, PREVIEW_EDGE, raw_half=True)
        self.state0 = self._state()
        self.view = 'After'
        self.region = 'subject' if self.masks is not None else 'background'
        self.band = 'red'
        self._quiet = False
        self._build()
        self.update()

    # -- settings -----------------------------------------------------------
    def _state(self):
        return copy.deepcopy({'controls': self.f['controls'], 'hsl': self.f['hsl'], 'local': self.f['local']})

    def render(self, x, masks, spatial_edge=None):
        return pf.render(x, self.models, self.f['controls'], self.f['hsl'], masks, self.f['local'])

    # -- UI -------------------------------------------------------------------
    def _build(self):
        self.fig = plt.figure(figsize=(16, 9))
        try:
            self.fig.canvas.manager.set_window_title(f"tune · {self.f['name']} · {Path(self.args.photo).name}")
        except Exception:
            pass
        self.ax = self.fig.add_axes([.01, .06, .60, .90])
        self.ax.axis('off')
        self.im = self.ax.imshow(self.preview)
        self.status = self.fig.text(.01, .015, '', fontsize=9, color='#444')

        def slider(rect, label, lo, hi, val):
            s = Slider(self.fig.add_axes(rect), label, lo, hi, valinit=val)
            s.label.set_fontsize(9)
            s.valtext.set_fontsize(8)
            return s

        c = self.f['controls']
        self.fig.text(.69, .965, 'Look and finishing', fontsize=10, weight='bold')
        self.sl = {}
        y = .93
        for name, lo, hi in SLIDERS:
            self.sl[name] = slider([.69, y, .11, .022], name, lo, hi, c[name])
            y -= .036
        self.fig.text(.69, y + .005, 'Tonal zones (lift)', fontsize=10, weight='bold')
        y -= .035
        self.zone_sl = []
        for k, zname in enumerate(pf.ZONE_NAMES):
            self.zone_sl.append(slider([.69, y, .11, .022], zname.lower(), -.2, .2, c['zones'][k]))
            y -= .036
        for s in list(self.sl.values()) + self.zone_sl:
            s.on_changed(self._on_global)

        # right column
        x0 = .87
        self.fig.text(.835, .965, 'View', fontsize=10, weight='bold')
        views = ['After', 'Before', 'Split', 'Masks'] + (['Reference'] if self.reference is not None else [])
        self.view_radio = RadioButtons(self.fig.add_axes([.835, .80, .07, .155], frameon=False), views)
        self.view_radio.on_clicked(self._on_view)

        self.fig.text(.835, .775, 'Region', fontsize=10, weight='bold')
        regions = pf.REGIONS if self.masks is not None else ['background']
        self.region_radio = RadioButtons(self.fig.add_axes([.835, .69, .08, .08], frameon=False), regions,
                                         active=regions.index(self.region))
        self.region_radio.on_clicked(self._on_region)
        self.local_sl = {}
        y = .655
        for name, lo, hi in LOCAL_SLIDERS:
            self.local_sl[name] = slider([x0, y, .09, .022], name, lo, hi, self.f['local'][self.region][name])
            self.local_sl[name].on_changed(self._on_local)
            y -= .036

        self.fig.text(.835, .54, 'HSL band', fontsize=10, weight='bold')
        self.band_radio = RadioButtons(self.fig.add_axes([.835, .33, .08, .2], frameon=False), pf.HSL_BANDS)
        self.band_radio.on_clicked(self._on_band)
        self.hsl_sl = {}
        y = .295
        for k, (name, lo, hi) in enumerate(HSL_SLIDERS):
            self.hsl_sl[name] = slider([x0, y, .09, .022], name, lo, hi, self.f['hsl'][self.band][k])
            self.hsl_sl[name].on_changed(self._on_hsl)
            y -= .036
        for r in (self.view_radio, self.region_radio, self.band_radio):
            for t in r.labels:
                t.set_fontsize(9)

        self.btn_reset = Button(self.fig.add_axes([.835, .10, .065, .04]), 'Reset')
        self.btn_save = Button(self.fig.add_axes([.905, .10, .085, .04]), 'Save filter')
        self.btn_img = Button(self.fig.add_axes([.835, .05, .155, .04]), 'Save full-size image')
        self.btn_reset.on_clicked(self.reset)
        self.btn_save.on_clicked(self.save_filter)
        self.btn_img.on_clicked(self.save_image)

    # -- callbacks ------------------------------------------------------------
    def _on_global(self, _):
        if self._quiet:
            return
        c = self.f['controls']
        for k, s in self.sl.items():
            c[k] = float(s.val)
        c['zones'] = [float(s.val) for s in self.zone_sl]
        self.update()

    def _on_local(self, _):
        if self._quiet:
            return
        self.f['local'][self.region] = {k: float(s.val) for k, s in self.local_sl.items()}
        self.update()

    def _on_hsl(self, _):
        if self._quiet:
            return
        self.f['hsl'][self.band] = [float(self.hsl_sl[n].val) for n, _, _ in HSL_SLIDERS]
        self.update()

    def _set_quiet(self, pairs):
        self._quiet = True
        for s, v in pairs:
            s.set_val(v)
        self._quiet = False

    def _on_region(self, label):
        self.region = label
        self._set_quiet([(self.local_sl[k], self.f['local'][label][k]) for k in self.local_sl])

    def _on_band(self, label):
        self.band = label
        self._set_quiet([(self.hsl_sl[n], self.f['hsl'][label][k]) for k, (n, _, _) in enumerate(HSL_SLIDERS)])

    def _on_view(self, label):
        self.view = label
        self.update()

    def reset(self, _=None):
        s = copy.deepcopy(self.state0)
        self.f['controls'], self.f['hsl'], self.f['local'] = s['controls'], s['hsl'], s['local']
        c = self.f['controls']
        self._set_quiet([(self.sl[k], c[k]) for k in self.sl] + list(zip(self.zone_sl, c['zones'])))
        self._on_region(self.region)
        self._on_band(self.band)
        self.update()

    # -- drawing / saving -----------------------------------------------------
    def update(self):
        if self.view == 'Reference':
            img = self.reference
        elif self.view == 'Before':
            img = self.preview
        elif self.view == 'Masks':
            img = (pf.mask_overlay(self.preview, self.preview_masks) if self.preview_masks is not None
                   else self.preview)
        else:
            img = np.clip(self.render(self.preview, self.preview_masks), 0, 1)
            if self.view == 'Split':
                img = img.copy()
                w = img.shape[1] // 2
                img[:, :w] = self.preview[:, :w]
                img[:, w - 1:w + 1] = 1
        self.im.set_data(img)
        self.im.set_extent((-.5, img.shape[1] - .5, img.shape[0] - .5, -.5))
        self.ax.set_xlim(-.5, img.shape[1] - .5)
        self.ax.set_ylim(img.shape[0] - .5, -.5)
        regions = pf.fitted_regions(self.models)
        self.status.set_text(f"{self.f['name']} on {Path(self.args.photo).name} · view: {self.view} · "
                             f"regions matched: {', '.join(regions) or 'global only'}")
        self.fig.canvas.draw_idle()

    def save_filter(self, _=None):
        out = Path(self.args.save_as or self.args.filter)
        pf.save_filter(self.f, out)
        self.status.set_text(f'Saved filter settings -> {out}')
        self.fig.canvas.draw_idle()
        return out

    def save_image(self, _=None):
        out = self.render(self.photo, self.masks)
        dst = Path(self.args.outdir) / f"{Path(self.args.photo).stem}_{self.f['name']}.{self.args.format}"
        pf.save_image(out, dst)
        self.status.set_text(f'Saved image -> {dst}')
        self.fig.canvas.draw_idle()
        return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('filter', help='filter .json made by make_filter.py')
    ap.add_argument('photo')
    ap.add_argument('--reference', help='reference photo to show (default: the one the filter was made from)')
    ap.add_argument('--save-as', help='save settings to a new filter .json instead of overwriting')
    ap.add_argument('-o', '--outdir', default=str(Path(__file__).parent / 'out'))
    ap.add_argument('--format', default='jpg', choices=['jpg', 'png', 'tif'])
    ap.add_argument('--raw-ev', type=float, default=0.0, help='exposure for RAW decoding, in stops')
    add_mask_args(ap, 'photo')
    args = ap.parse_args(argv)
    t = Tuner(args)
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    return t


if __name__ == '__main__':
    main()
