"""photofilter: turn a reference photo into a reusable Instagram-style filter.

Method (ported from the "Reference Lab RAW Style Matching" notebook, global mode)
-------------------------------------------------------------------------------
Everything works on display-referred sRGB values in [0, 1].

1. Light curve. Match 25 quantiles (1%..99%) of the photo's luma to the
   reference's with a monotonic PCHIP curve whose ends are pinned (0 -> 0,
   1 -> 1). Each point may move at most 0.30, and the curve is blended
   85% fitted / 15% identity. Applied as an additive luma shift, so colour
   differences are kept.
2. RGB curves. After step 1, fit the same kind of curve per channel (R, G, B;
   max shift 0.18). Only their colour part is used: the luma of the change is
   removed so they don't fight the light curve.
3. Zone offsets. Five overlapping tonal zones centred at 0, .25, .5, .75, 1
   (blacks, shadows, midtones, highlights, whites). In each zone, the mean RGB
   difference between reference and photo (capped at 0.15, luma removed,
   down-weighted when a zone has few pixels).
4. Mix. colour change = 0.65 * RGB-curve change + 0.35 * zone offsets.
   Defaults: strength 0.85, tone 0.85, colour 0.80.
5. Finishing controls: exposure, temperature, tint, contrast, saturation,
   vibrance, per-zone lift, an 8-band HSL mixer, clarity and texture.

6. Regions (optional). With subject / foreground masks on the reference and
   the photo, each region present in both gets its own fit (steps 1-3),
   plus per-region strength, exposure and saturation. Anything not covered
   is background. Regions missing or < 1.5% of either image fall back to the
   global fit. Masks come from white-on-black PNGs, or the subject mask can
   be detected automatically with rembg (u2netp).

A saved filter (.json) holds only the reference's statistics (globally and
per region) plus control settings, so it can be applied to any photo: the
curves are fitted against each new photo at apply time, as the notebook does.

Requires numpy, scipy and Pillow. Optional: rawpy (RAW input), tifffile
(16-bit TIFF output), matplotlib (HSL mixer, tune.py), rembg (auto subject).
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

FILTER_FORMAT = 'photofilter-1'
SEED = 41
FIT_LONG_EDGE = 1280         # photos are fitted at this size, applied at full size
SAMPLES = 40000              # pixels sampled for fitting
QS = np.linspace(.01, .99, 25)
GRID = np.linspace(0, 1, 1024, dtype='float32')
CENTERS = np.linspace(0, 1, 5, dtype='float32')
ZONE_NAMES = ['Blacks', 'Shadows', 'Midtones', 'Highlights', 'Whites']
HSL_BANDS = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']
HUE_CENTERS = np.array([0, 30, 60, 120, 180, 240, 280, 320]) / 360
RAW_EXT = {'.arw', '.cr2', '.cr3', '.nef', '.nrw', '.raf', '.rw2', '.orf', '.dng',
           '.pef', '.srw', '.raw'}

DEFAULT_CONTROLS = dict(strength=.85, tone=.85, color=.80, exposure=0., temperature=0.,
                        tint=0., contrast=0., saturation=1., vibrance=0., clarity=0.,
                        texture=0., zones=[0., 0., 0., 0., 0.])
DEFAULT_HSL = {name: [0., 0., 0.] for name in HSL_BANDS}  # [hue deg, sat, lightness]
REGIONS = ['subject', 'foreground', 'background']
DEFAULT_LOCAL = dict(strength=1., exposure=0., saturation=1.)
MIN_REGION = .015            # regions smaller than this fraction use the global fit

# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------

def srgb_encode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055).astype('float32')


def srgb_decode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4).astype('float32')


def light(x):
    """Display-referred luma (deliberately not linear luminance)."""
    return x @ np.array([.2126, .7152, .0722], dtype='float32')


def zone_weights(y):
    z = np.maximum(1 - np.abs(y[..., None] - CENTERS) / .25, 0)
    return z / np.maximum(z.sum(-1, keepdims=True), 1e-8)


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------

def resize_float(x, shape):
    h, w = shape
    if x.shape[:2] == (h, w):
        return x.astype('float32', copy=False)
    if x.ndim == 2:
        return np.asarray(Image.fromarray(x.astype('float32')).resize((w, h), Image.Resampling.BILINEAR)).copy()
    return np.stack([resize_float(x[..., i], shape) for i in range(x.shape[-1])], axis=-1)


def downsize(x, edge):
    h, w = x.shape[:2]
    scale = min(1, edge / max(h, w)) if edge else 1
    return resize_float(x, (max(1, round(h * scale)), max(1, round(w * scale))))


def load_image(path, edge=0, raw_half=False, raw_ev=0.0):
    """Load a photo as float32 sRGB in [0, 1]. edge=0 keeps full size.

    RAW files (needs rawpy): camera white balance, no auto-brightness, linear
    16-bit decode, then sRGB encoding. raw_ev brightens/darkens the decode.
    Other files: EXIF-rotated and converted to sRGB using their ICC profile.
    """
    if Path(path).suffix.lower() in RAW_EXT:
        try:
            import rawpy
        except ImportError as e:
            raise ImportError('RAW files need rawpy: pip install rawpy') from e
        with rawpy.imread(str(path)) as raw:
            arr = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                                  output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1), output_bps=16,
                                  half_size=raw_half, highlight_mode=rawpy.HighlightMode.ReconstructDefault)
        x = srgb_encode(arr.astype('float32') / 65535 * 2 ** raw_ev)
        del arr
    else:
        with Image.open(path) as im:
            icc = im.info.get('icc_profile')
            im = ImageOps.exif_transpose(im)
            if icc:
                im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                               ImageCms.createProfile('sRGB'), outputMode='RGB')
            x = np.asarray(im.convert('RGB'), dtype='float32') / 255
    return np.clip(downsize(x, edge), 0, 1).astype('float32')


def _srgb_icc():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def save_image(rgb, path, quality=95):
    """Save as JPEG/PNG (8-bit), or 16-bit TIFF for .tif/.tiff (needs tifffile)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb, 0, 1)
    icc = _srgb_icc()
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        tifffile.imwrite(path, np.rint(rgb * 65535).astype('uint16'), photometric='rgb',
                         metadata=None, extratags=[(34675, 'B', len(icc), icc, False)])
    else:
        Image.fromarray(np.uint8(np.rint(rgb * 255))).save(path, quality=quality, icc_profile=icc)


# --------------------------------------------------------------------------
# Reference statistics (what a saved filter contains)
# --------------------------------------------------------------------------

def sample_pixels(x, mask=None, n=SAMPLES):
    """Up to n pixels (N x 3), from where mask > .55 if a mask is given.
    Returns None if fewer than 64 pixels qualify."""
    flat = x.reshape(-1, 3)
    idx = np.arange(len(flat)) if mask is None else np.flatnonzero(mask.ravel() > .55)
    if len(idx) < 64:
        return None
    if len(idx) > n:
        idx = np.random.default_rng(SEED).choice(idx, n, replace=False)
    return flat[idx]


# --------------------------------------------------------------------------
# Region masks
# --------------------------------------------------------------------------

_rembg_session = None


def read_mask(path, shape):
    """White-on-black image -> float mask in [0, 1] resized to shape (h, w)."""
    with Image.open(path) as im:
        m = np.asarray(ImageOps.exif_transpose(im).convert('L'), dtype='float32') / 255
    return np.clip(resize_float(m, shape), 0, 1)


def auto_subject_mask(x):
    """Salient-subject mask from rembg's small u2netp model (pip install "rembg[cpu]").
    It finds a salient subject, not semantic classes: inspect the result."""
    global _rembg_session
    try:
        from rembg import remove, new_session
    except ImportError as e:
        raise ImportError('Automatic subject masks need rembg: pip install "rembg[cpu]"') from e
    if _rembg_session is None:
        _rembg_session = new_session('u2netp', providers=['CPUExecutionProvider'])
    small = downsize(x, 768)
    m = remove(Image.fromarray(np.uint8(np.clip(small, 0, 1) * 255)), session=_rembg_session, only_mask=True)
    return resize_float(np.asarray(m.convert('L'), dtype='float32') / 255, x.shape[:2])


def region_masks(x, subject=None, foreground=None, auto_subject=False):
    """Soft subject / foreground / background masks for image x that sum to 1.

    subject, foreground: a path to a white-on-black image, an array, or None.
    auto_subject: detect the subject with rembg when no subject mask is given.
    Subject wins over foreground; everything not covered is background.
    """
    shape = x.shape[:2]

    def get(m):
        if m is None:
            return np.zeros(shape, dtype='float32')
        if isinstance(m, (str, Path)):
            return read_mask(m, shape)
        return np.clip(resize_float(np.asarray(m, dtype='float32'), shape), 0, 1)

    s = auto_subject_mask(x) if subject is None and auto_subject else get(subject)
    s = np.clip(gaussian_filter(s, 1.0), 0, 1)
    fg = np.clip(gaussian_filter(get(foreground), 1.0), 0, 1) * (1 - s)
    return {'subject': s, 'foreground': fg, 'background': np.clip(1 - s - fg, 0, 1)}


def mask_overlay(x, masks):
    """Preview: red = subject, green = foreground, blue = background."""
    shape = x.shape[:2]
    m = [resize_float(masks[k], shape) for k in REGIONS]
    return np.clip(.55 * x + .45 * np.stack(m, -1), 0, 1)


def _has_regions(masks):
    return masks is not None and any(float(masks[k].mean()) > 0 for k in ('subject', 'foreground'))


def reference_stats(b):
    """Everything the fit needs from the reference's sampled pixels b (N x 3)."""
    zb = zone_weights(light(b))
    wsum = zb.sum(0)
    means = (zb.T @ b) / np.maximum(wsum, 1)[:, None]
    return {'light_q': np.quantile(light(b), QS).tolist(),
            'rgb_q': [np.quantile(b[:, c], QS).tolist() for c in range(3)],
            'zone_mean': means.tolist(),
            'zone_weight': wsum.tolist()}


def make_filter(reference_path, name=None, controls=None, hsl=None, local=None,
                subject_mask=None, foreground_mask=None, auto_subject=False):
    """Build a filter from a reference photo (any image, or RAW with rawpy).

    Give subject_mask / foreground_mask (paths or arrays) or auto_subject=True
    to also store per-region statistics of the reference.
    """
    p = Path(reference_path)
    ref = load_image(p, FIT_LONG_EDGE, raw_half=True)
    stats = {'global': reference_stats(sample_pixels(ref))}
    coverage = {}
    if subject_mask is not None or foreground_mask is not None or auto_subject:
        masks = region_masks(ref, subject_mask, foreground_mask, auto_subject)
        for k in REGIONS:
            coverage[k] = float(masks[k].mean())
            b = sample_pixels(ref, masks[k])
            if b is not None and coverage[k] >= MIN_REGION:
                stats[k] = reference_stats(b)
    return {'format': FILTER_FORMAT,
            'name': name or p.stem,
            'source': str(p.resolve()),
            'stats': stats,
            'region_coverage': coverage,
            'controls': {**DEFAULT_CONTROLS, **(controls or {})},
            'hsl': {**DEFAULT_HSL, **(hsl or {})},
            'local': {k: {**DEFAULT_LOCAL, **(local or {}).get(k, {})} for k in REGIONS}}


def save_filter(f, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(f, indent=2))


def load_filter(path):
    f = json.loads(Path(path).read_text())
    if f.get('format') != FILTER_FORMAT:
        raise ValueError(f'{path} is not a filter made by make_filter.py')
    f.setdefault('local', {})
    f['local'] = {k: {**DEFAULT_LOCAL, **f['local'].get(k, {})} for k in REGIONS}
    f['controls'] = {**DEFAULT_CONTROLS, **f.get('controls', {})}
    f['hsl'] = {**DEFAULT_HSL, **f.get('hsl', {})}
    return f


# --------------------------------------------------------------------------
# Fitting (same maths as the notebook's fit_curve / fit_model)
# --------------------------------------------------------------------------

def _curve(xa, yb, max_shift, flat=False):
    # Duplicate quantiles occur in flat/clipped areas; never pass them to PCHIP.
    ux, inv = np.unique(xa, return_inverse=True)
    uy = np.bincount(inv, weights=yb) / np.bincount(inv)
    if len(ux) < 3 or flat:
        return GRID.copy()
    inside = (ux > 1e-5) & (ux < 1 - 1e-5)
    xx = np.r_[0, ux[inside], 1]
    yy = np.r_[0, uy[inside], 1]
    lut = PchipInterpolator(xx, np.maximum.accumulate(yy))(GRID)
    lut = np.clip(lut, GRID - max_shift, GRID + max_shift)
    return np.clip(.15 * GRID + .85 * lut, 0, 1).astype('float32')


def apply_light(x, lut):
    y = light(x)
    yt = np.interp(y, GRID, lut).astype('float32')
    return x + (yt - y)[..., None]


def fit_model(a, stats):
    """Fit the transform from photo samples a (N x 3) to the reference stats."""
    ya = light(a)
    lum = _curve(np.quantile(ya, QS), np.asarray(stats['light_q']), .30, np.ptp(ya) < 1e-4)
    at = apply_light(a, lum)
    curves = []
    for c in range(3):
        ac = np.clip(at[:, c], 0, 1)
        curves.append(_curve(np.quantile(ac, QS), np.asarray(stats['rgb_q'][c]), .18, np.ptp(ac) < 1e-4))
    za = zone_weights(np.clip(light(at), 0, 1))
    offsets = []
    for k in range(5):
        wa = za[:, k]
        ma = (at * wa[:, None]).sum(0) / max(wa.sum(), 1)
        mb = np.asarray(stats['zone_mean'][k])
        d = np.clip(mb - ma, -.15, .15)
        d -= light(d)
        reliability = min(1, wa.sum() / 200, stats['zone_weight'][k] / 200)
        offsets.append(d * reliability)
    return {'light': lum, 'rgb': np.stack(curves), 'zones': np.asarray(offsets, dtype='float32')}


def fit(photo, f, masks=None):
    """Fit a filter to a photo (float sRGB array, any size).

    Returns {'global': model, <region>: model, ...}. Regions are fitted only
    where both the photo's mask and the filter's reference cover >= 1.5%.
    """
    small = downsize(photo, FIT_LONG_EDGE)
    stats = f['stats']
    models = {'global': fit_model(sample_pixels(small), stats['global'])}
    if not _has_regions(masks):
        return models
    for k in REGIONS:
        m = resize_float(masks[k], small.shape[:2])
        a = sample_pixels(small, m)
        cov = float(m.mean())
        if a is None or k not in stats or cov < MIN_REGION:
            if k != 'background' and max(cov, f.get('region_coverage', {}).get(k, 0)) > MIN_REGION:
                warnings.warn(f'{k}: missing or too small in the photo or the reference; using the global grade.')
            continue
        models[k] = fit_model(a, stats[k])
    return models


def fitted_regions(models):
    return [k for k in models if k != 'global']


# --------------------------------------------------------------------------
# Rendering (same maths as the notebook's apply_model / render, global mode)
# --------------------------------------------------------------------------

def apply_model(x, model, tone=1, color=1):
    base = apply_light(x, model['light'])
    base = x + tone * (base - x)
    curved = np.stack([np.interp(np.clip(base[..., c], 0, 1), GRID, model['rgb'][c])
                       for c in range(3)], -1).astype('float32')
    delta = curved - base
    delta -= light(delta)[..., None]
    offsets = zone_weights(np.clip(light(base), 0, 1)) @ np.asarray(model['zones'], dtype='float32')
    return base + color * (.65 * delta + .35 * offsets)


def hsl_mixer(x, hsl):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(np.clip(x, 0, 1))
    h, s, v = np.moveaxis(hsv, -1, 0)
    dh = np.zeros_like(h); ds = dh.copy(); dv = dh.copy(); total = dh.copy()
    for center, name in zip(HUE_CENTERS, HSL_BANDS):
        settings = hsl.get(name, [0, 0, 0])
        distance = np.abs((h - center + .5) % 1 - .5)
        w = np.exp(-.5 * (distance / (24 / 360)) ** 2)
        dh += w * settings[0] / 360; ds += w * settings[1]; dv += w * settings[2]; total += w
    neutral = np.clip(s / .15, 0, 1)
    hsv[..., 0] = (h + dh / np.maximum(total, 1e-6) * neutral) % 1
    hsv[..., 1] = np.clip(s * (1 + ds / np.maximum(total, 1e-6)), 0, 1)
    hsv[..., 2] = np.clip(v + dv / np.maximum(total, 1e-6) * neutral, 0, 1)
    return hsv_to_rgb(hsv).astype('float32')


def render(x, models, controls=None, hsl=None, masks=None, local=None, spatial=True):
    """Apply fitted models plus finishing controls. Returns unclipped float32.

    models: what fit() returns (a single model dict also works).
    masks:  region masks (any resolution; resized and normalised here).
            None = the whole image is one region using the global fit.
    local:  {region: {strength, exposure, saturation}}.
    spatial=False skips clarity/texture (the only non per-pixel steps), which
    is what a LUT can represent.
    """
    if 'light' in models:
        models = {'global': models}
    c = {**DEFAULT_CONTROLS, **(controls or {})}
    hsl = hsl or {}
    local = local or {}
    if masks is None:
        masks = {'background': np.ones(x.shape[:-1], dtype='float32')}
    else:
        masks = {k: resize_float(v, x.shape[:2]) if v.shape != x.shape[:-1] else v for k, v in masks.items()}
        total = sum(masks.values())
        masks = {k: v / np.maximum(total, 1e-8) for k, v in masks.items()}
    out = np.zeros_like(x)
    cache = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        m = models.get(name, models['global'])
        if id(m) not in cache:
            cache[id(m)] = apply_model(x, m, c['tone'], c['color'])
        loc = {**DEFAULT_LOCAL, **local.get(name, {})}
        graded = x + loc['strength'] * (cache[id(m)] - x)
        graded = srgb_encode(srgb_decode(graded) * 2 ** loc['exposure'])
        gy = light(graded)[..., None]
        graded = gy + (graded - gy) * loc['saturation']
        out += graded * mask[..., None]
    out = x + c['strength'] * (out - x)
    out = srgb_encode(srgb_decode(out) * 2 ** c['exposure'])
    out *= np.array([1 + c['temperature'] * .15, 1 + c['tint'] * .10, 1 - c['temperature'] * .15],
                    dtype='float32')
    y = light(out)
    out += (zone_weights(np.clip(y, 0, 1)) @ np.asarray(c['zones'], dtype='float32'))[..., None]
    out = (out - .5) * (1 + c['contrast']) + .5
    y = light(out)
    chroma = out - y[..., None]
    sat = np.max(out, -1) - np.min(out, -1)
    out = y[..., None] + chroma * (c['saturation'] + c['vibrance'] * (1 - np.clip(sat, 0, 1)))[..., None]
    if any(any(v) for v in hsl.values()):
        out = hsl_mixer(out, hsl)
    if spatial and x.ndim == 3 and min(x.shape[:2]) > 1:
        edge = max(x.shape[:2])
        y = light(out)
        for key, scale in [('clarity', .008), ('texture', .0012)]:
            if c[key]:
                detail = y - gaussian_filter(y, max(.5, edge * scale), mode='reflect', truncate=4)
                out += c[key] * detail[..., None]
    return out.astype('float32')


def apply_filter(photo, f, masks=None, **overrides):
    """Fit and apply a filter. overrides replace the filter's saved controls.
    masks: from region_masks() on the photo (computing them on
    downsize(photo, FIT_LONG_EDGE) matches the notebook exactly)."""
    controls = {**f['controls'], **overrides}
    return render(photo, fit(photo, f, masks), controls, f.get('hsl'), masks, f.get('local'))


# --------------------------------------------------------------------------
# Export as a 3D LUT (.cube)
# --------------------------------------------------------------------------

def bake_lut(model, controls=None, hsl=None, size=33):
    """Evaluate a fitted model on an RGB grid -> (size, size, size, 3) LUT
    indexed [b, g, r] (.cube ordering: red changes fastest). Clarity and
    texture are spatial and can't be stored in a LUT, and neither can regions:
    pass one model, e.g. fit(...)['global']."""
    if 'global' in model:
        model = model['global']
    g = np.linspace(0, 1, size, dtype='float32')
    bb, gg, rr = np.meshgrid(g, g, g, indexing='ij')
    grid = np.stack([rr, gg, bb], axis=-1).reshape(-1, 1, 3)
    out = render(grid, model, controls, hsl, spatial=False)
    return np.clip(out, 0, 1).reshape(size, size, size, 3)


def save_cube(lut, path, title='photofilter'):
    size = lut.shape[0]
    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {size}',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0']
    lines += [f'{r:.6f} {g:.6f} {b:.6f}' for r, g, b in lut.reshape(-1, 3)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def apply_lut(rgb, lut):
    """Apply a [b, g, r]-indexed 3D LUT with trilinear interpolation."""
    n = lut.shape[0]
    x = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    t = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    tr, tg, tb = t[..., :1], t[..., 1:2], t[..., 2:3]
    out = 0
    for db, wb in ((0, 1 - tb), (1, tb)):
        for dg, wg in ((0, 1 - tg), (1, tg)):
            for dr, wr in ((0, 1 - tr), (1, tr)):
                out = out + wb * wg * wr * lut[b0 + db, g0 + dg, r0 + dr]
    return out
