"""Stage 1: photos -> 2D detections (+ optional metric depth per detection).

usage:
  python detect/detect.py detect/samples/*.jpg -o detect/out/detections.json [--depth] [--viz]

Detector: YOLO-World (open vocabulary; we hand it the PROMPTS list).
Depth:    Depth Anything V2, metric indoor model -> meters per pixel. Saved as
          <stem>_depth.npy; place.py uses it for object height and to place objects
          whose floor contact is hidden.
"""
import argparse, json, pathlib, sys

import numpy as np
from PIL import Image, ExifTags

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from priors import PROMPTS, LABEL_TO_CLASS  # noqa: E402
from camera import box_depth  # noqa: E402

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DET_MODEL = str(pathlib.Path(__file__).resolve().parent / "weights" / "yolov8s-worldv2.pt")


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def focal_px(img: Image.Image, hfov_deg: float) -> tuple[float, str]:
    """Focal length in pixels: from EXIF 35mm-equivalent focal if present, else from an
    assumed horizontal FOV (phone main cameras are ~65-70 deg)."""
    w = img.width
    try:
        exif = img.getexif().get_ifd(ExifTags.IFD.Exif)
        f35 = exif.get(ExifTags.Base.FocalLengthIn35mmFilm)
        if f35:
            return w * float(f35) / 36.0, f"exif {f35}mm"
    except Exception:
        pass
    return (w / 2) / np.tan(np.radians(hfov_deg) / 2), f"assumed hfov {hfov_deg}"


def load_detector():
    from ultralytics import YOLOWorld
    m = YOLOWorld(DET_MODEL)
    m.set_classes(PROMPTS)
    return m


def load_depth():
    from transformers import pipeline
    return pipeline("depth-estimation", model=DEPTH_MODEL, device=device())


def depth_map(pipe, img: Image.Image) -> np.ndarray:
    import torch
    pred = pipe(img)["predicted_depth"]
    if pred.dim() == 2:
        pred = pred[None]
    pred = torch.nn.functional.interpolate(pred[None].float(), size=(img.height, img.width),
                                           mode="bilinear", align_corners=False)
    return pred[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default="detect/out/detections.json")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--depth", action="store_true", help="also run metric depth (slower)")
    ap.add_argument("--hfov", type=float, default=65.0, help="assumed horizontal FOV if no EXIF")
    ap.add_argument("--viz", action="store_true", help="save annotated images next to the output")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    det = load_detector()
    dpipe = load_depth() if args.depth else None

    results = []
    for path in args.images:
        img = Image.open(path).convert("RGB")
        f, f_src = focal_px(Image.open(path), args.hfov)
        r = det.predict(path, conf=args.conf, device=device(), verbose=False)[0]
        D = depth_map(dpipe, img) if dpipe else None

        dets = []
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = PROMPTS[int(cls)]
            d = {"label": label, "class": LABEL_TO_CLASS.get(label, label),
                 "conf": round(conf, 3), "box": [round(v, 1) for v in box]}
            if D is not None:
                z = box_depth(D, box)
                d["depth_m"] = round(z, 3) if z else None
                d["height_m"] = round((box[3] - box[1]) * z / f, 3) if z else None
                d["width_m"] = round((box[2] - box[0]) * z / f, 3) if z else None
            dets.append(d)
        dets.sort(key=lambda d: -d["conf"])
        entry = {"image": str(path), "width": img.width, "height": img.height,
                 "focal_px": round(f, 1), "focal_source": f_src, "detections": dets}
        if D is not None:
            dpath = out.parent / (pathlib.Path(path).stem + "_depth.npy")
            np.save(dpath, D.astype(np.float32))
            entry["depth_map"] = str(dpath)
        results.append(entry)
        print(f"{path}: {len(dets)} detections " + ", ".join(f"{d['class']}({d['conf']:.2f})" for d in dets))

        if args.viz:
            vis = r.plot()
            if D is not None:
                import cv2
                dn = cv2.applyColorMap((255 * (1 - D / max(D.max(), 1e-6))).astype(np.uint8), cv2.COLORMAP_MAGMA)
                vis = np.hstack([vis, dn])
            import cv2
            cv2.imwrite(str(out.parent / (pathlib.Path(path).stem + "_det.jpg")), vis)

    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
