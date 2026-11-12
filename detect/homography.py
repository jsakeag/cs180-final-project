"""Image <-> floor-plan homography (same model as proj4a: p' ~ H p, then divide by w).

The floor is a plane, so any pixel that lies on the floor maps to floor-plan
coordinates (x, z) in meters through a single 3x3 H. 4 clicked pairs are the
minimum; more pairs are solved by least squares.
"""
import numpy as np


def _normalize(pts):
    """Hartley normalization: centroid to origin, mean distance sqrt(2). Keeps the
    least-squares system well conditioned when mixing pixels (~1000) and meters (~1)."""
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    s = np.sqrt(2) / max(np.linalg.norm(pts - c, axis=1).mean(), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, T


def compute_h(src, dst):
    """H with dst ~ H @ src. src, dst: (N, 2), N >= 4. Solved as the null vector
    of the stacked DLT system (smallest singular vector)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    assert src.shape == dst.shape and len(src) >= 4, "need >= 4 point pairs"
    s, Ts = _normalize(src)
    d, Td = _normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H, pts):
    """Map (N, 2) points through H with the divide-by-w step."""
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def reprojection_error(H, src, dst):
    """Mean distance (in dst units, i.e. meters) between H(src) and dst."""
    return float(np.linalg.norm(apply_h(H, src) - np.asarray(dst, float), axis=1).mean())
