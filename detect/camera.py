"""Recover the camera from the floor homography.

For a plane, H_plan->img = K [r1 r2 t] (up to scale), where r1, r2 are the floor's
plan-x / plan-z axes in camera coordinates and t is the plan origin in camera coords.
So with K we get the full camera pose relative to the floor plan, and from that any
pixel + depth can be lifted to a plan position. Without EXIF, K's focal length can
itself be estimated from H because r1, r2 must be orthonormal.

World frame here: (plan x, plan z, n) with n = r1 x r2 (right-handed). n is
perpendicular to the floor; `up_sign` says whether +n points up or down.
"""
import numpy as np


def K_matrix(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def estimate_focal(H_p2i, cx, cy):
    """Focal length (px) implied by H, assuming square pixels and principal point at
    the image center. Returns None if H doesn't constrain it (e.g. camera looking
    straight down) or the estimate is unusable."""
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    h = T @ H_p2i
    h1, h2 = h[:, 0], h[:, 1]
    ests = []
    # r1 . r2 = 0
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        ests.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
    # |r1| = |r2|
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        ests.append(-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den)
    ests = [e for e in ests if e > 0]
    return float(np.sqrt(np.median(ests))) if ests else None


def pose_from_floor_h(H_p2i, K, ref_plan_pt):
    """Returns dict with R (world->cam), t, C (camera center, world), height (m),
    up_sign. ref_plan_pt: any plan point known to be in front of the camera
    (e.g. a clicked point) to fix the overall sign."""
    M = np.linalg.inv(K) @ H_p2i
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if (M @ np.array([ref_plan_pt[0], ref_plan_pt[1], 1.0]))[2] * lam < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)        # nearest true rotation
    R = U @ Vt
    C = -R.T @ t
    up_sign = 1.0 if C[2] > 0 else -1.0
    return {"R": R, "t": t, "C": C, "height": abs(C[2]), "up_sign": up_sign}


def pixel_to_world(pose, K, u, v, depth):
    """Pixel + z-depth (along the optical axis, meters) -> (plan x, plan z, height)."""
    X_cam = depth * (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    W = pose["R"].T @ (X_cam - pose["t"])
    return np.array([W[0], W[1], W[2] * pose["up_sign"]])


def plan_point_depth(pose, plan_pt):
    """Camera z-depth of a floor point (used to calibrate the depth model's scale)."""
    W = np.array([plan_pt[0], plan_pt[1], 0.0])
    return float((pose["R"] @ W + pose["t"])[2])


def box_depth(D, box):
    """Robust object distance: median of the central 50% of the box (edges are often
    background)."""
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    patch = D[max(int(cy - hh), 0):int(cy + hh) + 1, max(int(cx - hw), 0):int(cx + hw) + 1]
    return float(np.median(patch)) if patch.size else None
