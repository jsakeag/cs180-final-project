"""run: python detect/tests/test_homography.py"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from homography import compute_h, apply_h, reprojection_error  # noqa: E402

rng = np.random.default_rng(0)


def pinhole_floor_h(f=500, cx=320, cy=240, cam_h=1.5, pitch_deg=25, cam_x=2.0, cam_z=6.0, yaw_deg=10):
    """Ground-truth floor->image homography of a real pinhole camera, so the test
    exercises perspective (not just an affine map)."""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    # world: x right, y up, z toward viewer; camera looks along -z, pitched down
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    R_wc = Ry @ Rx                      # camera->world
    flip = np.diag([1, -1, -1])         # world-camera(-z fwd, y up) -> opencv (z fwd, y down)
    R = flip @ R_wc.T
    C = np.array([cam_x, cam_h, cam_z])
    t = -R @ C
    # floor point (x, 0, z): image ~ K [r1 r3 t] [x z 1]
    return K @ np.c_[R[:, 0], R[:, 2], t]


def main():
    H_floor_to_img = pinhole_floor_h()
    H_img_to_floor_true = np.linalg.inv(H_floor_to_img)

    # exact recovery from 4 clicks (corners of a 4 x 3 m patch in front of the camera)
    floor = np.array([[0.5, 1.0], [4.0, 1.0], [4.0, 4.0], [0.5, 4.0]])
    img = apply_h(H_floor_to_img, floor)
    H = compute_h(img, floor)
    assert reprojection_error(H, img, floor) < 1e-9
    test_floor = rng.uniform([0.5, 1.0], [4.0, 4.0], size=(50, 2))
    assert np.allclose(apply_h(H, apply_h(H_floor_to_img, test_floor)), test_floor, atol=1e-6)

    # noisy clicks (+-2 px) with 8 points: least squares stays within a few cm
    floor8 = rng.uniform([0.0, 0.5], [4.5, 4.5], size=(8, 2))
    img8 = apply_h(H_floor_to_img, floor8) + rng.normal(0, 2, size=(8, 2))
    H8 = compute_h(img8, floor8)
    err = np.linalg.norm(apply_h(H8, apply_h(H_floor_to_img, test_floor)) - test_floor, axis=1)
    assert err.mean() < 0.05, err.mean()

    print(f"homography tests passed (noisy mean err {err.mean() * 100:.1f} cm)")


if __name__ == "__main__":
    main()
