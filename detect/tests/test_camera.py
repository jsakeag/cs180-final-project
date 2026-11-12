"""run: python detect/tests/test_camera.py"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from homography import compute_h, apply_h  # noqa: E402
from camera import K_matrix, estimate_focal, pose_from_floor_h, pixel_to_world, plan_point_depth  # noqa: E402
from test_homography import pinhole_floor_h  # noqa: E402

f, cx, cy, cam_h, cam_x, cam_z = 500, 320, 240, 1.5, 2.0, 6.0
H_true = pinhole_floor_h(f=f, cx=cx, cy=cy, cam_h=cam_h, pitch_deg=25, cam_x=cam_x, cam_z=cam_z, yaw_deg=10)

# simulate the clicking workflow: 4 floor points -> H (img->plan) -> invert
floor = np.array([[0.5, 1.0], [4.0, 1.0], [4.0, 4.0], [0.5, 4.0]])
img = apply_h(H_true, floor)
H_i2p = compute_h(img, floor)
H_p2i = np.linalg.inv(H_i2p)

f_est = estimate_focal(H_p2i, cx, cy)
assert abs(f_est - f) < 1.0, f_est

K = K_matrix(f_est, cx, cy)
pose = pose_from_floor_h(H_p2i, K, floor[0])
assert np.allclose(pose["C"][:2], [cam_x, cam_z], atol=1e-3), pose["C"]
assert abs(pose["height"] - cam_h) < 1e-3, pose["height"]

# a point 0.8 m above the floor at plan (3, 2): project it, lift it back with its depth
W_up = np.array([3.0, 2.0])
# build its camera coords from the pose (height along up direction)
Xw = np.array([W_up[0], W_up[1], 0.8 * pose["up_sign"]])
Xc = pose["R"] @ Xw + pose["t"]
uv = (K @ Xc)[:2] / Xc[2]
back = pixel_to_world(pose, K, uv[0], uv[1], Xc[2])
assert np.allclose(back, [3.0, 2.0, 0.8], atol=1e-6), back

# floor point depth matches projecting it
assert abs(plan_point_depth(pose, floor[2]) - (pose["R"] @ [*floor[2], 0] + pose["t"])[2]) < 1e-12
print(f"camera tests passed (f est {f_est:.1f}, cam at {pose['C'][:2].round(3)}, h {pose['height']:.3f})")
