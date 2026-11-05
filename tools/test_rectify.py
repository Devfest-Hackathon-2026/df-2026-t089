"""Check tools/rectify.py against a synthetic camera view of a known wall.

The camera image is rendered by casting a ray through every pixel and intersecting it with the wall, which is
independent of the projection code being tested. Rectifying that image must give back the wall's pattern.

  python tools/test_rectify.py
"""
import math
import numpy as np
import rectify

W, H, F = 480, 360, 400.0
FLOOR_Y = -1.3


def pattern(u, v):
    """Vertical stripes 0.5 m wide and a bright band between 0.6 m and 0.8 m above the floor."""
    stripes = (np.floor(u / 0.5).astype(int) % 2) * 120 + 60
    band = ((v >= FLOOR_Y + 0.6) & (v < FLOOR_Y + 0.8)) * 70
    return np.clip(stripes + band, 0, 255).astype(np.uint8)


def look_at(position, target):
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -forward, position
    return m


def render(cam):
    """Wall is the plane z = -3 seen from the front; u = x, v = y."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    d_cam = np.stack([(xs - W / 2) / F, -(ys - H / 2) / F, -np.ones_like(xs)], -1)
    d = d_cam @ cam[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):  # rays parallel to the wall never hit it
        t = (-3.0 - cam[2, 3]) / d[..., 2]
        hit = cam[:3, 3] + t[..., None] * d
        gray = pattern(np.nan_to_num(hit[..., 0]), np.nan_to_num(hit[..., 1]))
    return np.repeat(gray[..., None], 3, -1)


def frame_for(cam):
    return {"cameraToWorld": cam.T.reshape(-1).tolist(), "fx": F, "fy": F, "cx": W / 2, "cy": H / 2}


WALL = {"origin": [0.0, 0.0, -3.0], "axisU": [1.0, 0.0, 0.0], "axisV": [0.0, 1.0, 0.0],
        "uMin": -3.0, "uMax": 3.0, "vMin": -1.3, "vMax": 1.0}


def check(name, cam, min_match):
    img = render(cam)
    result = rectify.rectify(img, frame_for(cam), WALL, FLOOR_Y)
    assert result is not None, f"{name}: wall not rectified"
    strip, u0, u1 = result
    us = u0 + (np.arange(strip.shape[1]) + 0.5) / rectify.PX_PER_M
    vs = FLOOR_Y + rectify.FLOOR_TO_TOP_M - (np.arange(strip.shape[0]) + 0.5) / rectify.PX_PER_M
    expected = pattern(*np.meshgrid(us, vs))
    valid = strip[..., 0] > 0
    match = (np.abs(strip[..., 0].astype(int) - expected.astype(int))[valid] < 30).mean()
    print(f"{name}: u {u0:.2f}..{u1:.2f} m, size {strip.shape[1]}x{strip.shape[0]}, pixels matching {match:.3f}")
    assert match >= min_match, f"{name}: only {match:.3f} of pixels match"


check("front", look_at(np.array([0.0, 0.0, 0.0]), [0, 0, -3]), 0.95)
check("oblique 45 deg", look_at(np.array([-2.0, 0.0, -1.0]), [0, -0.3, -3]), 0.9)
check("rolled phone", look_at(np.array([0.5, 0.2, -0.5]), [0.5, -0.4, -3]) @ np.array(
    [[math.cos(0.3), -math.sin(0.3), 0, 0], [math.sin(0.3), math.cos(0.3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), 0.9)
far = rectify.rectify(render(look_at(np.array([0.0, 0.0, 5.0]), [0, 0, -3])), frame_for(look_at(np.array([0.0, 0.0, 5.0]), [0, 0, -3])), WALL, FLOOR_Y)
assert far is None, "a wall 8 m away should be skipped"
side = look_at(np.array([0.0, 0.0, -1.0]), [5, 0, -1])
assert rectify.rectify(render(side), frame_for(side), WALL, FLOOR_Y) is None, "a wall seen edge-on should be skipped"
print("all rectify checks passed")
