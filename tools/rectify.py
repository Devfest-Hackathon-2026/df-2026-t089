"""Turn frames recorded while walking into front-facing images of each wall (shelf face).

The app's training recorder saves, every 0.5 s, the camera image in sensor orientation, the camera intrinsics
and pose, and the walls found by the surface map. Each wall has a plane basis in metres: point = origin + u*U + v*V,
where V points straight up. This script samples every wall that faces the camera onto a grid of PX_PER_M pixels
per metre, so shelf uprights become vertical lines and the horizontal axis is metres along the wall.

  adb pull /sdcard/Android/data/jp.devfest.shelfsight/files/training tools/out/recordings
  python tools/rectify.py tools/out/recordings/<session>

Writes <session>/rectified/<frame>_<wall>.jpg and rectified.jsonl (frame, wall id, u range in metres).
tools/out is not committed: recordings can show customers.
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image

PX_PER_M = 100
FLOOR_TO_TOP_M = 1.9  # convenience-store shelves are at most about this tall
MIN_DISTANCE_M = 0.4
MAX_DISTANCE_M = 4.0
MAX_ANGLE_DEG = 65  # beyond this the wall is too oblique to read
MIN_VISIBLE_FRACTION = 0.6  # a column must be mostly inside the image to be kept
MIN_WIDTH_M = 0.4


def project(world, cam_to_world, fx, fy, cx, cy):
    """World points (..., 3) -> pixel x, y and depth in front of the camera."""
    m = np.asarray(cam_to_world, np.float64).reshape(4, 4).T  # stored column-major
    r, t = m[:3, :3], m[:3, 3]
    cam = (world - t) @ r  # = R^T (p - t)
    depth = -cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cx + fx * cam[..., 0] / depth
        y = cy - fy * cam[..., 1] / depth
    return x, y, depth


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx = np.clip(x - x0, 0, 1)[..., None]
    fy = np.clip(y - y0, 0, 1)[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy).astype(np.uint8)


def rectify(img, frame, wall, floor_y):
    """Returns (image, u0, u1) for the part of the wall seen in this frame, or None."""
    o, U, V = (np.asarray(wall[k], np.float64) for k in ("origin", "axisU", "axisV"))
    m = np.asarray(frame["cameraToWorld"], np.float64).reshape(4, 4).T
    cam_pos, forward = m[:3, 3], -m[:3, 2]
    normal = np.cross(U, V)
    normal /= np.linalg.norm(normal)
    distance = abs(np.dot(cam_pos - o, normal))
    if not MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M:
        return None
    if abs(np.dot(forward, normal)) < np.cos(np.radians(MAX_ANGLE_DEG)):
        return None

    v_bottom = (floor_y - o[1]) if floor_y is not None else wall["vMin"]
    v_top = v_bottom + FLOOR_TO_TOP_M
    us = np.arange(wall["uMin"], wall["uMax"], 1 / PX_PER_M)
    vs = np.arange(v_top, v_bottom, -1 / PX_PER_M)  # top row first
    if len(us) < 2 or len(vs) < 2:
        return None
    uu, vv = np.meshgrid(us, vs)
    world = o + uu[..., None] * U + vv[..., None] * V
    x, y, depth = project(world, frame["cameraToWorld"], frame["fx"], frame["fy"], frame["cx"], frame["cy"])
    h, w = img.shape[:2]
    inside = (depth > 0.1) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    keep_cols = inside.mean(axis=0) >= MIN_VISIBLE_FRACTION
    if keep_cols.sum() < MIN_WIDTH_M * PX_PER_M:
        return None
    cols = np.flatnonzero(keep_cols)
    c0, c1 = cols[0], cols[-1] + 1
    out = bilinear(img, x[:, c0:c1], y[:, c0:c1])
    out[~inside[:, c0:c1]] = 0
    return out, float(us[c0]), float(us[c1 - 1] + 1 / PX_PER_M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    a = ap.parse_args()
    out_dir = a.session / "rectified"
    out_dir.mkdir(exist_ok=True)
    index = []
    for line in (a.session / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        frame = json.loads(line)
        img = np.asarray(Image.open(a.session / frame["image"]).convert("RGB"))
        for wall in frame["walls"]:
            result = rectify(img, frame, wall, frame.get("floorY"))
            if result is None:
                continue
            strip, u0, u1 = result
            name = f'{Path(frame["image"]).stem}_{wall["id"]}.jpg'
            Image.fromarray(strip).save(out_dir / name, quality=90)
            index.append({"image": name, "frame": frame["image"], "wall": wall["id"], "u0": round(u0, 3), "u1": round(u1, 3),
                          "fromPoints": wall["fromPoints"]})
    (a.session / "rectified.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index), encoding="utf-8")
    walls = {r["wall"] for r in index}
    print(f"{len(index)} rectified strips from {len({r['frame'] for r in index})} frames, {len(walls)} walls -> {out_dir}")


if __name__ == "__main__":
    main()
