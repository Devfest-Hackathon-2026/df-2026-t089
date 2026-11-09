"""記録の再生を、作ったデータで確かめる（検出器は差し替え）。  python products/test_replay.py"""
import json, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import replay as r

with tempfile.TemporaryDirectory() as t:
    r.RECORDINGS = Path(t)
    d = Path(t) / "20261109_120000"
    for sub in ("frames", "depth", "points"):
        (d / sub).mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    lines = []
    for n in range(12):   # 歩きながら右（+x）を向いて棚を見る。カメラは +x を向く: カメラの -Z が世界の +x
        name = f"{n:05d}"
        Image.new("RGB", (72, 128), (90, 90, 90)).save(d / "frames" / f"{name}.jpg")
        np.full((90, 160), 1500, dtype="<u2").tofile(d / "depth" / f"{name}.d16")
        z = n * 0.3
        m = [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0.0, 1.3, z, 1]   # 列優先: 軸 X=+z, Y=+y, Z=-x
        state = "TRACKING" if n else "PAUSED"
        lines.append(json.dumps({"n": n, "t": int(n * 1e9), "image": f"frames/{name}.jpg", "state": state, "depth": [160, 90],
                                 "geom": {"w": 1280, "h": 720, "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0, "rot": 0, "m": m, "u0": 0.0, "du": 1 / 1280, "v0": 0.0, "dv": 1 / 720}}))
    (d / "keyframes.jsonl").write_text("\n".join(lines) + "\n{broken")   # 途中で切れた最後の行は無視する
    (d / "events.jsonl").write_text(json.dumps({"type": "tap", "t": int(3e9), "wall_ms": 0}) + "\n")
    calls = []

    def fake(a):
        calls.append(a.shape)
        boxes = [[0.1 + i * 0.1, 0.2 + j * 0.2, 0.18 + i * 0.1, 0.36 + j * 0.2] for i in range(6) for j in range(3)]
        return [{"box": b, "score": 0.9, "name": "緑茶" if k % 2 == 0 else None, "genre": "ソフトドリンク" if k % 2 == 0 else None, "fine": "緑茶" if k % 2 == 0 else None, "sim": 0.85} for k, b in enumerate(boxes)]

    res = r.replay("20261109_120000", "app", 1.0, fake)
    assert len(calls) == 12, calls                      # 追跡が止まっても画像認識は試せる
    s = res["summary"]
    assert s["steps"] == 12 and s["tracked_ratio"] == round(11 / 12, 3) and s["boxes_per_step"] == 18 and s["depth_ratio"] == round(11 / 12, 3), s
    assert s["items"] >= 1 and s["named_per_step"] == 9, s
    assert len(res["steps"][0]["dets"]) == 18 and "tap" not in res["steps"][0]
    last = res["steps"][-1]
    assert abs(last["cam"][2] - 3.3) < 1e-6 and last["fwd"] == [1.0, 0.0], last   # 向きが取れている
    # 深度 1.5 m 先の商品は、カメラの +x 側に置かれる
    g = json.loads(lines[1])["geom"]
    p = r.world_at(0.5, 0.5, 1.5, g)
    assert abs(p[0] - 1.5) < 1e-6 and abs(p[1] - 1.3) < 1e-6 and abs(p[2] - 0.3) < 1e-6, p
    assert last["tap"]["kind"] in ("sentence", "around", "unknown") and last["context"].startswith("- 方向と距離"), last["tap"]
    assert res["events"][0]["at"] == 3.0
    assert (d / "replay_app.json").exists() and r.recordings()[0]["replays"] == ["app"]

# 壁に寄せる / 深度がなければ壁と交わらせる
wall = {"c": (2.0, 1.0, 0.0), "n": (1.0, 0.0, 0.0), "r": 3.0}
assert r.place((0, 1, 0), (1, 1, 0), (1.9, 1.0, 0.0), [wall])[0] == 2.0
assert abs(r.place((0, 1, 0), (1, 1, 0), None, [wall])[0] - 2.0) < 1e-6
assert r.place((0, 1, 0), (1, 1, 0), None, []) is None
assert r.focus([]) == ("NOTHING", None) and r.focus([[0.1, 0.1, 0.9, 0.9]])[0] == "PRODUCT"
assert "カップ麺はどこ" in r.prompt_for("カップ麺はどこ", "- 地図") and "- 地図" in r.prompt_for("x", "- 地図") and "$" not in r.prompt_for("x", "y")
print("ok")
