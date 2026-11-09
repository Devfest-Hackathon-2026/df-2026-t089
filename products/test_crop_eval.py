"""②e カテゴリの人手チェックのサーバー側を一時フォルダのデータで確かめる。  python products/test_crop_eval.py"""
import json, tempfile
from pathlib import Path

import dashboard as d
from s5_classify import norm

with tempfile.TemporaryDirectory() as t:
    d.WORK = Path(t)
    names = {f"a_{i}": {"maker": "M", "product": "水"} for i in range(3)} | {f"b_{i}": {"maker": "M", "product": "茶"} for i in range(2)}
    (d.WORK / "names.json").write_text(json.dumps(names), encoding="utf-8")
    (d.WORK / "label_exclude.json").write_text('["a_2"]', encoding="utf-8")
    cats = {norm("M水"): "水カテゴリ", norm("M茶"): "茶カテゴリ"}
    d.category_of = lambda: cats

    v = d.crop_eval_view("水カテゴリ")
    assert v["cat"] == "水カテゴリ" and sorted(x[0] for x in v["next"]) == ["a_0", "a_1"], v  # 外した a_2 は出ない
    assert d.crop_eval_view("水カテゴリ", n=1)["next"] == v["next"][:1]  # 標本の順序は固定

    d.post("/api/crop_eval", {"category": "水カテゴリ", "ok": ["a_0"], "none": [], "other": {"a_1": "茶カテゴリ"}})
    e = json.loads((d.WORK / "crop_eval.json").read_text(encoding="utf-8"))
    assert e["a_0"]["true"] == "水カテゴリ" and e["a_1"]["true"] == "茶カテゴリ" and e["a_1"]["assigned"] == "水カテゴリ", e
    v = d.crop_eval_view("水カテゴリ")
    assert v["next"] == [], v
    assert sorted(map(list, v["judged"])) == [["a_0", "M 水", "水カテゴリ"], ["a_1", "M 水", "茶カテゴリ"]], v  # 完了後も見直せる
    assert next(c for c in v["cats"] if c["cat"] == "水カテゴリ") == {"cat": "水カテゴリ", "total": 2, "done": 2, "in": 1}, v
    assert d.crop_eval_view("")["cat"] == "茶カテゴリ"  # 無効なら未完了の先頭

    d.post("/api/crop_eval", {"category": "水カテゴリ", "ok": [], "none": ["a_1"], "other": {}})  # やり直しは上書き
    e = json.loads((d.WORK / "crop_eval.json").read_text(encoding="utf-8"))
    assert e["a_1"]["true"] == "", e

    d.post("/api/crop_eval", {"category": "茶カテゴリ", "hold": ["b_0"]})  # ✖ は再分類待ちにたまり、未チェックからは消える
    v = d.crop_eval_view("茶カテゴリ")
    assert [x[0] for x in v["next"]] == ["b_1"] and list(map(list, v["pending"])) == [["b_0", "M 茶", "茶カテゴリ"]], v
    d.post("/api/crop_eval", {"category": "茶カテゴリ", "undo": ["b_0"]})  # Ctrl+Z で未チェックに戻る
    v = d.crop_eval_view("茶カテゴリ")
    assert sorted(x[0] for x in v["next"]) == ["b_0", "b_1"] and v["pending"] == [], v

    cats[norm("M水")] = "茶カテゴリ"  # 商品のカテゴリが変わった → もう一度チェック対象
    d._ccache["key"] = None
    assert sorted(x[0] for x in d.crop_eval_view("茶カテゴリ")["next"]) == ["a_0", "a_1", "b_0", "b_1"]
    cats[norm("M水")] = "水カテゴリ"
    d._ccache["key"] = None

    for bad in ({"category": "水カテゴリ", "ok": ["b_0"], "none": [], "other": {}},  # 別カテゴリの切り抜き
                {"category": "水カテゴリ", "ok": ["a_2"], "none": [], "other": {}},  # 外した切り抜き
                {"category": "無い", "ok": [], "none": [], "other": {}},
                {"category": "水カテゴリ", "ok": [], "none": [], "other": {"a_0": "無い"}},
                {"category": "水カテゴリ", "ok": "a_0", "none": [], "other": {}}):
        try:
            d.post("/api/crop_eval", bad)
        except ValueError:
            continue
        raise AssertionError(bad)
print("ok")
