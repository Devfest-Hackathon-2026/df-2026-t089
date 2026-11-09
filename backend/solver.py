"""観測（歩いた経路と商品の観測）から、OR-Tools CP-SAT で店の棚の配置を解く。

棚の候補づくり（面の抽出・角・視線で削る処理）はアプリと同じ products/store_model.py を使い、
候補の「選び方」だけを、貪欲法から全候補を同時に見る最適化に置き換える。
"""
import sys
from pathlib import Path

from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "products"))
from store_model import MODULE_COST, SHELF_COST, StoreModel, text  # noqa: E402

TIME_LIMIT_S = 3.0   # 棚とガラスで最大 3 回解くので、アプリの待ち時間（12 秒）に収める


class CpSatStoreModel(StoreModel):
    """棚の採否を CP-SAT で決める StoreModel。"""

    def __init__(self):
        super().__init__()
        self.statuses = []

    def _variants(self, candidates, accepted):
        """候補と、当たる相手の分だけ長手方向に削った版。同じ元の候補からは 1 つしか選ばない"""
        groups = []
        for c in candidates:
            group = [c]
            for other in candidates + accepted:
                if other is c or not self._overlaps(c["fixture"], other["fixture"]):
                    continue
                trimmed = dict(c)
                if self._trim(trimmed, other["fixture"]) and not self._overlaps(trimmed["fixture"], other["fixture"]):
                    group.append(trimmed)
            groups.append(group)
        return groups

    def _touching_islands(self, end, options):
        rect = self._rect(end["fixture"])
        out = []
        for o in options:
            if o["fixture"]["kind"] == "island":
                q = self._rect(o["fixture"])
                if rect[0] <= q[2] + 0.15 and rect[2] >= q[0] - 0.15 and rect[1] <= q[3] + 0.15 and rect[3] >= q[1] - 0.15:
                    out.append(o)
        return out

    def _choose(self, candidates, accepted):
        """重ならない棚の組のうち、説明できる点の数 − 棚のコストが最大のものを選ぶ"""
        if not candidates:
            return accepted
        explained = set().union(*(c["explains"] for c in accepted))
        groups = self._variants(candidates, accepted)
        options = [o for g in groups for o in g]
        model = cp_model.CpModel()
        chosen = {id(o): model.NewBoolVar(f"rack{i}") for i, o in enumerate(options)}
        for group in groups:
            model.AddAtMostOne(chosen[id(o)] for o in group)
        for i, a in enumerate(options):
            if any(self._overlaps(a["fixture"], f["fixture"]) for f in accepted):
                model.Add(chosen[id(a)] == 0)
            for b in options[i + 1:]:
                if self._overlaps(a["fixture"], b["fixture"]):
                    model.AddBoolOr(chosen[id(a)].Not(), chosen[id(b)].Not())
            if a["fixture"]["kind"] == "end":   # エンドは島 1 つの端にだけ付く
                fixed = len(self._touching_islands(a, accepted))
                model.Add(sum(chosen[id(o)] for o in self._touching_islands(a, options)) + fixed == 1).OnlyEnforceIf(chosen[id(a)])
        covers = {}
        for o in options:
            for p in o["explains"] - explained:
                covers.setdefault(p, []).append(chosen[id(o)])
        covered = []
        for i, racks in enumerate(covers.values()):
            y = model.NewBoolVar(f"point{i}")
            model.AddBoolOr(racks).OnlyEnforceIf(y)
            covered.append(y)
        model.Maximize(sum(covered) + sum(
            (int(o["bonus"]) - SHELF_COST - MODULE_COST * o["fixture"]["n"]) * chosen[id(o)] for o in options))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = TIME_LIMIT_S
        solver.parameters.random_seed = 1
        solver.parameters.num_workers = 4
        status = solver.Solve(model)
        self.statuses.append(solver.StatusName(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return super()._choose(candidates, accepted)   # 時間内に解がなければ今までの貪欲法で返す
        return accepted + [o for o in options if solver.Value(chosen[id(o)])]


def fill(model, path, observations, counters):
    for x, z in path:
        model.add_camera(x, z)
    for o in observations:
        model.add_product(o["cam"][0], o["cam"][1], o["point"][0], o["point"][1], o.get("category", ""), o.get("genre", ""))
    for x, z in counters:
        model.add_counter(x, z)
    return model


def solve(path, observations, counters=()):
    """@return 推定した店（yaw, room, fixtures）と、記述言語の文字列・解の状態"""
    model = fill(CpSatStoreModel(), path, observations, counters)
    view = model.estimate()
    return {"model": view, "text": text(view), "solver": model.statuses}
