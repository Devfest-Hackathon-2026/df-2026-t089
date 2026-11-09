"""python -m pytest backend/test_backend.py"""
from fastapi.testclient import TestClient

from app import create_app
from solver import solve
from stores import MapStore

KEY = "secret"
HEADERS = {"X-Api-Key": KEY}


class FakeResolver:
    def nearest(self, lat, lng):
        return {"place_id": "store-1", "name": "テスト店"} if lat > 0 else None


def client(tmp_path):
    return TestClient(create_app(MapStore(str(tmp_path / "maps.db")), FakeResolver(), KEY))


def walk():
    """通路を歩きながら、両側の壁の棚を見る"""
    path = [(0.0, i * 0.3) for i in range(14)]
    observations = []
    for i in range(160):
        z = i * 4.0 / 160
        observations.append({"cam": (0.0, z), "point": (1.5, z), "category": "お茶", "genre": "ソフトドリンク"})
        observations.append({"cam": (0.0, z), "point": (-1.5, z), "category": "電池", "genre": "日用品・衛生用品"})
    return path, observations


def rects(fixtures):
    return [(f["c"] - f["d"] / 2, f["c"] + f["d"] / 2, f["a"], f["b"]) if f["axis"] == 0 else (f["a"], f["b"], f["c"] - f["d"] / 2, f["c"] + f["d"] / 2) for f in fixtures]


def test_solver_finds_both_shelves_without_overlap():
    path, observations = walk()
    result = solve(path, observations)
    assert result["solver"] and result["solver"][0] == "OPTIMAL"
    fixtures = result["model"]["fixtures"]
    assert len(fixtures) == 2, result["text"]
    (a, b) = rects(fixtures)
    assert min(a[1], b[1]) - max(a[0], b[0]) <= 0.01 or min(a[3], b[3]) - max(a[2], b[2]) <= 0.01


def test_api_key_is_required(tmp_path):
    c = client(tmp_path)
    assert c.post("/v1/maps/lookup", json={"lat": 35.0, "lng": 139.0}).status_code == 401
    assert c.post("/v1/maps/lookup", json={"lat": 35.0, "lng": 139.0}, headers={"X-Api-Key": "no"}).status_code == 401


def test_bad_input_is_rejected(tmp_path):
    c = client(tmp_path)
    assert c.post("/v1/maps/lookup", json={"lat": 91.0, "lng": 139.0}, headers=HEADERS).status_code == 422
    assert c.post("/v1/solve", json={"path": [[1e9, 0.0]], "observations": []}, headers=HEADERS).status_code == 422


def test_solved_map_is_saved_and_loaded_on_the_next_visit(tmp_path):
    c = client(tmp_path)
    path, observations = walk()
    body = {"path": path, "observations": observations}
    solved = c.post("/v1/solve", json=body, headers=HEADERS)
    assert solved.status_code == 200, solved.text
    model = solved.json()["model"]
    assert c.post("/v1/maps/lookup", json={"lat": 35.0, "lng": 139.0}, headers=HEADERS).json()["model"] is None
    saved = c.post("/v1/maps", json={"lat": 35.0, "lng": 139.0, "model": model}, headers=HEADERS).json()
    assert saved == {"place_id": "store-1", "name": "テスト店", "saved": True, "version": 1}
    loaded = c.post("/v1/maps/lookup", json={"lat": 35.0, "lng": 139.0}, headers=HEADERS).json()
    assert loaded["name"] == "テスト店" and len(loaded["model"]["fixtures"]) == 2


def test_a_smaller_walk_does_not_replace_a_better_map(tmp_path):
    c = client(tmp_path)
    path, observations = walk()
    model = c.post("/v1/solve", json={"path": path, "observations": observations}, headers=HEADERS).json()["model"]
    c.post("/v1/maps", json={"lat": 35.0, "lng": 139.0, "model": model}, headers=HEADERS)
    smaller = {**model, "fixtures": [{**model["fixtures"][0], "obs": 1.0}]}
    assert c.post("/v1/maps", json={"lat": 35.0, "lng": 139.0, "model": smaller}, headers=HEADERS).json()["saved"] is False
    assert len(c.post("/v1/maps/lookup", json={"lat": 35.0, "lng": 139.0}, headers=HEADERS).json()["model"]["fixtures"]) == 2


def test_no_store_nearby_is_a_404(tmp_path):
    assert client(tmp_path).post("/v1/maps/lookup", json={"lat": -1.0, "lng": 139.0}, headers=HEADERS).status_code == 404
