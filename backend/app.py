"""店のモデルのバックエンド。

  pip install -r backend/requirements.txt
  BACKEND_API_KEY=... GCP_API_KEY=... uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000

POST /v1/solve        歩いた経路と商品の観測から、OR-Tools で棚の配置を解いて返す
POST /v1/maps         観測を OR-Tools で解き、GPS から求めた店の地図として保存（更新）する
POST /v1/maps/lookup  GPS から店を求め、保存済みの 2D モデルを返す
GET  /v1/stores       登録された店と地図の一覧（Web の画面用。鍵なしで読める）
GET  /                登録された店を 3D で見る画面

GPS と経路は店を求める・解くのに使うだけで保存しない。保存するのは店と 2D モデルだけ。
"""
import os
import secrets
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from solver import solve
from stores import BlobMapStore, MapStore, PlacesResolver

Coord = Annotated[float, Field(ge=-1000.0, le=1000.0)]
Point = tuple[Coord, Coord]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
Text = Annotated[str, Field(max_length=100)]


class Strict(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class Observation(Strict):
    cam: Point
    point: Point
    category: Text = ""
    genre: Text = ""


class SolveRequest(Strict):
    path: list[Point] = Field(max_length=20_000)
    observations: list[Observation] = Field(max_length=50_000)
    counters: list[Point] = Field(default=[], max_length=20_000)


class Location(Strict):
    lat: Latitude
    lng: Longitude


class UploadRequest(SolveRequest, Location):
    pass


def create_app(store, resolver, api_key):
    """店の解決と保存先を差し替えられるように、アプリは関数で作る。"""
    app = FastAPI(title="ShelfSight store maps")

    def authorized(x_api_key: str = Header(default="")):
        if not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "bad api key")

    def resolve_store(loc):
        try:
            place = resolver.nearest(loc.lat, loc.lng)
        except httpx.HTTPError as e:
            raise HTTPException(503, "store lookup is unavailable") from e
        if place is None:
            raise HTTPException(404, "no convenience store near this location")
        return place

    @app.post("/v1/solve", dependencies=[Depends(authorized)])
    def solve_store(req: SolveRequest):
        return solve(req.path, [o.model_dump() for o in req.observations], req.counters)

    @app.post("/v1/maps", dependencies=[Depends(authorized)])
    def upload_map(req: UploadRequest):
        place = resolve_store(req)   # 店が分からないなら解かない
        model = solve(req.path, [o.model_dump() for o in req.observations], req.counters)["model"]
        if not model["fixtures"]:
            return {**place, "saved": False, "version": 0}
        saved, version = store.save(place["place_id"], place["name"], model)
        return {**place, "saved": saved, "version": version}

    @app.post("/v1/maps/lookup", dependencies=[Depends(authorized)])
    def lookup_map(req: Location):
        place = resolve_store(req)
        found = store.get(place["place_id"])
        return found if found else {**place, "model": None}

    @app.get("/v1/stores")
    def list_stores(response: Response):
        response.headers["Cache-Control"] = "public, s-maxage=30, stale-while-revalidate=300"
        return sorted(store.all(), key=lambda r: -r["updated_at"])

    @app.get("/", include_in_schema=False)
    def web():
        return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")

    return app


def from_env():
    api_key = os.environ["BACKEND_API_KEY"]   # 無ければ起動させない
    db = os.environ.get("BACKEND_DB", str(Path(__file__).resolve().parent / "maps.db"))
    store = BlobMapStore() if os.environ.get("VERCEL") else MapStore(db)
    return create_app(store, PlacesResolver(os.environ["GCP_API_KEY"]), api_key)


if os.environ.get("BACKEND_API_KEY"):
    app = from_env()
