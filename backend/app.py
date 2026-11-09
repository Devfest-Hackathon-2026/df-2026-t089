"""店のモデルのバックエンド。

  pip install -r backend/requirements.txt
  BACKEND_API_KEY=... GCP_API_KEY=... uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000

POST /v1/solve        歩いた経路と商品の観測から、OR-Tools で棚の配置を解いて返す
POST /v1/maps         2D モデルを、GPS から求めた店に保存（更新）する
POST /v1/maps/lookup  GPS から店を求め、保存済みの 2D モデルを返す

GPS と経路は店を求める・解くのに使うだけで保存しない。保存するのは店と 2D モデルだけ。
"""
import os
import secrets
from pathlib import Path
from typing import Annotated, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
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


class Fixture(Strict):
    kind: Literal["island", "wall", "case", "glass", "end", "register"]
    axis: Literal[0, 1]
    c: Coord
    a: Coord
    b: Coord
    d: Annotated[float, Field(gt=0, le=10)]
    h: Annotated[float, Field(gt=0, le=10)]
    n: Annotated[int, Field(ge=1, le=100)]
    w: Annotated[float, Field(gt=0, le=100)]
    widths: list[Annotated[float, Field(gt=0, le=100)]] = Field(default=[], max_length=100)
    faces: dict[Literal["+", "-"], list[list[tuple[Text, float]]]] = {}
    themes: dict[Literal["+", "-"], list[Optional[Text]]] = {}
    obs: Optional[Annotated[float, Field(ge=0)]] = None


class StoreMap(Strict):
    yaw: float
    room: Optional[tuple[Coord, Coord, Coord, Coord]]
    fixtures: list[Fixture] = Field(max_length=500)


class Location(Strict):
    lat: Latitude
    lng: Longitude


class SaveRequest(Location):
    model: StoreMap


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
    def save_map(req: SaveRequest):
        place = resolve_store(req)
        saved, version = store.save(place["place_id"], place["name"], req.model.model_dump())
        return {**place, "saved": saved, "version": version}

    @app.post("/v1/maps/lookup", dependencies=[Depends(authorized)])
    def lookup_map(req: Location):
        place = resolve_store(req)
        found = store.get(place["place_id"])
        return found if found else {**place, "model": None}

    return app


def from_env():
    api_key = os.environ["BACKEND_API_KEY"]   # 無ければ起動させない
    db = os.environ.get("BACKEND_DB", str(Path(__file__).resolve().parent / "maps.db"))
    store = BlobMapStore() if os.environ.get("VERCEL") else MapStore(db)
    return create_app(store, PlacesResolver(os.environ["GCP_API_KEY"]), api_key)


if os.environ.get("BACKEND_API_KEY"):
    app = from_env()
