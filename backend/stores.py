"""GPS から店を特定し（Google Places）、店ごとの 2D モデルを SQLite に保存する。"""
import json
import sqlite3
import time

import httpx

PLACES_URL = "https://places.googleapis.com/v1/places:searchNearby"
SEARCH_RADIUS_M = 80.0   # 店内の GPS は数十 m ずれる


class PlacesResolver:
    """緯度経度から、いちばん近いコンビニの (place_id, 店名) を返す。"""

    def __init__(self, api_key):
        self.api_key = api_key

    def nearest(self, lat, lng):
        r = httpx.post(PLACES_URL, timeout=10.0, headers={
            "X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": "places.id,places.displayName",
        }, json={
            "includedTypes": ["convenience_store"], "maxResultCount": 1, "rankPreference": "DISTANCE", "languageCode": "ja",
            "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": SEARCH_RADIUS_M}},
        })
        r.raise_for_status()
        places = r.json().get("places", [])
        if not places:
            return None
        return {"place_id": places[0]["id"], "name": places[0]["displayName"]["text"]}


class MapStore:
    """店ごとに 1 つの 2D モデル。歩いた範囲が狭い地図で、よく観測された地図を上書きしない。"""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""create table if not exists maps (
            place_id text primary key, name text not null, model text not null,
            observations real not null, version integer not null, updated_at real not null)""")
        self.db.commit()

    def get(self, place_id):
        row = self.db.execute("select name, model, observations, version, updated_at from maps where place_id = ?", (place_id,)).fetchone()
        if row is None:
            return None
        return {"place_id": place_id, "name": row[0], "model": json.loads(row[1]), "observations": row[2], "version": row[3], "updated_at": row[4]}

    def save(self, place_id, name, model):
        """@return (保存したか, 今の版)。観測量が今の地図より少なければ保存しない"""
        observations = sum(f.get("obs") or 0.0 for f in model["fixtures"])
        with self.db:
            current = self.db.execute("select observations, version from maps where place_id = ?", (place_id,)).fetchone()
            if current is not None and observations < current[0]:
                return False, current[1]
            version = 1 if current is None else current[1] + 1
            self.db.execute("insert or replace into maps values (?, ?, ?, ?, ?, ?)",
                            (place_id, name, json.dumps(model, ensure_ascii=False), observations, version, time.time()))
        return True, version
