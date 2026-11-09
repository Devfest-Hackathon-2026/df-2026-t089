"""GPS から店を特定し（Google Places）、店ごとの 2D モデルを保存する。"""
import hashlib
import json
import sqlite3
import time

import httpx
from vercel.blob.errors import BlobNotFoundError
from vercel.blob.sync import get_blob, put_blob, query_blobs

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


def observations_of(model):
    return sum(f.get("obs") or 0.0 for f in model["fixtures"])


class BlobMapStore:
    """Vercel Blob（非公開）に店ごとの JSON を置く。Vercel の関数はファイルを残せないため。"""

    # shortcut: Blob の読み取りは書き込みから数秒遅れるので、直後の版番号と観測量の比較は古い値になり得る。同時に保存が増えたら DB に移す

    @staticmethod
    def _path(place_id):
        return "maps/" + hashlib.sha256(place_id.encode()).hexdigest() + ".json"

    def get(self, place_id):
        try:
            result = get_blob(self._path(place_id), access="private")
        except BlobNotFoundError:
            return None
        return json.loads(bytes(result.body))

    def all(self):
        # shortcut: 一覧のたびに全店の Blob を読む。店が数十を超えたら一覧用の小さな索引を別に保存する
        records = []
        for blob in query_blobs(prefix="maps/"):
            records.append(json.loads(bytes(get_blob(blob.pathname, access="private").body)))
        return records

    def save(self, place_id, name, model):
        """@return (保存したか, 今の版)。観測量が今の地図より少なければ保存しない"""
        observations = observations_of(model)
        current = self.get(place_id)
        if current is not None and observations < current["observations"]:
            return False, current["version"]
        version = 1 if current is None else current["version"] + 1
        record = {"place_id": place_id, "name": name, "model": model, "observations": observations, "version": version, "updated_at": time.time()}
        put_blob(self._path(place_id), json.dumps(record, ensure_ascii=False).encode(), access="private", allow_overwrite=True, content_type="application/json")
        return True, version


class MapStore:
    """店ごとに 1 つの 2D モデル（手元で動かすときの SQLite）。歩いた範囲が狭い地図で、よく観測された地図を上書きしない。"""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""create table if not exists maps (
            place_id text primary key, name text not null, model text not null,
            observations real not null, version integer not null, updated_at real not null)""")
        self.db.commit()

    def all(self):
        ids = [r[0] for r in self.db.execute("select place_id from maps")]
        return [self.get(i) for i in ids]

    def get(self, place_id):
        row = self.db.execute("select name, model, observations, version, updated_at from maps where place_id = ?", (place_id,)).fetchone()
        if row is None:
            return None
        return {"place_id": place_id, "name": row[0], "model": json.loads(row[1]), "observations": row[2], "version": row[3], "updated_at": row[4]}

    def save(self, place_id, name, model):
        """@return (保存したか, 今の版)。観測量が今の地図より少なければ保存しない"""
        observations = observations_of(model)
        with self.db:
            current = self.db.execute("select observations, version from maps where place_id = ?", (place_id,)).fetchone()
            if current is not None and observations < current[0]:
                return False, current[1]
            version = 1 if current is None else current[1] + 1
            self.db.execute("insert or replace into maps values (?, ?, ?, ?, ?, ?)",
                            (place_id, name, json.dumps(model, ensure_ascii=False), observations, version, time.time()))
        return True, version
