"""Vercel の入口。本体は backend/app.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import from_env  # noqa: E402

app = from_env()
