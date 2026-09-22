"""API 응답을 data/raw 에 JSON 으로 저장해 같은 요청을 반복하지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from ..config import ROOT

CACHE_DIR = ROOT / "data" / "raw"


def cached_json(namespace: str, key_parts: dict, fetch: Callable[[], object], refresh: bool = False):
    key = hashlib.sha1(json.dumps(key_parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    path = CACHE_DIR / namespace / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": key_parts, "data": data}, ensure_ascii=False), encoding="utf-8")
    return {"request": key_parts, "data": data}
