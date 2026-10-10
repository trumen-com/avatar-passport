"""Nearby body / paint shops from OpenStreetMap (Overpass API), ranked body/paint matches first.

Source and rules: docs/research/phase0_findings_deploy_market.md section 8.
- OSM data is ODbL 1.0; every response carries the attribution and the copyright URL.
- Tashkent OSM has almost no ``service:vehicle:body_repair`` / ``painting`` tags, so shop names are also matched
  (кузов|kuzov|покрас|kraska|маляр|рихтов|polir|body|paint, case-insensitive).
- Main server overpass-api.de, fallback the VK mirror; unique User-Agent; 25 s timeout per server.
- Results are cached on disk for 24 h per rounded position and radius (fewer requests to a donated service).
- On network failure the caller gets ``ShopsUnavailable`` (the API answers 503). Shops are never invented.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "AvtoPassport/0.1 (hackathon demo)"
TIMEOUT_S = 25.0
CACHE_TTL_S = 24 * 3600
MAX_SHOPS = 100
SOURCE = "OpenStreetMap contributors (ODbL)"
ATTRIBUTION_URL = "https://www.openstreetmap.org/copyright"
NAME_KEYWORDS = re.compile(r"кузов|kuzov|покрас|kraska|маляр|рихтов|polir|body|paint", re.IGNORECASE)
NAME_TAGS = ("name", "name:ru", "name:uz", "name:en", "brand", "operator")

Match = Literal["body_repair_tag", "name_keyword", "car_repair"]
_MATCH_RANK: dict[str, int] = {"body_repair_tag": 0, "name_keyword": 1, "car_repair": 2}

# fetch(url, query, timeout_s) -> parsed Overpass JSON. Injected in tests; the default uses urllib (stdlib).
Fetcher = Callable[[str, str, float], dict[str, Any]]


class ShopsUnavailable(RuntimeError):
    """Every Overpass endpoint failed; the message lists each failure."""


def _urllib_fetch(url: str, query: str, timeout_s: float) -> dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    })
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed https endpoints
        doc = json.loads(resp.read().decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Overpass returned non-object JSON")
    return doc


def build_query(lat: float, lon: float, radius_m: int, timeout_s: float = TIMEOUT_S) -> str:
    around = f"(around:{int(radius_m)},{lat:.6f},{lon:.6f})"
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        f"(\n  nwr[\"shop\"=\"car_repair\"]{around};\n  nwr[\"craft\"=\"car_painter\"]{around};\n);\n"
        "out tags center;"
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_008.8  # mean Earth radius, metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _shop_name(tags: dict[str, str]) -> str | None:
    for key in NAME_TAGS:
        value = tags.get(key)
        if value:
            return value
    return None


def classify(tags: dict[str, str]) -> Match:
    if (tags.get("service:vehicle:body_repair") == "yes" or tags.get("service:vehicle:painting") == "yes"
            or tags.get("craft") == "car_painter"):
        return "body_repair_tag"
    if any(NAME_KEYWORDS.search(tags.get(key, "")) for key in NAME_TAGS):
        return "name_keyword"
    return "car_repair"


def parse_elements(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Overpass JSON -> [{osm_type, osm_id, name, lat, lon, tags}] (elements without a position are skipped)."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for el in doc.get("elements", []) or []:
        if not isinstance(el, dict):
            continue
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        pos = el if "lat" in el and "lon" in el else (el.get("center") or {})
        try:
            lat, lon = float(pos["lat"]), float(pos["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(el.get("type", "")), int(el.get("id", 0) or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append({"osm_type": key[0], "osm_id": key[1], "name": _shop_name(tags), "lat": lat, "lon": lon,
                    "tags": tags})
    return out


def rank(shops: Sequence[dict[str, Any]], lat: float, lon: float, radius_m: int) -> list[dict[str, Any]]:
    """Add distance_m + match; keep shops within radius; body/paint matches first, then by distance."""
    ranked: list[dict[str, Any]] = []
    for s in shops:
        d = haversine_m(lat, lon, s["lat"], s["lon"])
        if d > radius_m * 1.05:  # small slack: Overpass "around" measures to the way geometry, we to its centre
            continue
        ranked.append({**s, "distance_m": round(d, 1), "match": classify(s["tags"])})
    ranked.sort(key=lambda s: (_MATCH_RANK[s["match"]], s["distance_m"]))
    return ranked[:MAX_SHOPS]


class ShopFinder:
    """Overpass lookup with a 24 h on-disk cache in ``cache_dir``."""

    def __init__(self, cache_dir: Path, *, fetch: Fetcher | None = None,
                 endpoints: Sequence[str] = OVERPASS_ENDPOINTS, timeout_s: float = TIMEOUT_S,
                 ttl_s: float = CACHE_TTL_S, clock: Callable[[], float] = time.time) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetch: Fetcher = fetch or _urllib_fetch
        self.endpoints = tuple(endpoints)
        self.timeout_s = timeout_s
        self.ttl_s = ttl_s
        self.clock = clock

    def cache_path(self, lat: float, lon: float, radius_m: int) -> Path:
        # 3 decimals ~ 110 m; the radius is part of the key so a larger search is never served from a smaller one
        return self.cache_dir / f"shops_{lat:.3f}_{lon:.3f}_{int(radius_m)}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict) or self.clock() - float(doc.get("fetched_ts", 0)) > self.ttl_s:
            return None
        return doc

    def _fetch_any(self, query: str) -> tuple[dict[str, Any], str]:
        errors: list[str] = []
        for url in self.endpoints:
            try:
                doc = self.fetch(url, query, self.timeout_s)
                remark = str(doc.get("remark") or "")
                if "error" in remark.lower():  # Overpass reports query timeouts / overload as a 200 with a remark
                    raise ValueError(f"Overpass remark: {remark[:200]}")
                return doc, url
            except Exception as exc:  # network boundary: HTTP, timeout, DNS, bad JSON - try the next endpoint
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        raise ShopsUnavailable("OpenStreetMap Overpass API unreachable - " + "; ".join(errors))

    def find(self, lat: float, lon: float, radius_m: int = 5000) -> dict[str, Any]:
        path = self.cache_path(lat, lon, radius_m)
        cached = self._read_cache(path)
        if cached is not None:
            raw, fetched_utc, endpoint, from_cache = cached["shops"], cached["fetched_utc"], cached["endpoint"], True
        else:
            # query around the rounded centre, so the cached answer is valid for every request mapped to this file
            rlat, rlon = round(lat, 3), round(lon, 3)
            doc, endpoint = self._fetch_any(build_query(rlat, rlon, radius_m + 150, self.timeout_s))
            raw = parse_elements(doc)
            now = self.clock()
            fetched_utc = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            from_cache = False
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"fetched_ts": now, "fetched_utc": fetched_utc, "endpoint": endpoint,
                                           "shops": raw}, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
            except OSError:
                pass  # a cache write failure must not fail the request
        return {
            "source": SOURCE,
            "attribution_url": ATTRIBUTION_URL,
            "fetched_utc": fetched_utc,
            "endpoint": endpoint,
            "cached": from_cache,
            "query": {"lat": lat, "lon": lon, "radius_m": int(radius_m)},
            "shops": rank(raw, lat, lon, radius_m),
        }
