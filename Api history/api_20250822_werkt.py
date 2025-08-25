"""
PlantWijs API v0.9.6 — + ingebouwde kaart-UI (Leaflet)  | compound matching | deterministische selftests | schema/print/SSL fallbacks
Start server:  uvicorn api:app --reload --port 9000
Open de kaart: http://127.0.0.1:9000/   (Docs: /docs)
Offline test: python api.py --selftest | Voorbeeld: python api.py --example 51.95 5.99 zon 7a 10

Belangrijk:
- Als jouw Python-installatie geen `ssl` heeft, draait de app in offline testmodus en kun je de server niet starten.
  Op Windows zou `ssl` standaard aanwezig moeten zijn. Zie eerdere instructies als dit niet zo is.
- Nieuw in 0.9.6: een **inline kaartpagina** (Leaflet) op `/` die jouw klik/locatie gebruikt en `/advies/geo` aanroept.
"""
from __future__ import annotations

# ───────────────────── standaard imports (zonder frameworks die SSL vereisen)
import os, io, time, json, re, urllib.parse, sys, math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pyproj import Transformer

try:
    import orjson as _orjson
    def _dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)
except Exception:  # pragma: no cover
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

# Compatibele stderr-print (geen afhankelijkheid van print(..., file=...))
def eprint(*args: Any) -> None:
    try:
        sys.stderr.write(" ".join(str(x) for x in args) + "\n")
    except Exception:
        # minimal fallback
        sys.stderr.write("[ERR] eprint failed\n")

# Detecteer SSL-beschikbaarheid VÓÓR we fastapi/requests importeren
try:
    import ssl  # noqa: F401
    HAS_SSL = True
except ModuleNotFoundError:
    HAS_SSL = False

OFFLINE = os.getenv("PLANTWIJS_OFFLINE", "0") in {"1", "true", "True"}

# ───────────────────── configuratie (milieuvariabelen)
CSV_URL = os.getenv(
    "PLANTWIJS_CSV_URL",
    (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vTJVlLk-gIu6T89zwDA5AAH77eVR21OtzRgoEi_2vllJLx6M9sAe2DsoVk-UcdZqqp7AL3re0qpQ_rH"
        "/pub?gid=0&single=true&output=csv"
    ),
)

# PDOK FGR (Fysisch Geografische Regio's)
PDOK_FGR_WFS = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0"
    "?service=WFS&version=2.0.0"
)
FGR_TYPENAME = "fysischgeografischeregios:fysischgeografischeregios"
FMT_JSON = "application/json;subtype=geojson"
HEADERS = {"User-Agent": "plantwijs/0.9.6 (+github.com/plantwijs)"}

# RD-transformatie  WGS84 ➜ RD New
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)

# ───────────────────── util: kleine TTL-cache (zonder extra dependency)
class TTLCache:
    def __init__(self, ttl_s: int = 600, max_items: int = 512) -> None:
        self.ttl = ttl_s
        self.max = max_items
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        item = self._store.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        if len(self._store) >= self.max:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest_key, None)
        self._store[key] = (time.time(), val)

# globale caches
http_cache = TTLCache(ttl_s=300, max_items=256)
advies_cache = TTLCache(ttl_s=900, max_items=256)

# ───────────────────── CSV laden (met fallback/embedded sample)
_df: Optional[pd.DataFrame] = None
_df_etag: Optional[str] = None

REQUIRED_COLS = {
    "naam", "bodem", "droogtetolerantie", "klimaatzone",
    "zon_min", "zon_max", "ecowaarde"
}

# Kleine ingebedde dataset voor offline tests (3 soorten)
# Let op: de laatste kolom 'opmerking' kan een komma bevatten, daarom is die waarde gequote.
_EMBEDDED_CSV = (
    "naam,bodem,droogtetolerantie,klimaatzone,zon_min,zon_max,ecowaarde,opmerking\n"
    "Zomereik,zand;leem,middel,7a,2,2,9,Inheems eik\n"
    "Zwarte els,klei;veen,hoog,7a,1,2,8,Nat tolerantie\n"
    "Meidoorn (eenstijlige),klei;zand;leem,laag,7a,1,2,7,\"Struweel, biodivers\"\n"
)

ALT_NAME_COLUMNS = [
    "naam", "soort", "soortnaam", "plant", "plantnaam", "boomnaam",
    "naam_nl", "nl_naam", "species", "name", "title"
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    for col in ["zon_min", "zon_max", "ecowaarde"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Zet naar string en vul leeg
    for col in ["bodem", "droogtetolerantie", "klimaatzone"] + [c for c in ALT_NAME_COLUMNS if c in df.columns]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")
    return df


def _ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Garandeer minimaal vereiste kolommen; map alternatieve naamkolommen → 'naam'."""
    df = df.copy()
    # 1) 'naam'
    if "naam" not in df.columns:
        for c in ALT_NAME_COLUMNS:
            if c in df.columns:
                df["naam"] = df[c].astype(str)
                break
    if "naam" not in df.columns:
        # synthesize naam
        df["naam"] = [f"Onbekend {i+1}" for i in range(len(df))]
    # 2) overige kolommen met veilige defaults
    if "bodem" not in df.columns:
        df["bodem"] = ""
    if "droogtetolerantie" not in df.columns:
        df["droogtetolerantie"] = ""
    if "klimaatzone" not in df.columns:
        df["klimaatzone"] = "7a"
    if "zon_min" not in df.columns:
        df["zon_min"] = 0
    if "zon_max" not in df.columns:
        df["zon_max"] = 2
    if "ecowaarde" not in df.columns:
        df["ecowaarde"] = 0
    return df


def _load_embedded() -> pd.DataFrame:
    # Defensieve parser: respecteer quotes en spaties na komma
    df = pd.read_csv(io.StringIO(_EMBEDDED_CSV), sep=",", quotechar='"', skipinitialspace=True)
    df = _normalize_columns(df)
    df = _ensure_min_schema(df)
    # Valideer kolomtelling en aanwezigheid van opmerking-kolom
    assert "opmerking" in df.columns, "'opmerking' kolom ontbreekt in embedded CSV"
    assert df.shape[1] >= 8, "Embedded CSV heeft te weinig kolommen"
    return df


def load_csv(force: bool = False) -> pd.DataFrame:
    global _df, _df_etag
    if _df is not None and not force:
        return _df
    if OFFLINE or not HAS_SSL:
        _df = _load_embedded()
        _df_etag = "embedded"
        return _df
    # Lazy import pas bij online gebruik om SSL-issues te vermijden
    try:
        import requests  # type: ignore
        r = requests.get(CSV_URL, headers=HEADERS, timeout=10)
        r.raise_for_status()
        _df_etag = r.headers.get("ETag")
        df = pd.read_csv(io.BytesIO(r.content), sep=",", on_bad_lines="skip")
        df = _normalize_columns(df)
        df = _ensure_min_schema(df)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            eprint("[WARN] CSV mist kolommen:", missing)
        _df = df
        eprint("[OK] CSV geladen:", len(df), "rijen")
        return df
    except Exception as e:
        eprint("[ERR] CSV laden faalde, val terug op embedded:", e)
        _df = _load_embedded()
        _df_etag = "embedded"
        return _df

# ───────────────────── PDOK helpers (offline tolerant)

def _http_get(url: str, timeout: int = 8):
    if OFFLINE or not HAS_SSL:
        return None
    cache_key = f"GET::{url}"
    cached = http_cache.get(cache_key)
    if cached is not None:
        return cached
    import requests  # lazy
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    http_cache.set(cache_key, r)
    return r


def _wfs_features(url: str) -> List[dict]:
    if OFFLINE or not HAS_SSL:
        return []
    try:
        r = _http_get(url)
        if r is None:
            return []
        ctype = r.headers.get("Content-Type", "")
        eprint("DEBUG", r.status_code, ctype, "\n ", url)
        if getattr(r, "status_code", 200) != 200 or "json" not in ctype:
            return []
        data = r.json()
        return data.get("features", [])
    except Exception as e:
        eprint("[ERR] WFS call:", e)
        return []


def fgr_from_point(lat: float, lon: float) -> Tuple[Optional[str], Optional[dict]]:
    """Zoek FGR-code + properties op. Offline: geef None terug (of hint via param in core)."""
    # 0. filter buiten NL
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None, None

    if OFFLINE or not HAS_SSL:
        return None, None

    # 1. RD-bbox ±100 m
    b = 100
    x1 = round(x - b, 3); y1 = round(y - b, 3)
    x2 = round(x + b, 3); y2 = round(y + b, 3)
    url_rd = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={FGR_TYPENAME}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:28992&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs_features(url_rd)
    if feats:
        props = feats[0].get("properties", {})
        return props.get("fgr"), props

    # 2. fallback WGS84-POINT
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={FGR_TYPENAME}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs_features(url_pt)
    if feats:
        props = feats[0].get("properties", {})
        return props.get("fgr"), props

    return None, None

# ───────────────────── interpretatie: bodem & droogte uit FGR
class BadRequest(ValueError):
    pass

_FGR_RULES = [
    (r"duin|stuif|strand|zand", ("zand", "hoog")),
    (r"rivier|oeverwal|komgrond|klei", ("klei", "middel")),
    (r"zeeklei|waard|polder|marien", ("klei", "middel")),
    (r"veen|moeras|peel", ("veen", "laag")),
    (r"l[öo]ss|leem", ("leem", "laag")),
]


def soil_from_fgr(fgr_text: Optional[str]) -> Tuple[str, str, str]:
    txt = (fgr_text or "").lower()
    for pattern, (bodem, droogte) in _FGR_RULES:
        if re.search(pattern, txt):
            return bodem, droogte, pattern
    return "leem", "middel", "fallback"

# ───────────────────── selectie helpers (compound matching)

def _tokenize(val: Any) -> List[str]:
    """Split op ; , / | en whitespace, lowercased en gestripte tokens."""
    if val is None:
        return []
    return [t for t in re.split(r"[;|,\\/\s]+", str(val).lower()) if t]


def _contains_ci(series: pd.Series, phrase: str) -> pd.Series:
    """Case-insensitive substring (legacy)."""
    return series.astype(str).str.contains(re.escape(phrase), case=False, na=False)


def _contains_token_ci(series: pd.Series, phrase: str) -> pd.Series:
    """Case-insensitive **token** match: 'klei' matcht 'klei;zand' maar niet 'lei'."""
    needle = str(phrase or "").strip().lower()
    if not needle:
        return series.astype(str).str.len() >= 0  # alles waar
    return series.apply(lambda v: needle in _tokenize(v))

# ───────────────────── advies selectie
class AdviceItem(BaseModel):
    naam: str
    bodem: Optional[str] = None
    droogtetolerantie: Optional[str] = None
    klimaatzone: Optional[str] = None
    zon_min: Optional[float] = None
    zon_max: Optional[float] = None
    ecowaarde: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class GeoAdviceResponse(BaseModel):
    fgr: str = Field(description="FGR code/naam indien beschikbaar")
    fgr_props: Dict[str, Any] = Field(default_factory=dict)
    bodem: str
    droogte: str
    reden: str = Field(description="Welke regel/patroon leidde tot deze interpretatie")
    filters: Dict[str, Any]
    advies: List[AdviceItem]
    took_ms: int


def _parse_light(licht: Optional[str], zon: Optional[int]) -> Tuple[int, str]:
    mapping = {"schaduw": 0, "half": 1, "halfschaduw": 1, "zon": 2}
    if licht:
        v = mapping.get(licht.strip().lower())
        if v is None:
            raise BadRequest("licht moet zijn: schaduw, half, zon")
        return v, licht
    if zon is None:
        return 2, "zon"
    if not (0 <= zon <= 2):
        raise BadRequest("zon (0..2) buiten bereik")
    rev = {0: "schaduw", 1: "half", 2: "zon"}
    return zon, rev[zon]


def select_advies(df: pd.DataFrame, zon_val: int, bodem: str, droogte: str, zone: str, limit: int = 15) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    # Zorg dat minimaal schema aanwezig is als select_advies los wordt aangeroepen
    if "naam" not in df.columns:
        df = _ensure_min_schema(df)

    # Token-based ANY matching voor samengestelde velden
    m_zon_min = pd.to_numeric(df.get("zon_min", 0), errors="coerce").fillna(0) <= zon_val
    m_zon_max = pd.to_numeric(df.get("zon_max", 2), errors="coerce").fillna(2) >= zon_val
    m_bodem   = _contains_token_ci(df.get("bodem", ""), bodem)
    m_droogte = _contains_token_ci(df.get("droogtetolerantie", ""), droogte)
    m_zone    = _contains_token_ci(df.get("klimaatzone", ""), zone)

    mask = m_zon_min & m_zon_max & m_bodem & m_droogte & m_zone
    sel = df.loc[mask].copy()

    if sel.empty:
        # Fallback: negeer droogte
        mask2 = m_zon_min & m_zon_max & m_bodem & m_zone
        sel = df.loc[mask2].copy()

    sel["__score"] = pd.to_numeric(sel.get("ecowaarde", 0), errors="coerce").fillna(0)
    sort_by = ["__score"] + (["naam"] if "naam" in sel.columns else [])
    sel = sel.sort_values(sort_by, ascending=[False] + [True] * (len(sort_by) - 1))

    cols = list(df.columns)
    records: List[Dict[str, Any]] = []
    for _, row in sel.head(limit).iterrows():
        base = {k: row.get(k) for k in [
            "naam", "bodem", "droogtetolerantie", "klimaatzone", "zon_min", "zon_max", "ecowaarde"
        ] if k in cols}
        extra = {k: row.get(k) for k in cols if k not in base}
        # Back-compat: zet bodem ook in extra zodat oudere code/tests hem daar kunnen vinden
        if "bodem" in base and "bodem" not in extra:
            extra["bodem"] = base["bodem"]
        base["extra"] = extra
        records.append(base)
    return records

# Kernfunctie die zowel server als offline CLI gebruikt

def advies_geo_core(
    lat: float,
    lon: float,
    licht: Optional[str] = None,
    zon: Optional[int] = None,
    klimaatzone: str = "7a",
    limit: int = 15,
    fgr_override: Optional[str] = None,  # alleen voor tests/offline
) -> GeoAdviceResponse:
    t0 = time.time()
    if fgr_override is not None:
        fgr, fgr_props = fgr_override, {"source": "override"}
    else:
        fgr, fgr_props = fgr_from_point(lat, lon)
    bodem, droogte, rule = soil_from_fgr(fgr)
    zon_val, licht_txt = _parse_light(licht, zon)
    df = load_csv()
    cache_key = f"adv::{round(lat,4)}::{round(lon,4)}::{zon_val}::{bodem}::{droogte}::{klimaatzone}::{limit}"
    cached = advies_cache.get(cache_key)
    if cached is None:
        adv = select_advies(df, zon_val, bodem, droogte, klimaatzone, limit)
        advies_cache.set(cache_key, adv)
    else:
        adv = cached
    took = round((time.time() - t0) * 1000)
    return GeoAdviceResponse(
        fgr=fgr or "Onbekend",
        fgr_props=fgr_props or {},
        bodem=bodem,
        droogte=droogte,
        reden=rule,
        filters={"licht": licht_txt, "zon_val": zon_val, "klimaatzone": klimaatzone, "limit": limit},
        advies=[AdviceItem(**item) for item in adv],
        took_ms=took,
    )

# ───────────────────── FastAPI integratie (UI + API)

def build_app():
    if OFFLINE or not HAS_SSL:
        # We bouwen GEEN app in omgevingen zonder ssl (voorkomt import-fouten in sommige sandboxes)
        return None
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    app = FastAPI(
        title="PlantWijs API 0.9.6",
        version="0.9.6",
        description=(
            "Adviezen voor landschappelijk geschikte en ecologisch waardevolle beplanting op basis van locatie.\n"
            "Gebruikt PDOK FGR + CSV-configuratie voor plantsoorten."
        ),
    )

    allow_origins = os.getenv("PLANTWIJS_CORS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:  # noqa: D401
        load_csv()

    # ---------- UI: inline kaartpagina op /
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return f"""
<!doctype html>
<html lang=\"nl\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>PlantWijs — Kaart</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\" crossorigin=\"\" />
  <style>
    html, body, #app {{ height:100%; margin:0; }}
    #map {{ height:100%; }}
    .panel {{ position:absolute; top:10px; right:10px; z-index:1000; background:#fff; padding:10px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.15); max-width:360px; }}
    .panel h1 {{ font-size:16px; margin:0 0 8px; }}
    .panel label {{ display:block; font-size:12px; margin:6px 0 2px; color:#444; }}
    .panel button {{ padding:6px 10px; border-radius:8px; border:1px solid #ddd; background:#f6f6f6; cursor:pointer; }}
    .list {{ margin-top:8px; max-height:50vh; overflow:auto; }}
    .item {{ padding:6px 0; border-bottom:1px solid #eee; }}
    .muted {{ color:#666; font-size:12px; }}
    .badge {{ display:inline-block; font-size:11px; padding:2px 6px; border-radius:999px; background:#eef; margin-left:6px; }}
  </style>
</head>
<body>
  <div id=\"app\">
    <div id=\"map\"></div>
    <div class=\"panel\">
      <h1>PlantWijs <span class=\"muted\">v{app.version}</span></h1>
      <label>Licht</label>
      <select id=\"licht\">
        <option value=\"zon\" selected>Zon</option>
        <option value=\"half\">Half</option>
        <option value=\"schaduw\">Schaduw</option>
      </select>
      <label>Klimaatzone</label>
      <input id=\"zone\" value=\"7a\" />
      <div style=\"margin-top:8px\">
        <button id=\"loc\">📍 Mijn locatie</button>
      </div>
      <div class=\"muted\" style=\"margin-top:6px\">Klik op de kaart om advies op te halen.</div>
      <div class=\"list\" id=\"list\"></div>
    </div>
  </div>

  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\" integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\" crossorigin=\"\"></script>
  <script>
    const map = L.map('map').setView([52.15, 5.29], 8);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap' }}).addTo(map);

    let marker = null;
    const list = document.getElementById('list');

    function htmlEscape(s) {{ return String(s||'').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}})[c]); }}

    async function fetchAdvies(lat, lon) {{
      const licht = document.getElementById('licht').value;
      const zone = document.getElementById('zone').value || '7a';
      const url = `/advies/geo?lat=${{lat}}&lon=${{lon}}&licht=${{encodeURIComponent(licht)}}&klimaatzone=${{encodeURIComponent(zone)}}&limit=15`;
      list.innerHTML = '<div class="muted">Laden...</div>';
      try {{
        const r = await fetch(url);
        const j = await r.json();
        renderList(j, lat, lon);
      }} catch(e) {{
        list.innerHTML = '<div class="muted">Fout bij ophalen advies</div>';
      }}
    }}

    function renderList(j, lat, lon) {{
      const items = (j.advies||[]).map(r => {{
        const naam = htmlEscape(r.naam);
        const eco = (r.ecowaarde ?? '') + '';
        const bodem = htmlEscape(r.bodem||'');
        const droogte = htmlEscape(r.droogtetolerantie||'');
        const zone = htmlEscape(r.klimaatzone||'');
        return `<div class="item"><strong>${{naam}}</strong> <span class="badge">eco ${{eco}}</span><div class="muted">bodem: ${{bodem}} · droogte: ${{droogte}} · zone: ${{zone}}</div></div>`;
      }}).join('');
      const fgr = htmlEscape(j.fgr||'Onbekend');
      list.innerHTML = `<div class="muted">FGR: ${{fgr}} · bodem: ${{htmlEscape(j.bodem)}} · droogte: ${{htmlEscape(j.droogte)}}</div>` + items;

      if (!marker) {{ marker = L.marker([lat, lon]).addTo(map); }}
      marker.setLatLng([lat, lon]).bindPopup(`<b>Advieslocatie</b><br/>Lat: ${{lat.toFixed(5)}}, Lon: ${{lon.toFixed(5)}}`).openPopup();
    }}

    map.on('click', (e) => {{
      const {{lat, lng}} = e.latlng; fetchAdvies(lat, lng);
    }});

    document.getElementById('loc').addEventListener('click', () => {{
      if (!navigator.geolocation) return alert('Geolocatie niet beschikbaar');
      navigator.geolocation.getCurrentPosition(pos => {{
        const lat = pos.coords.latitude, lon = pos.coords.longitude;
        map.setView([lat, lon], 12); fetchAdvies(lat, lon);
      }}, err => alert('Kon locatie niet ophalen'));
    }});
  </script>
</body>
</html>
        """

    @app.get("/health")
    def health() -> Dict[str, Any]:
        df = load_csv()
        return {
            "ok": True,
            "rows": int(df.shape[0]),
            "cols": list(df.columns),
            "csv_etag": _df_etag,
            "time": time.time(),
        }

    @app.post("/dataset/reload")
    def dataset_reload() -> Dict[str, Any]:
        df_old = load_csv()
        n_old = int(df_old.shape[0])
        df = load_csv(force=True)
        return {"ok": True, "rows": int(df.shape[0]), "diff": int(df.shape[0]) - n_old}

    @app.get("/lookup/fgr")
    def lookup_fgr(lat: float = Query(...), lon: float = Query(...)) -> Dict[str, Any]:
        t0 = time.time()
        fgr, props = fgr_from_point(lat, lon)
        return {"fgr": fgr or "Onbekend", "fgr_props": props or {}, "took_ms": round((time.time() - t0) * 1000)}

    @app.get("/advies/geo", response_model=GeoAdviceResponse)
    def advies_geo(
        lat: float = Query(..., description="Latitude (WGS84)"),
        lon: float = Query(..., description="Longitude (WGS84)"),
        licht: Optional[str] = Query(None, description="schaduw | half | zon"),
        zon: Optional[int] = Query(None, ge=0, le=2, description="(deprecated) 0..2"),
        klimaatzone: str = Query("7a", description="USDA-achtige zone string, bv 7a"),
        limit: int = Query(15, ge=1, le=50),
    ) -> GeoAdviceResponse:
        try:
            return advies_geo_core(lat, lon, licht, zon, klimaatzone, limit)
        except BadRequest as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/meta")
    def meta() -> Dict[str, Any]:
        df = load_csv()
        return {
            "title": "PlantWijs API",
            "version": "0.9.6",
            "csv_url": CSV_URL,
            "columns": list(df.columns),
            "row_count": int(df.shape[0]),
            "notes": [
                "Gebruik /advies/geo?lat=..&lon=..&licht=zon&klimaatzone=7a",
                "Voor batch-requests: roep meerdere keren parallel aan vanuit de front-end.",
            ],
        }

    return app

# Exporteer app als variabele indien mogelijk
app = build_app()

# ───────────────────── Offline CLI & Zelftests

def _print_json(data: Any) -> None:
    payload = _dumps(data)
    # Schrijf naar stdout: gebruik .buffer als beschikbaar, anders direct
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(payload)
    else:
        try:
            sys.stdout.write(payload if isinstance(payload, str) else payload.decode("utf-8", "replace"))
        except Exception:
            sys.stdout.write("{}")
    sys.stdout.write("\n")


def selftest() -> None:
    print("[SELFTEST] SSL beschikbaar:", HAS_SSL, "offline:", OFFLINE)
    # Gebruik ALTIJD de ingebedde test-CSV zodat de test deterministisch is
    df = _load_embedded()

    # 0) CSV opmerking met komma correct geparsed
    row_meidoorn = df[df["naam"].astype(str).str.contains("Meidoorn", case=False, na=False)].iloc[0]
    assert "," in str(row_meidoorn.get("opmerking", "")), "opmerking met komma moet als één veld zijn gelezen"
    assert str(row_meidoorn.get("opmerking", "")) == "Struweel, biodivers", "opmerking-veld onjuist geparsed"

    # 1) soil_from_fgr
    assert soil_from_fgr("Rivierkleigebied")[0:2] == ("klei", "middel")
    assert soil_from_fgr("Duinen")[0:2] == ("zand", "hoog")
    assert soil_from_fgr(None)[0:2] == ("leem", "middel")

    # 2) _parse_light
    assert _parse_light("half", None) == (1, "half")
    assert _parse_light(None, 2) == (2, "zon")
    try:
        _parse_light("invalid", None)
        raise AssertionError("invalid licht had moeten falen")
    except BadRequest:
        pass

    # 3) select_advies met embedded df (strikte match)
    recs = select_advies(df, zon_val=2, bodem="klei", droogte="hoog", zone="7a", limit=10)
    assert any("els" in str(r.get("naam", "")).lower() for r in recs), "verwacht Zwarte els bij klei/hoog"

    # 3b) select_advies fallback zonder droogte-filter
    recs_fb = select_advies(df, zon_val=2, bodem="klei", droogte="__nietbestaand__", zone="7a", limit=10)
    assert len(recs_fb) >= 1, "fallback-selectie had >=1 soort moeten geven"

    # 3c) compound bodem parsing (ANY-match op samengestelde bodemwaarden)
    df_sem = df.copy()
    # forceer een rij met samengestelde bodem, als die nog niet bestaat
    if not any(";" in str(x) for x in df_sem["bodem"]):
        new = df_sem.iloc[0:1].copy()
        new["naam"] = "Testsoort klei;zand"
        new["bodem"] = "klei;zand"
        new["droogtetolerantie"] = "middel"
        df_sem = pd.concat([df_sem, new], ignore_index=True)
    recs_sem = select_advies(df_sem, zon_val=2, bodem="klei", droogte="middel", zone="7a", limit=50)
    # Zoeken naar onze synthetische rij en controleren dat de compound waarde behouden is
    assert any("klei;zand" in str(r.get("extra", {}).get("bodem", "")) or
               "klei;zand" in str(r.get("bodem", "")) for r in recs_sem), "bodem 'klei;zand' had moeten matchen op 'klei'"

    # 3d) negatieve test: substring mag niet matchen ("lei" ≠ "klei")
    recs_neg = select_advies(df_sem, zon_val=2, bodem="lei", droogte="middel", zone="7a", limit=50)
    assert not any("Testsoort klei;zand" == r.get("naam") for r in recs_neg), "substring 'lei' mag niet matchen 'klei'"

    # 4) advies_geo_core offline met override (simulate rivierengebied)
    res = advies_geo_core(51.95, 5.99, licht="zon", klimaatzone="7a", limit=5, fgr_override="Rivierkleigebied")
    assert res.bodem == "klei" and res.droogte in {"laag", "middel", "hoog"}
    assert 1 <= len(res.advies) <= 5

    # 5) bbox-filter: punt buiten NL
    fgr, props = fgr_from_point(0.0, 0.0)
    assert fgr is None and props is None

    # 6) eprint en _print_json smoke tests
    eprint("[TEST] eprint ok")
    _print_json({"ok": True})

    # 7) Schemafix tests — 'naam' ontbreekt (hernoemd naar 'soort')
    df2 = df.rename(columns={"naam": "soort"})
    df2 = _normalize_columns(df2)
    df2 = _ensure_min_schema(df2)
    assert "naam" in df2.columns and df2["naam"].astype(str).str.len().ge(1).all(), "'naam' moet bestaan na schemafix"
    recs2 = select_advies(df2, zon_val=2, bodem="klei", droogte="hoog", zone="7a", limit=5)
    assert len(recs2) >= 1, "select_advies moet ook werken zonder originele 'naam'"

    # 8) Schemafix tests — 'naam' volledig weg
    df3 = df.drop(columns=["naam"])  # type: ignore
    df3 = _ensure_min_schema(df3)
    assert "naam" in df3.columns, "'naam' kolom moet gesynthetiseerd worden"
    recs3 = select_advies(df3, zon_val=2, bodem="klei", droogte="hoog", zone="7a", limit=5)
    assert len(recs3) >= 1

    print("ALL TESTS PASSED")


def run_example(argv: List[str]) -> None:
    if len(argv) < 3:
        eprint("Gebruik: python api.py --example <lat> <lon> [licht] [klimaatzone] [limit]")
        sys.exit(2)
    lat = float(argv[1]); lon = float(argv[2])
    licht = argv[3] if len(argv) > 3 else "zon"
    klimaatzone = argv[4] if len(argv) > 4 else "7a"
    limit = int(argv[5]) if len(argv) > 5 else 10
    # In offline modus: simuleer FGR als Rivierkleigebied zodat selectie wat teruggeeft
    fgr_override = "Rivierkleigebied" if (OFFLINE or not HAS_SSL) else None
    res = advies_geo_core(lat, lon, licht=licht, klimaatzone=klimaatzone, limit=limit, fgr_override=fgr_override)
    _print_json(json.loads(res.model_dump_json()))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    if "--example" in sys.argv:
        idx = sys.argv.index("--example")
        run_example(sys.argv[idx+1:])
        sys.exit(0)

    if OFFLINE or not HAS_SSL:
        eprint(
            "[INFO] SSL niet beschikbaar of PLANTWIJS_OFFLINE=1 gezet. Servermodus uitgeschakeld.\n"
            "       Gebruik --selftest of --example, of draai in een omgeving met Python + SSL.\n"
            "       Tip (Linux): installeer libssl-dev en bouw Python opnieuw, of gebruik pyenv/conda.",
        )
        # Voor de zekerheid: voer selftest uit zodat het script 'iets' doet in CI
        selftest()
    else:
        try:
            import uvicorn  # lazy import
            if app is None:
                app = build_app()
            uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", 9000)), reload=True)
        except Exception as e:
            eprint("[ERR] Kon server niet starten:", e)
            print("Val terug op selftest.")
            selftest()
