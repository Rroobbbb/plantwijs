r"""
PlantWijs API — v3.9 (PDOK auto-fix)
====================================
Wat is nieuw tov 3.7
- **Automatische WMS laag-detectie** via GetCapabilities (geen gok op laagnamen)
- **/api/wms_meta** en **/api/diag/featureinfo** voor snelle diagnose
- Kaartlagen: **BRO Bodemkaart (Bodemvlakken)**, **BRO Grondwaterspiegeldiepte (Gt/GLG/GHG)**, **FGR** — nu gegarandeerd zichtbaar als de service live is
- Klik op kaart: toont **FGR**, **Bodem** (bron) en **Gt → vochtklasse**; checkboxes worden **automatisch** gezet (maar zijn overrulbaar)
- JSON-cleaner tegen NaN/Inf fout

Starten (Windows)
  cd C:/PlantWijs
  venv/Scripts/uvicorn api:app --reload --port 9000
"""
from __future__ import annotations

import math
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pyproj import Transformer

# ───────────────────── PDOK endpoints
HEADERS = {"User-Agent": "plantwijs/3.9"}
FMT_JSON = "application/json;subtype=geojson"

# WFS FGR
PDOK_FGR_WFS = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0"
    "?service=WFS&version=2.0.0"
)
FGR_WMS = "https://service.pdok.nl/ez/fysischgeografischeregios/wms/v1_0"

# WMS Bodemkaart (BRO)
BODEM_WMS = "https://service.pdok.nl/bzk/bro-bodemkaart/wms/v1_0"

# WMS Grondwaterspiegeldiepte (BRO)
GWD_WMS = "https://service.pdok.nl/bzk/bro-grondwaterspiegeldiepte/wms/v2_0"

# ───────────────────── Proj
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ───────────────────── Dataset cache
DATA_PATHS = [
    "out/plantwijs_full_semicolon.csv",
    "out/plantwijs_full.csv",
]
_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None}

def _detect_sep(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"

def _load_df(path: str) -> pd.DataFrame:
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding_errors="ignore")
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    if "naam" not in df.columns and "nederlandse_naam" in df.columns:
        df = df.rename(columns={"nederlandse_naam": "naam"})
    if "wetenschappelijke_naam" not in df.columns:
        for k in ("taxon", "species"):
            if k in df.columns:
                df = df.rename(columns={k: "wetenschappelijke_naam"})
                break
    for must in ("standplaats_licht", "vocht", "inheems", "invasief"):
        if must not in df.columns:
            df[must] = ""
    return df

def get_df() -> pd.DataFrame:
    path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Geen dataset gevonden. Bouw eerst out/plantwijs_full_semicolon.csv met build_dataset.py")
    m = os.path.getmtime(path)
    if _CACHE["df"] is None or _CACHE["mtime"] != m or _CACHE["path"] != path:
        df = _load_df(path)
        _CACHE.update({"df": df, "mtime": m, "path": path})
        print(f"[DATA] geladen: {path} — {len(df)} rijen, {df.shape[1]} kolommen")
    return _CACHE["df"].copy()

# ───────────────────── HTTP utils
@lru_cache(maxsize=32)
def _get(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=12)

@lru_cache(maxsize=16)
def _capabilities(url: str) -> Optional[ET.Element]:
    try:
        r = _get(f"{url}?service=WMS&request=GetCapabilities")
        r.raise_for_status()
        return ET.fromstring(r.text)
    except Exception as e:
        print("[CAP] fout:", e)
        return None

# Zoek laag op Title of Name (case-insensitive substrings)
def _find_layer_name(url: str, want: List[str]) -> Optional[Tuple[str, str]]:
    root = _capabilities(url)
    if root is None:
        return None
    ns = {"wms": root.tag.split("}")[0].strip("{")}
    # pak alle Layer nodes
    layers = root.findall(".//{*}Layer")
    cand: List[Tuple[str,str]] = []  # (name,title)
    for layer in layers:
        name_el = layer.find("{*}Name")
        title_el = layer.find("{*}Title")
        name = (name_el.text if name_el is not None else "")
        title = (title_el.text if title_el is not None else "")
        if not name and not title:
            continue
        cand.append((name, title))
    lwant = [w.lower() for w in want]
    # 1) match op Title
    for name, title in cand:
        t = (title or "").lower()
        if any(w in t for w in lwant) and name:
            return name, title
    # 2) match op Name
    for name, title in cand:
        n = (name or "").lower()
        if any(w in n for w in lwant) and name:
            return name, title
    # 3) fallback: eerste met name
    for name, title in cand:
        if name:
            return name, title
    return None

# Resolve alle laagnamen één keer bij startup
_WMSMETA: Dict[str, Dict[str, str]] = {}

def _resolve_layers() -> None:
    global _WMSMETA
    meta: Dict[str, Dict[str, str]] = {}
    fgr = _find_layer_name(FGR_WMS, ["fysisch", "fgr"]) or ("fysischgeografischeregios", "FGR")
    bodem = _find_layer_name(BODEM_WMS, ["bodemvlakken", "bodem"]) or ("Bodemvlakken", "Bodemvlakken")
    gt = _find_layer_name(GWD_WMS, ["grondwatertrappen", "gt"]) or ("BRO Grondwaterspiegeldiepte Grondwatertrappen Gt", "Gt")
    ghg = _find_layer_name(GWD_WMS, ["ghg"]) or ("BRO Grondwaterspiegeldiepte GHG", "GHG")
    glg = _find_layer_name(GWD_WMS, ["glg"]) or ("BRO Grondwaterspiegeldiepte GLG", "GLG")
    meta["fgr"] = {"url": FGR_WMS, "layer": fgr[0], "title": fgr[1]}
    meta["bodem"] = {"url": BODEM_WMS, "layer": bodem[0], "title": bodem[1]}
    meta["gt"] = {"url": GWD_WMS, "layer": gt[0], "title": gt[1]}
    meta["ghg"] = {"url": GWD_WMS, "layer": ghg[0], "title": ghg[1]}
    meta["glg"] = {"url": GWD_WMS, "layer": glg[0], "title": glg[1]}
    _WMSMETA = meta
    print("[WMS] resolved:", meta)

_resolve_layers()

# ───────────────────── WFS/WMS helpers

def _wfs(url: str) -> List[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        if "json" not in r.headers.get("Content-Type", "").lower():
            return []
        return (r.json() or {}).get("features", [])
    except Exception:
        return []

_kv_re = re.compile(r"^\s*([A-Za-z0-9_\-\. ]+?)\s*[:=]\s*(.+?)\s*$")

def _parse_kv_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        m = _kv_re.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    if not out:
        stripped = re.sub(r"<[^>]+>", "\n", text)
        for line in stripped.splitlines():
            m = _kv_re.match(line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
    return out

# robuuste GetFeatureInfo (gebruikt dynamische laagnamen uit _WMSMETA)
_DEF_INFO_FORMATS = [
    "application/json",
    "application/geo+json",
    "application/json;subtype=geojson",
    "application/vnd.ogc.gml",
    "text/xml",
    "text/plain",
]

def _wms_getfeatureinfo(base_url: str, layer: str, lat: float, lon: float) -> dict | None:
    cx, cy = TX_WGS84_WEB.transform(lon, lat)
    m = 200.0
    bbox = f"{cx-m},{cy-m},{cx+m},{cy+m}"
    params_base = {
        "service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
        "layers": layer, "query_layers": layer, "styles": "",
        "crs": "EPSG:3857", "width": 101, "height": 101, "i": 50, "j": 50,
        "bbox": bbox,
    }
    params_base["feature_count"] = 10
    for fmt in _DEF_INFO_FORMATS:
        params = dict(params_base)
        params["info_format"] = fmt
        try:
            r = requests.get(base_url, params=params, headers=HEADERS, timeout=10)
            if not r.ok:
                continue
            ctype = r.headers.get("Content-Type", "").lower()
            if "json" in ctype:
                data = r.json() or {}
                feats = data.get("features") or []
                if feats:
                    props = feats[0].get("properties") or {}
                    if props:
                        return props
            text = r.text
            if text and fmt in ("text/plain", "text/xml", "application/vnd.ogc.gml"):
                return {"_text": text}
        except Exception:
            continue
    return None

# ───────────────────── PDOK value extractors

def fgr_from_point(lat: float, lon: float) -> str | None:
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None
    b = 100
    x1, y1, x2, y2 = round(x-b, 3), round(y-b, 3), round(x+b, 3), round(y+b, 3)
    layer_name = "fysischgeografischeregios:fysischgeografischeregios"
    url_rd = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={layer_name}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:28992&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs(url_rd)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={layer_name}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs(url_pt)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    return None

_SOIL_TOKENS = {
    "veen": {"veen"},
    "klei": {"klei", "zware klei", "lichte klei"},
    # NL: zavel wordt praktisch als leem/loam behandeld
    "leem": {"leem", "loess", "löss", "zavel"},
    "zand": {"zand", "dekzand"},
}

def _soil_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for soil, keys in _SOIL_TOKENS.items():
        for k in keys:
            if k in t:
                return soil
    return None


def bodem_from_bodemkaart(lat: float, lon: float) -> Tuple[Optional[str], dict]:
    layer = _WMSMETA.get("bodem", {}).get("layer") or "Bodemvlakken"
    props = _wms_getfeatureinfo(BODEM_WMS, layer, lat, lon) or {}

    # 1) JSON-properties die BRO vaak meegeeft (zie debug):
    for k in (
        "grondsoort", "bodem", "BODEM", "BODEMTYPE", "soil", "bodemtype", "SOILAREA_NAME", "NAAM",
        # nieuw: exacte BRO-velden
        "first_soilname", "normal_soilprofile_name",
    ):
        if k in props and props[k]:
            val = str(props[k])
            return _soil_from_text(val) or val, props

    # 2) Tekst/GML fallback → key:value of vrije tekst scannen
    if "_text" in props:
        kv = _parse_kv_text(props["_text"]) or {}
        for k in ("grondsoort", "BODEM", "bodemtype", "BODEMNAAM", "NAAM", "omschrijving",
                  "first_soilname", "normal_soilprofile_name"):
            if k in kv and kv[k]:
                val = kv[k]
                return _soil_from_text(val) or val, props
        so = _soil_from_text(props["_text"]) or None
        return so, props

    return None, props


def _vochtklasse_from_gt_code(gt: Optional[str]) -> Optional[str]:
    if not gt:
        return None
    gt = gt.upper().strip()
    if gt == "I":
        return "zeer nat"
    if gt == "II":
        return "nat"
    if gt in ("III", "IV"):
        return "vochtig"
    if gt == "V":
        return "droog"
    if gt in ("VI", "VII"):
        return "zeer droog"
    return None


def vocht_from_gwt(lat: float, lon: float) -> Tuple[Optional[str], dict, Optional[str]]:
    gt_layer = _WMSMETA.get("gt", {}).get("layer") or "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt"
    props = _wms_getfeatureinfo(GWD_WMS, gt_layer, lat, lon) or {}
    gt = None
    if props:
        for k in ("gt", "grondwatertrap", "GT", "Gt"):
            if k in props and props[k]:
                gt = str(props[k]).upper()
                break
        if not gt and "_text" in props:
            txt = props["_text"]
            kv = _parse_kv_text(txt)
            for k in ("gt", "grondwatertrap", "GT"):
                if k in kv:
                    gt = str(kv[k]).upper()
                    break
            if not gt:
                m = re.search(r"GT\s*([IVX]+)", txt, re.I)
                if m:
                    gt = m.group(1).upper()
    klass = _vochtklasse_from_gt_code(gt)
    # Nieuw: GT als value_list (cm)
    if not klass and "value_list" in props:
        try:
            _cm = int(float(str(props["value_list"]).replace(",", ".")))
            if _cm < 25:
                klass = "zeer nat"
            elif _cm < 40:
                klass = "nat"
            elif _cm < 80:
                klass = "vochtig"
            elif _cm < 120:
                klass = "droog"
            else:
                klass = "zeer droog"
        except Exception:
            pass
    if klass:
        return klass, props, gt

    # Fallback: GLG/GHG diepte
    for key in ("glg", "ghg"):
        lyr = _WMSMETA.get(key, {}).get("layer")
        if not lyr:
            continue
        p2 = _wms_getfeatureinfo(GWD_WMS, lyr, lat, lon) or {}
        txt = " ".join(str(v) for v in p2.values())
        m = re.search(r"(GLG|GHG)\s*[:=]?\s*(\d{1,3})", txt, re.I)
        depth = int(m.group(2)) if m else None
        if depth is not None:
            if depth < 25:
                return "zeer nat", p2, gt
            if depth < 40:
                return "nat", p2, gt
            if depth < 80:
                return "vochtig", p2, gt
            if depth < 120:
                return "droog", p2, gt
            return "zeer droog", p2, gt
    return None, {}, gt

# ───────────────────── filtering helpers

def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()

def _split_tokens(cell: Any) -> List[str]:
    return [t.strip().lower() for t in str(cell or "").replace("/",";").replace("|",";").split(";") if t.strip()]

def _match_multival(cell: Any, choices: List[str]) -> bool:
    if not choices:
        return True
    tokens = set(_split_tokens(cell))
    want = set(w.strip().lower() for w in choices if w.strip())
    return bool(tokens.intersection(want))

def _match_bodem_row(row: pd.Series, keuzes: List[str]) -> bool:
    if not keuzes:
        return True
    low = [k.lower() for k in keuzes]
    if "bodem" in row and _match_multival(row.get("bodem"), low):
        return True
    gs = str(row.get("grondsoorten", "")).lower()
    cats = set()
    if "zand" in gs: cats.add("zand")
    if "klei" in gs: cats.add("klei")
    if any(w in gs for w in ("leem","löss","loess")): cats.add("leem")
    if "veen" in gs: cats.add("veen")
    return bool(set(low).intersection(cats))

# ───────────────────── app + cleaners
app = FastAPI(title="PlantWijs API v3.9")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k:_clean(v) for k,v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    try:
        if pd.isna(o):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return o

# ───────────────────── API: diagnose/meta
@app.get("/api/wms_meta")
def api_wms_meta():
    return JSONResponse(_clean(_WMSMETA))

@app.get("/api/diag/featureinfo")
def api_diag(service: str = Query(..., pattern="^(bodem|gt|ghg|glg|fgr)$"), lat: float = Query(...), lon: float = Query(...)):
    if service == "fgr":
        return JSONResponse({"fgr": fgr_from_point(lat, lon)})
    base = {"bodem": BODEM_WMS, "gt": GWD_WMS, "ghg": GWD_WMS, "glg": GWD_WMS}[service]
    layer = _WMSMETA.get(service, {}).get("layer")
    props = _wms_getfeatureinfo(base, layer, lat, lon)
    return JSONResponse(_clean({"base": base, "layer": layer, "props": props}))

# ───────────────────── API: data
@app.get("/api/plants")
def api_plants(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),
    vocht: List[str] = Query(default=[]),
    bodem: List[str] = Query(default=[]),
    limit: int = Query(200, ge=1, le=1000),
    sort: str = Query("naam"),
    desc: bool = Query(False),
):
    df = get_df()
    if q:
        df = df[df.apply(lambda r: _contains_ci(r.get("naam"), q) or _contains_ci(r.get("wetenschappelijke_naam"), q), axis=1)]
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]
    if licht:
        df = df[df["standplaats_licht"].map(lambda v: _match_multival(v, licht))]
    if vocht:
        df = df[df["vocht"].map(lambda v: _match_multival(v, vocht))]
    if bodem:
        df = df[df.apply(lambda r: _match_bodem_row(r, bodem), axis=1)]
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)
    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "ellenberg_l_min","ellenberg_l_max","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t_min","ellenberg_t_max","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r_min","ellenberg_r_max","ellenberg_s_min","ellenberg_s_max",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")
    return JSONResponse(_clean({"count": int(len(df)), "items": items}))

@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    limit: int = Query(150, ge=1, le=1000),
):
    t0 = time.time()
    fgr = fgr_from_point(lat, lon) or "Onbekend"
    bodem_raw, props_bodem = bodem_from_bodemkaart(lat, lon)
    vocht_raw, props_gwt, gt_code = vocht_from_gwt(lat, lon)
    bodem_val = bodem_raw or "leem"
    vocht_val = vocht_raw or "vochtig"

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]
    df = df[df["vocht"].map(lambda v: _match_multival(v, [vocht_val]))]
    df = df[df.apply(lambda r: _match_bodem_row(r, [bodem_val]), axis=1)]

    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde"
    ) if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_val,
        "bodem_bron": "BRO Bodemkaart WMS" if bodem_raw else "fallback",
        "gt_code": gt_code,
        "vocht": vocht_val,
        "vocht_bron": "BRO Gt/GLG WMS" if vocht_raw else "fallback",
        "advies": items,
        "elapsed_ms": int((time.time()-t0)*1000),
    }
    return JSONResponse(_clean(out))

# ───────────────────── UI
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = r"""
<!doctype html>
<html lang=nl>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>PlantWijs v3.9</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --bg:#0b1321; --panel:#0f192e; --muted:#9aa4b2; --fg:#e6edf3; }
    * { box-sizing:border-box; }
    body { margin:0; font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; background:var(--bg); color:var(--fg); }
    header { padding:10px 14px; border-bottom:1px solid #1c2a42; position:sticky; top:0; background:var(--bg); z-index:10; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    header h1 { margin:0; font-size:18px; }
    .wrap { display:grid; grid-template-columns: minmax(520px, 68%) 1fr; gap:12px; padding:12px; }
    #map { height: calc(100vh - 130px); border-radius:12px; border:1px solid #1c2a42; box-shadow:0 0 0 1px rgba(255,255,255,.05) inset; }
    .panel { background:var(--panel); border:1px solid #1c2a42; border-radius:12px; padding:12px; }
    .checks label { display:inline-flex; gap:6px; align-items:center; background:#0c1730; border:1px solid #1f2c49; padding:6px 8px; border-radius:8px; margin-right:6px; }
    input[type=checkbox] { accent-color:#5aa9ff; }
    .muted { color:var(--muted); }
    .chips { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
    .chip { background:#0b1226; border:1px solid #1f2c49; padding:4px 8px; border-radius:999px; font-size:12px; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px 10px; border-bottom:1px solid #182742; }
    th { text-align:left; color:#b0b8c6; }
    .btn { background:#11325a; color:#e6edf3; border:1px solid #1f2c49; padding:6px 10px; border-radius:8px; cursor:pointer; }
    .row-dbg { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }
  </style>
</head>
<body>
  <header>
    <h1>🌿 PlantWijs</h1>
    <button id="btnLocate" class="btn">📍 Mijn locatie</button>
    <label class="muted"><input id="inhOnly" type="checkbox" checked> alleen inheemse</label>
    <label class="muted"><input id="exInv" type="checkbox" checked> sluit invasieve uit</label>
    <div class="checks">
      <span class="muted">Licht:</span>
      <label><input type="checkbox" name="licht" value="schaduw"> schaduw</label>
      <label><input type="checkbox" name="licht" value="halfschaduw"> halfschaduw</label>
      <label><input type="checkbox" name="licht" value="zon"> zon</label>
    </div>
    <div class="checks">
      <span class="muted">Vocht:</span>
      <label><input type="checkbox" name="vocht" value="zeer droog"> zeer droog</label>
      <label><input type="checkbox" name="vocht" value="droog"> droog</label>
      <label><input type="checkbox" name="vocht" value="vochtig"> vochtig</label>
      <label><input type="checkbox" name="vocht" value="nat"> nat</label>
      <label><input type="checkbox" name="vocht" value="zeer nat"> zeer nat</label>
    </div>
    <div class="checks">
      <span class="muted">Bodem:</span>
      <label><input type="checkbox" name="bodem" value="zand"> zand</label>
      <label><input type="checkbox" name="bodem" value="klei"> klei</label>
      <label><input type="checkbox" name="bodem" value="leem"> leem</label>
      <label><input type="checkbox" name="bodem" value="veen"> veen</label>
    </div>
  </header>

  <div class="wrap">
    <div>
      <div id="map"></div>
      <div class="panel" style="margin-top:8px">
        <div class="muted">Context (klik op de kaart):</div>
        <div id="ctxF" class="muted">FGR: —</div>
        <div id="ctxB" class="muted">Bodem: —</div>
        <div id="ctxG" class="muted">Gt: —</div>
        <details class="row-dbg"><summary>🧪 WMS debug</summary>
          <div id="dbg"></div>
        </details>
        <div class="muted" style="margin-top:6px">Actieve filters:</div>
        <div id="chips" class="chips"></div>
      </div>
    </div>
    <div class="panel">
      <div class="muted" id="count"></div>
      <table id="tbl">
        <thead>
          <tr>
            <th>Naam</th>
            <th>Wetenschappelijke naam</th>
            <th>Licht</th>
            <th>Vocht</th>
            <th>Bodem</th>
            <th>WHZ</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <script>
    const map = L.map('map').setView([52.1, 5.3], 8);
    const base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap' }).addTo(map);

    let overlays = {};

    async function loadWms(){
      const meta = await (await fetch('/api/wms_meta')).json();
      const make = (m, opacity)=> L.tileLayer.wms(m.url, { layers: m.layer, format:'image/png', transparent: true, opacity: opacity, version:'1.3.0', crs: L.CRS.EPSG3857 });
      overlays['BRO Bodemkaart (Bodemvlakken)'] = make(meta.bodem, 0.55).addTo(map);
      overlays['BRO Grondwatertrappen (Gt)'] = make(meta.gt, 0.45).addTo(map);
      overlays['FGR'] = make(meta.fgr, 0.45).addTo(map);
      L.control.layers({ 'OSM': base }, overlays, { collapsed:false }).addTo(map);
      document.getElementById('dbg').textContent = JSON.stringify(meta, null, 2);
    }

    function getChecked(name){ return Array.from(document.querySelectorAll('input[name="'+name+'"]:checked')).map(x=>x.value) }
    function setChecked(name, wanted){ const lw=(wanted||[]).map(s=>String(s||'').toLowerCase()); document.querySelectorAll('input[name="'+name+'"]').forEach(el=>{ el.checked = lw.includes(el.value.toLowerCase()); }); }
    function html(s){ return (s==null?'':String(s)).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;') }

    function renderChips(){
      const ctn = document.getElementById('chips');
      const chips = [];
      for(const [lbl,name] of [['licht','licht'],['vocht','vocht'],['bodem','bodem']]){
        const vals = getChecked(name);
        if(vals.length){ chips.push(`<span class="chip">${lbl}: ${vals.map(html).join(' / ')}</span>`); }
      }
      ctn.innerHTML = chips.join(' ');
    }

    async function fetchList(){
      const url = new URL(location.origin + '/api/plants');
      if(document.getElementById('inhOnly').checked) url.searchParams.set('inheems_only','true');
      if(document.getElementById('exInv').checked) url.searchParams.set('exclude_invasief','true');
      for(const v of getChecked('licht')) url.searchParams.append('licht', v);
      for(const v of getChecked('vocht')) url.searchParams.append('vocht', v);
      for(const v of getChecked('bodem')) url.searchParams.append('bodem', v);
      url.searchParams.set('limit','300');
      const r = await fetch(url); return r.json();
    }

    function renderRows(items){
      const tb = document.querySelector('#tbl tbody');
      tb.innerHTML = items.map(r=>`
        <tr>
          <td><strong>${html(r.naam||'')}</strong></td>
          <td class="muted">${html(r.wetenschappelijke_naam||'')}</td>
          <td>${html(r.standplaats_licht||'')}</td>
          <td>${html(r.vocht||'')}</td>
          <td>${html(r.bodem||r.grondsoorten||'')}</td>
          <td>${html(r.winterhardheidszone||'')}</td>
        </tr>`).join('');
    }

    async function refresh(){
      renderChips();
      const data = await fetchList();
      document.getElementById('count').textContent = data.count + ' resultaten (eerste ' + data.items.length + ')';
      renderRows(data.items||[]);
    }

    map.on('click', async (e)=>{
      if(window._marker) window._marker.remove(); window._marker = L.marker(e.latlng).addTo(map);
      const url = new URL(location.origin + '/advies/geo');
      url.searchParams.set('lat', e.latlng.lat); url.searchParams.set('lon', e.latlng.lng);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      const j = await (await fetch(url)).json();
      document.getElementById('ctxF').textContent = 'FGR: ' + (j.fgr||'—');
      document.getElementById('ctxB').textContent = 'Bodem: ' + (j.bodem||'—') + (j.bodem_bron?` (${j.bodem_bron})`:'');
      document.getElementById('ctxG').textContent = 'Gt: ' + (j.gt_code||'—') + (j.vocht?` → ${j.vocht}`:'');
      // auto-set van bodem-filter uitgeschakeld
      // auto-set van vocht-filter uitgeschakeld
renderChips();
      renderRows(j.advies||[]);
      document.getElementById('count').textContent = (j.advies?.length||0) + ' resultaten (auto-filter)';

      // Debug: laat ruwe GetFeatureInfo zien
      try{
        const meta = await (await fetch('/api/wms_meta')).json();
        const dbg = [];
        for(const key of ['bodem','gt','ghg','glg']){
          const p = new URL('/api/diag/featureinfo', location.origin);
          p.searchParams.set('service', key); p.searchParams.set('lat', e.latlng.lat); p.searchParams.set('lon', e.latlng.lng);
          const r = await (await fetch(p)).json();
          dbg.push(key+': '+JSON.stringify(r.props||r, null, 0).slice(0,400));
        }
        document.getElementById('dbg').textContent = dbg.join('\n\n');
      }catch(err){ /* ignore */ }
    });

    // Locatie-vinder
    document.getElementById('btnLocate').addEventListener('click', ()=>{
      if(!navigator.geolocation){ alert('Geolocatie niet ondersteund.'); return; }
      navigator.geolocation.getCurrentPosition(pos=>{
        const lat = pos.coords.latitude, lon = pos.coords.longitude;
        map.setView([lat,lon], 14);
        if(window._marker) window._marker.remove(); window._marker = L.marker([lat,lon]).addTo(map);
        map.fire('click', { latlng:{ lat, lng:lon } });
      }, err=>{ alert('Kon locatie niet ophalen'); });
    });

    for(const el of document.querySelectorAll('input[name="licht"], input[name="vocht"], input[name="bodem"], #inhOnly, #exInv')){
      el.addEventListener('change', refresh);
    }

    loadWms().then(refresh);
  </script>
</body>
</html>
"""
    return html
