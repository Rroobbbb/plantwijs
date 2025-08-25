r"""
PlantWijs API — v3.3
=====================
- Grotere kaart (full height) + lagen: BRO Bodemkaart (Bodemvlakken), BRO Grondwatertrappen (Gt), FGR.
- Klik op kaart ⇒ haal bodem, Gt (en klasse: zeer nat → zeer droog) en FGR op via PDOK WMS/WFS.
- Zet automatisch de filters bodem + vocht; gebruiker kan overrulen met checkboxes.
- Toon onder de kaart: FGR, Bodem (bron), Gt-code + vochtklasse, en actieve filter-chips.
- Filtert soorten op licht/vocht/bodem en (optioneel) inheems/invasief.
- JSON-sanitizer voorkomt “Out of range float values”.

Starten (Windows):
  cd C:/PlantWijs
  venv/Scripts/uvicorn api:app --reload --port 9000
"""

from __future__ import annotations

import math
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pyproj import Transformer

# ───────────────────── PDOK endpoints
PDOK_FGR_WFS = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0"
    "?service=WFS&version=2.0.0"
)
FGR_LAYER = "fysischgeografischeregios:fysischgeografischeregios"

PDOK_BODEM_WMS = "https://service.pdok.nl/bzk/bro-bodemkaart/wms/v1_0"
PDOK_GWD_WMS = "https://service.pdok.nl/bzk/bro-grondwaterspiegeldiepte/wms/v2_0"
BODEM_LAYER = "Bodemvlakken"
GWT_LAYER_CANDIDATES = [
    "BRO Grondwaterspiegeldiepte Grondwatertrappen Gt",
    "GrondwatertrappenGt",
    "Grondwatertrappen_Gt",
]
HEADERS = {"User-Agent": "plantwijs/3.3"}
FMT_JSON = "application/json;subtype=geojson"

# ───────────────────── Proj
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)
TX_WGS84_WEB = Transformer.from_crs(4326, 3857, always_xy=True)

# ───────────────────── Dataset-setup
DATA_PATHS = [
    os.getenv("OUT_CSV_SEMI", "out/plantwijs_full_semicolon.csv"),
    os.getenv("OUT_CSV", "out/plantwijs_full.csv"),
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
    path = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise FileNotFoundError("Geen dataset gevonden. Bouw eerst out/plantwijs_full_semicolon.csv met build_dataset.py")
    m = os.path.getmtime(path)
    if _CACHE["df"] is None or _CACHE["mtime"] != m or _CACHE["path"] != path:
        df = _load_df(path)
        _CACHE.update({"df": df, "mtime": m, "path": path})
        print(f"[DATA] geladen: {path} — {len(df)} rijen, {df.shape[1]} kolommen")
    return _CACHE["df"].copy()

# ───────────────────── PDOK helpers

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

def _wms_getfeatureinfo(base_url: str, layer: str, lat: float, lon: float, crs: str = "EPSG:3857") -> dict | None:
    if crs == "EPSG:3857":
        cx, cy = TX_WGS84_WEB.transform(lon, lat)
        m = 50.0
        minx, miny, maxx, maxy = cx - m, cy - m, cx + m, cy + m
        bbox = f"{minx},{miny},{maxx},{maxy}"
        srs = "EPSG:3857"; i = j = 50
    else:
        d = 0.0005
        bbox = f"{lon-d},{lat-d},{lon+d},{lat+d}"
        srs = "EPSG:4326"; i = j = 50
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
        "layers": layer, "query_layers": layer, "crs": srs,
        "info_format": "application/json", "width": 101, "height": 101,
        "i": i, "j": j, "bbox": bbox,
    }
    try:
        r = requests.get(base_url, params=params, headers=HEADERS, timeout=8)
        if r.ok and "json" in r.headers.get("Content-Type", "").lower():
            data = r.json() or {}
            feats = data.get("features") or []
            if feats:
                return feats[0].get("properties") or {}
    except Exception as e:
        print("[GFI JSON]", e)
    try:
        params["info_format"] = "text/plain"
        r = requests.get(base_url, params=params, headers=HEADERS, timeout=8)
        if r.ok:
            return {"_text": r.text}
    except Exception as e:
        print("[GFI TEXT]", e)
    return None

def fgr_from_point(lat: float, lon: float) -> str | None:
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None
    b = 100
    x1, y1, x2, y2 = round(x-b, 3), round(y-b, 3), round(x+b, 3), round(y+b, 3)
    url_rd = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={FGR_LAYER}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:28992&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs(url_rd)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_FGR_WFS}&request=GetFeature&typenames={FGR_LAYER}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs(url_pt)
    if feats:
        return feats[0].get("properties", {}).get("fgr")
    return None

def bodem_from_bodemkaart(lat: float, lon: float) -> tuple[Optional[str], dict]:
    props = _wms_getfeatureinfo(PDOK_BODEM_WMS, BODEM_LAYER, lat, lon) or {}
    val = None
    for k in ("grondsoort", "BODEM", "bodem", "soil", "bodemtype"):
        if k in props and props[k]:
            val = str(props[k]); break
    txt = (val or " ".join(str(v) for v in props.values())).lower()
    if "veen" in txt:  return "veen", props
    if "klei" in txt:  return "klei", props
    if any(w in txt for w in ("löss", "loess", "leem")): return "leem", props
    if any(w in txt for w in ("zand", "sand")): return "zand", props
    return None, props

def vocht_from_gwt(lat: float, lon: float) -> tuple[Optional[str], dict, Optional[str]]:
    props = None
    for layer in GWT_LAYER_CANDIDATES:
        props = _wms_getfeatureinfo(PDOK_GWD_WMS, layer, lat, lon)
        if props:
            break
    if not props:
        return None, {}, None
    alltxt = " ".join(str(v) for v in props.values()).upper()
    gt = None
    for k in ("gt", "grondwatertrap"):
        if k in props and props[k]:
            gt = str(props[k]).upper(); break
    if not gt:
        for cand in ("I", "II", "III", "IV", "V", "VI", "VII"):
            if f"GT {cand}" in alltxt or f"({cand})" in alltxt or f" {cand} " in alltxt:
                gt = cand; break
    klass = None
    if gt in ("I",): klass = "zeer nat"
    elif gt in ("II",): klass = "nat"
    elif gt in ("III", "IV"): klass = "vochtig"
    elif gt in ("V",): klass = "droog"
    elif gt in ("VI", "VII"): klass = "zeer droog"
    return klass, (props or {}), gt

# ───────────────────── Filtering helpers

def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()

def _match_multival(cell: Any, choices: List[str]) -> bool:
    if not choices:
        return True
    tokens = [t.strip().lower() for t in str(cell or "").replace("/", ";").replace("|", ";").split(";") if t.strip()]
    want = set(w.strip().lower() for w in choices if w.strip())
    return bool(want.intersection(tokens))

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
    if any(w in gs for w in ("leem", "löss", "loess")): cats.add("leem")
    if "veen" in gs: cats.add("veen")
    return bool(set(low).intersection(cats))

# ───────────────────── FastAPI
app = FastAPI(title="PlantWijs API v3.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)

def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    try:
        if pd.isna(o):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return o

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
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "ellenberg_l_min","ellenberg_l_max","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t_min","ellenberg_t_max","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r_min","ellenberg_r_max","ellenberg_s_min","ellenberg_s_max",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde","bodem"
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
    bodem_guess, vocht_guess = "leem", "vochtig"
    bodem_bk, props_bodem = bodem_from_bodemkaart(lat, lon)
    vocht_gt, props_gwt, gt_code = vocht_from_gwt(lat, lon)
    bodem_val = bodem_bk or bodem_guess
    vocht_val = vocht_gt or vocht_guess

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]
    df = df[df["vocht"].map(lambda v: _match_multival(v, [vocht_val]))]
    df = df[df.apply(lambda r: _match_bodem_row(r, [bodem_val]), axis=1)]

    cols = [c for c in (
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "hoogte","breedte","winterhardheidszone","grondsoorten","ecowaarde","bodem"
    ) if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")

    out = {
        "fgr": fgr,
        "bodem": bodem_val,
        "bodem_bron": "BRO Bodemkaart WMS" if bodem_bk else "fallback",
        "gt_code": gt_code,
        "vocht": vocht_val,
        "vocht_bron": "BRO Gt WMS" if vocht_gt else "fallback",
        "advies": items,
        "elapsed_ms": int((time.time()-t0)*1000),
    }
    return JSONResponse(_clean(out))

# ───────────────────── Homepage (UI)
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = r"""
<!doctype html>
<html lang=nl>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>PlantWijs v3.3</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --bg:#0b1321; --panel:#0f192e; --muted:#9aa4b2; --fg:#e6edf3; }
    * { box-sizing:border-box; }
    body { margin:0; font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; background:var(--bg); color:var(--fg); }
    header { padding:10px 14px; border-bottom:1px solid #1c2a42; position:sticky; top:0; background:var(--bg); z-index:10; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    header h1 { margin:0; font-size:18px; }
    .wrap { display:grid; grid-template-columns: 58% 42%; gap:12px; padding:12px; }
    #map { height: calc(100vh - 120px); border-radius:12px; border:1px solid #1c2a42; box-shadow:0 0 0 1px rgba(255,255,255,.05) inset; }
    .panel { background:var(--panel); border:1px solid #1c2a42; border-radius:12px; padding:12px; }
    .checks label { display:inline-flex; gap:6px; align-items:center; background:#0c1730; border:1px solid #1f2c49; padding:6px 8px; border-radius:8px; margin-right:6px; }
    input[type=checkbox] { accent-color:#5aa9ff; }
    .muted { color:var(--muted); }
    .chips { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
    .chip { background:#0b1226; border:1px solid #1f2c49; padding:4px 8px; border-radius:999px; font-size:12px; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px 10px; border-bottom:1px solid #182742; }
    th { text-align:left; color:#b0b8c6; }
  </style>
</head>
<body>
  <header>
    <h1>🌿 PlantWijs</h1>
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
    // Leaflet
    const map = L.map('map').setView([52.1, 5.3], 8);
    const base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OSM' }).addTo(map);

    // WMS overlays (aan bij start)
    const wmsBodem = L.tileLayer.wms('{PDOK_BODEM_WMS}', {
      layers: '{BODEM_LAYER}', format: 'image/png', transparent: true, opacity: 0.55, version: '1.3.0', attribution: 'BRO Bodemkaart (PDOK)'
    }).addTo(map);
    const wmsFGR = L.tileLayer.wms('https://service.pdok.nl/ez/fysischgeografischeregios/wms/v1_0', {
      layers: 'fysischgeografischeregios', format: 'image/png', transparent: true, opacity: 0.45, version: '1.3.0', attribution: 'FGR (PDOK)'
    }).addTo(map);
    const wmsGt = L.tileLayer.wms('{PDOK_GWD_WMS}', {
      layers: 'BRO Grondwaterspiegeldiepte Grondwatertrappen Gt', format: 'image/png', transparent: true, opacity: 0.45, version: '1.3.0', attribution: 'BRO Gt (PDOK)'
    }).addTo(map);
    L.control.layers({ 'OSM': base }, {
      'BRO Bodemkaart (Bodemvlakken)': wmsBodem,
      'BRO Grondwatertrappen (Gt)': wmsGt,
      'FGR': wmsFGR
    }, { collapsed:false }).addTo(map);

    let marker = null;

    function getChecked(name){ return Array.from(document.querySelectorAll('input[name="'+name+'"]:checked')).map(x=>x.value) }
    function setChecked(name, wanted){
      const lw = (wanted||[]).map(s=>String(s||'').toLowerCase());
      document.querySelectorAll('input[name="'+name+'"]').forEach(el=>{ el.checked = lw.includes(el.value.toLowerCase()); });
    }
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
      if(marker) marker.remove(); marker = L.marker(e.latlng).addTo(map);
      const url = new URL(location.origin + '/advies/geo');
      url.searchParams.set('lat', e.latlng.lat); url.searchParams.set('lon', e.latlng.lng);
      url.searchParams.set('inheems_only', document.getElementById('inhOnly').checked);
      url.searchParams.set('exclude_invasief', document.getElementById('exInv').checked);
      const j = await (await fetch(url)).json();
      document.getElementById('ctxF').textContent = 'FGR: ' + (j.fgr||'—');
      document.getElementById('ctxB').textContent = 'Bodem: ' + (j.bodem||'—') + (j.bodem_bron?` (${j.bodem_bron})`:'');
      document.getElementById('ctxG').textContent = 'Gt: ' + (j.gt_code||'—') + (j.vocht?` → ${j.vocht}`:'');
      if(j.bodem) setChecked('bodem', [j.bodem]);
      if(j.vocht) setChecked('vocht', [j.vocht]);
      renderChips();
      renderRows(j.advies||[]);
      document.getElementById('count').textContent = (j.advies?.length||0) + ' resultaten (auto-filter)';
    });

    for(const el of document.querySelectorAll('input[name="licht"], input[name="vocht"], input[name="bodem"], #inhOnly, #exInv')){
      el.addEventListener('change', refresh);
    }

    refresh();
  </script>
</body>
</html>
"""
    return html.replace('{PDOK_BODEM_WMS}', PDOK_BODEM_WMS).replace('{BODEM_LAYER}', BODEM_LAYER).replace('{PDOK_GWD_WMS}', PDOK_GWD_WMS)
