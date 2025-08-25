r"""
PlantWijs API — V3.1
=====================
Starten (Windows):
  cd C:\PlantWijs
  venv\Scripts\uvicorn api:app --reload --port 9000

Wijzigingen in V3.1
-------------------
- **Grotere kaart** (±72vh) en bredere kolom (±58% van pagina). Responsive fallback naar 1 kolom.
- **Filters licht/vocht → checkboxes** (meervoud tegelijk selecteerbaar i.p.v. multi-select).
- **JSON NaN/Inf fix**: alle NaN/Inf uit responses worden naar `null` geconverteerd (lost `ValueError: Out of range float values` op).

Vereist:
- `out/plantwijs_full.csv` (of `out/plantwijs_full_semicolon.csv`) uit je builder.
- Internet voor PDOK WFS en Leaflet op de homepage.
"""
from __future__ import annotations

import math
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware

# Probeer upload types optioneel te importeren (geen crash als python-multipart ontbreekt)
try:
    from fastapi import File, UploadFile
    HAVE_UPLOAD = True
except Exception:  # pragma: no cover
    HAVE_UPLOAD = False

# ----------------------------- PDOK FGR -----------------------------
PDOK_ROOT = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0"
    "?service=WFS&version=2.0.0"
)
LAYER = "fysischgeografischeregios:fysischgeografischeregios"
FMT_JSON = "application/json;subtype=geojson"
HEADERS = {"User-Agent": "plantwijs/1.0"}

from pyproj import Transformer
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)


def _wfs(url: str) -> List[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return []
        if "json" not in r.headers.get("Content-Type", ""):
            return []
        return r.json().get("features", [])
    except Exception:
        return []


def fgr_from_point(lat: float, lon: float) -> Optional[str]:
    """Probeer eerst kleine RD-bbox (±100 m), valt terug op WGS84-POINT."""
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None
    b = 100
    x1, y1, x2, y2 = round(x - b, 3), round(y - b, 3), round(x + b, 3), round(y + b, 3)
    url_rd = (
        f"{PDOK_ROOT}&request=GetFeature&typenames={LAYER}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:28992&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs(url_rd)
    if feats:
        return feats[0]["properties"].get("fgr")
    # fallback point intersect
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_ROOT}&request=GetFeature&typenames={LAYER}"
        f"&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs(url_pt)
    if feats:
        return feats[0]["properties"].get("fgr")
    return None


def soil_from_fgr(fgr: Optional[str]) -> tuple[str, str]:
    txt = (fgr or "").lower()
    if "duin" in txt or "zand" in txt:
        return "zand", "hoog"
    if "klei" in txt:
        return "klei", "middel"
    if "veen" in txt:
        return "veen", "laag"
    if "löss" in txt or "loess" in txt or "leem" in txt:
        return "leem", "laag"
    return "leem", "middel"

# ----------------------------- Dataset cache -----------------------------
DATA_PATHS = [
    os.getenv("OUT_CSV", "out/plantwijs_full.csv"),
    os.getenv("OUT_CSV_SEMI", "out/plantwijs_full_semicolon.csv"),
]

_CACHE: Dict[str, Any] = {"df": None, "mtime": None, "path": None}


def _detect_sep(path: str) -> str:
    # Detecteer ; of ,
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2048)
        return ";" if head.count(";") > head.count(",") else ","
    except Exception:
        return ","


def _load_df(path: str) -> pd.DataFrame:
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding_errors="ignore")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Zorg dat de belangrijkste kolommen bestaan
    if "naam" not in df.columns and "nederlandse_naam" in df.columns:
        df = df.rename(columns={"nederlandse_naam": "naam"})
    if "wetenschappelijke_naam" not in df.columns:
        for k in ("taxon", "species"):
            if k in df.columns:
                df = df.rename(columns={k: "wetenschappelijke_naam"})
                break
    for must in ("standplaats_licht","vocht","inheems","invasief","zon_min","zon_max"):
        if must not in df.columns:
            df[must] = "" if must not in ("zon_min","zon_max") else (0 if must=="zon_min" else 2)
    return df


def get_df() -> pd.DataFrame:
    # kies eerste bestaande pad
    path = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("Geen dataset gevonden. Bouw eerst out/plantwijs_full.csv met build_dataset.py")
    m = os.path.getmtime(path)
    if _CACHE["df"] is None or _CACHE["mtime"] != m or _CACHE["path"] != path:
        df = _load_df(path)
        _CACHE.update({"df": df, "mtime": m, "path": path})
        print(f"[DATA] geladen: {path} — {len(df)} rijen, {df.shape[1]} kolommen")
    return _CACHE["df"].copy()

# ----------------------------- Helpers JSON-schoon -----------------------------

def _clean_records(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in recs:
        rr: Dict[str, Any] = {}
        for k, v in r.items():
            try:
                if v is None:
                    rr[k] = None
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rr[k] = None
                elif pd.isna(v):  # type: ignore[arg-type]
                    rr[k] = None
                else:
                    rr[k] = v
            except Exception:
                rr[k] = v
        out.append(rr)
    return out

# ----------------------------- Filtering -----------------------------

def _contains_ci(s: Any, needle: str) -> bool:
    return needle.lower() in str(s or "").lower()


def _match_multival(cell: str, choices: List[str]) -> bool:
    if not choices:
        return True
    tokens = [t.strip().lower() for t in str(cell or "").replace("/", ";").replace("|", ";").split(";") if t.strip()]
    want = set([w.strip().lower() for w in choices if w.strip()])
    return bool(want.intersection(tokens))

# ----------------------------- FastAPI -----------------------------
app = FastAPI(title="PlantWijs API V3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/plants")
def api_plants(
    q: str = Query(""),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    licht: List[str] = Query(default=[]),  # herhaalbare query param
    vocht: List[str] = Query(default=[]),
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
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)
    cols_priority = [
        "naam","wetenschappelijke_naam","inheems","invasief","standplaats_licht","vocht",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "ellenberg_l_min","ellenberg_l_max","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t_min","ellenberg_t_max","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r_min","ellenberg_r_max","ellenberg_s_min","ellenberg_s_max",
        "hoogte","breedte","winterhardheidszone","grondsoorten",
        "stikstofbinder","ecowaarde","bodem","droogtetolerantie","klimaatzone","opmerking",
    ]
    cols = [c for c in cols_priority if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")
    items = _clean_records(items)
    return {"count": int(len(df)), "items": items}


@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    zon: int = Query(2, ge=0, le=2),
    inheems_only: bool = Query(True),
    exclude_invasief: bool = Query(True),
    limit: int = Query(150, ge=1, le=1000),
):
    from pyproj import Transformer  # lokaal houden voor snellere cold start

    t0 = time.time()
    fgr = fgr_from_point(lat, lon) or "Onbekend"
    bodem, droogte = soil_from_fgr(fgr)

    df = get_df()
    if inheems_only and "inheems" in df.columns:
        df = df[df["inheems"].astype(str).str.lower() == "ja"]
    if exclude_invasief and "invasief" in df.columns:
        df = df[(df["invasief"].astype(str).str.lower() != "ja") | (df["invasief"].isna())]

    # filter op zon-range als beschikbare kolommen bestaan
    if "zon_min" in df.columns and "zon_max" in df.columns:
        df = df[(pd.to_numeric(df["zon_min"], errors="coerce").fillna(0) <= zon) & (pd.to_numeric(df["zon_max"], errors="coerce").fillna(2) >= zon)]

    cols_priority = [
        "naam","wetenschappelijke_naam","inheems","standplaats_licht","vocht","stikstofbinder",
        "hoogte","breedte","winterhardheidszone","grondsoorten",
        "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
        "ellenberg_l_min","ellenberg_l_max","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t_min","ellenberg_t_max","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r_min","ellenberg_r_max","ellenberg_s_min","ellenberg_s_max",
        "ecowaarde","opmerking",
    ]
    cols = [c for c in cols_priority if c in df.columns]
    items = df[cols].head(limit).to_dict(orient="records")
    items = _clean_records(items)

    dt_ms = int((time.time() - t0) * 1000)
    return {
        "fgr": fgr,
        "bodem": bodem,
        "droogte": droogte,
        "elapsed_ms": dt_ms,
        "advies": items,
    }


# ----------------------------- Upload (optioneel) -----------------------------
if HAVE_UPLOAD:
    from fastapi import File, UploadFile

    @app.post("/dataset/upload")
    async def dataset_upload(file: UploadFile = File(...)):
        """Upload een CSV; wordt opgeslagen als out/plantwijs_full.csv en direct gebruikt."""
        content = await file.read()
        os.makedirs("out", exist_ok=True)
        target = os.path.join("out", "plantwijs_full.csv")
        with open(target, "wb") as f:
            f.write(content)
        # reset cache
        _CACHE.update({"df": None, "mtime": None, "path": None})
        # Forceer een load om te valideren
        try:
            _ = get_df()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": target}
else:  # pragma: no cover
    @app.post("/dataset/upload")
    def dataset_upload_unavailable():
        return {"ok": False, "error": "Upload niet beschikbaar: installeer 'python-multipart'"}


# ----------------------------- Homepage -----------------------------
@app.get("/")
def index() -> Response:
    html = r"""
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlantWijs</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --bg:#0f172a; --card:#111827; --muted:#9ca3af; --fg:#e5e7eb; --acc:#22c55e; }
    * { box-sizing:border-box; }
    body { margin:0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--fg); background:linear-gradient(180deg,#0b1220,#0f172a); }
    header { padding:14px 18px; display:flex; gap:12px; align-items:center; border-bottom:1px solid #1f2937; position:sticky; top:0; background:#0f172a; z-index:10; flex-wrap:wrap; }
    header h1 { margin:0; font-size:18px; letter-spacing:0.5px; }
    .chip { background:#1f2937; color:#e5e7eb; padding:8px 10px; border-radius:999px; display:inline-flex; align-items:center; gap:8px; }
    .chip input[type=checkbox] { transform:scale(1.2); }
    .wrap { display:grid; grid-template-columns: 58% 42%; gap:12px; padding:12px; }
    #map { height: 72vh; border-radius:12px; box-shadow:0 4px 24px rgba(0,0,0,.35); border:1px solid #1f2937; }
    .panel { background:#0b1220; border:1px solid #1f2937; border-radius:12px; padding:12px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
    input[type=text] { background:#0b1220; color:#e5e7eb; border:1px solid #334155; padding:8px 10px; border-radius:8px; flex:1; min-width:240px; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px 10px; border-bottom:1px solid #1f2937; vertical-align:top; }
    th { text-align:left; color:#a1a1aa; font-weight:600; cursor:pointer; }
    tr:hover { background:#0b1120; }
    .muted { color:#9ca3af; font-size:12px; }
    .badge { background:#1f2937; color:#a1a1aa; padding:2px 6px; border-radius:6px; font-size:12px; }
    .pill { background:#111827; padding:2px 6px; border-radius:999px; font-size:12px; }
    footer { padding:18px; text-align:center; color:#6b7280; }
    .group { display:flex; gap:8px; align-items:center; }
    .checks { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .checks label { display:inline-flex; gap:6px; align-items:center; background:#111827; border:1px solid #334155; padding:6px 8px; border-radius:8px; }
    @media (max-width: 1100px) {
      .wrap { grid-template-columns: 1fr; }
      #map { height: 48vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>🌿 PlantWijs</h1>
    <span class="chip"><input id="inhOnly" type="checkbox" checked> <label for="inhOnly">Alleen inheemse</label></span>
    <span class="chip"><input id="exInv" type="checkbox" checked> <label for="exInv">Sluit invasieve soorten uit</label></span>

    <div class="group">
      <span class="muted">Licht:</span>
      <div class="checks">
        <label><input type="checkbox" name="licht" value="schaduw"> <span>schaduw</span></label>
        <label><input type="checkbox" name="licht" value="halfschaduw"> <span>halfschaduw</span></label>
        <label><input type="checkbox" name="licht" value="zon"> <span>zon</span></label>
      </div>
    </div>

    <div class="group">
      <span class="muted">Vocht:</span>
      <div class="checks">
        <label><input type="checkbox" name="vocht" value="zeer droog"> <span>zeer droog</span></label>
        <label><input type="checkbox" name="vocht" value="droog"> <span>droog</span></label>
        <label><input type="checkbox" name="vocht" value="vochtig"> <span>vochtig</span></label>
        <label><input type="checkbox" name="vocht" value="nat"> <span>nat</span></label>
        <label><input type="checkbox" name="vocht" value="zeer nat"> <span>zeer nat</span></label>
      </div>
    </div>

    <input id="q" type="text" placeholder="Zoek op naam of wetenschappelijke naam…"/>
  </header>

  <div class="wrap">
    <div>
      <div id="map"></div>
      <div class="panel" style="margin-top:8px">
        <div class="muted">Klik op de kaart voor locatie-advies. Zonpositie:
          <select id="zonSel"><option value="0">schaduw</option><option value="1" selected>halfschaduw</option><option value="2">zon</option></select>
        </div>
        <div id="ctx" class="muted" style="margin-top:6px"></div>
      </div>
    </div>
    <div class="panel">
      <div class="muted" id="count"></div>
      <table id="tbl">
        <thead>
          <tr>
            <th data-k="naam">Naam</th>
            <th data-k="wetenschappelijke_naam">Wetenschappelijke naam</th>
            <th data-k="standplaats_licht">Licht</th>
            <th data-k="vocht">Vocht</th>
            <th data-k="ellenberg_f">F</th>
            <th data-k="inheems">Inheems</th>
            <th data-k="invasief">Invasief</th>
            <th data-k="hoogte">H</th>
            <th data-k="breedte">B</th>
            <th data-k="winterhardheidszone">WHZ</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <footer>© PlantWijs</footer>

  <script>
    // Leaflet
    var map = L.map('map').setView([52.1, 5.3], 8);
    var base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OSM' }).addTo(map);

    // PDOK WMS overlays
    var wmsBodem = L.tileLayer.wms('https://service.pdok.nl/bzk/bro-bodemkaart/wms/v1_0', {
      layers: 'Bodemvlakken',
      format: 'image/png',
      transparent: true,
      attribution: 'BRO Bodemkaart (PDOK)'
    });
    var wmsFGR = L.tileLayer.wms('https://service.pdok.nl/ez/fysischgeografischeregios/wms/v1_0', {
      layers: 'fysischgeografischeregios',
      format: 'image/png',
      transparent: true,
      attribution: 'FGR (PDOK)'
    });
    L.control.layers({ 'OpenStreetMap': base }, { 'BRO Bodemkaart (SGM)': wmsBodem, 'Fysisch-Geografische Regio’s': wmsFGR }, { collapsed:false }).addTo(map);

    var marker = null;

    function qs(id){return document.getElementById(id)}
    function cbValues(name){ return Array.from(document.querySelectorAll('input[name="'+name+'"]:checked')).map(i=>i.value) }

    async function fetchPlants(){
      const url = new URL(location.origin + '/api/plants');
      if(qs('q').value.trim()) url.searchParams.set('q', qs('q').value.trim());
      if(qs('inhOnly').checked) url.searchParams.set('inheems_only','true');
      if(qs('exInv').checked) url.searchParams.set('exclude_invasief','true');
      for(const v of cbValues('licht')) url.searchParams.append('licht', v);
      for(const v of cbValues('vocht')) url.searchParams.append('vocht', v);
      url.searchParams.set('limit','300');
      const r = await fetch(url);
      return r.json();
    }

    function htmlEscape(s){return (''+s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}

    function renderRows(items){
      const tb = document.querySelector('#tbl tbody');
      tb.innerHTML = items.map(r => `
        <tr>
          <td>${htmlEscape(r.naam||'')}</td>
          <td class="muted">${htmlEscape(r.wetenschappelijke_naam||'')}</td>
          <td><span class="pill">${htmlEscape(r.standplaats_licht||'')}</span></td>
          <td><span class="pill">${htmlEscape(r.vocht||'')}</span> ${r.ellenberg_f_min!=null?`<span class="badge">${r.ellenberg_f_min}–${r.ellenberg_f_max}</span>`:''}</td>
          <td>${r.ellenberg_f ?? ''}</td>
          <td>${htmlEscape(r.inheems||'')}</td>
          <td>${htmlEscape(r.invasief||'')}</td>
          <td>${r.hoogte ?? ''}</td>
          <td>${r.breedte ?? ''}</td>
          <td>${htmlEscape(r.winterhardheidszone||'')}</td>
        </tr>`).join('');
    }

    async function refresh(){
      const data = await fetchPlants();
      qs('count').textContent = data.count + ' resultaten (eerste ' + data.items.length + ')';
      renderRows(data.items);
    }

    // kaart-click → advies
    map.on('click', async (e)=>{
      if(marker) marker.remove();
      marker = L.marker(e.latlng).addTo(map);
      const url = new URL(location.origin + '/advies/geo');
      url.searchParams.set('lat', e.latlng.lat);
      url.searchParams.set('lon', e.latlng.lng);
      url.searchParams.set('zon', qs('zonSel').value);
      url.searchParams.set('inheems_only', qs('inhOnly').checked);
      url.searchParams.set('exclude_invasief', qs('exInv').checked);
      const r = await fetch(url); const j = await r.json();
      qs('ctx').textContent = `Regio: ${j.fgr||'onbekend'} · bodem: ${j.bodem||'?'} · droogte: ${j.droogte||'?'} (in ${j.elapsed_ms} ms)`;
      renderRows(j.advies||[]);
    });

    // events
    qs('q').addEventListener('input', ()=>{ clearTimeout(window.__t); window.__t=setTimeout(refresh, 220); });
    qs('inhOnly').addEventListener('change', refresh);
    qs('exInv').addEventListener('change', refresh);
    for(const el of document.querySelectorAll('input[name="licht"], input[name="vocht"]')) el.addEventListener('change', refresh);

    refresh();
  </script>
</body>
</html>
"""
    return Response(content=html, media_type="text/html; charset=utf-8")
