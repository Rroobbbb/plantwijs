"""
PlantWijs API v0.11.2
- UI: kaart + resultaten-tabel
- Checkbox "Alleen inheemse planten" (standaard AAN)
- Dataset: accepteert extra kolommen met codes + *_uitleg (zeldzaamheid/rode_lijst/deelgroep)
- Endpoints: /, /advies/geo, /dataset/schema, /dataset/validate, /dataset/upload, /dataset/download, /dataset/reload

Starten:
  venv\Scripts\uvicorn api:app --reload --port 9000
"""
from __future__ import annotations
import os, io, math, time, json, re, urllib.parse
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pyproj import Transformer

# ----------------------- Config -----------------------
CSV_URL_ENV = os.getenv("PLANTWIJS_CSV_URL", "")
HEADERS = {"User-Agent": "PlantWijs/0.11.2"}

# Allowed tokens / schema
BODEM_TOKENS = {"zand","klei","leem","veen","löss","loess","loss",""}
DROOGTE_TOKENS = {"laag","middel","hoog",""}
LICHT_TOKENS = {"zon","half","halfschaduw","schaduw",""}
VOCHT_TOKENS = {"droog","fris","vochtig","nat",""}
INHEEMS_TOKENS = {"ja","nee",""}

PREF_COLS = [
    "naam","wetenschappelijke_naam","familie","inheems",
    "zeldzaamheid","zeldzaamheid_uitleg","rode_lijst","rode_lijst_uitleg","deelgroep","deelgroep_uitleg",
    "standplaats_licht","vocht","bodem","droogtetolerantie","klimaatzone",
    "stikstofbinder","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s","ecowaarde"
]

# ----------------------- Data loading -----------------------
_df: pd.DataFrame | None = None


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\W+", "_", c.strip().lower()).strip("_") for c in df.columns]
    # map back to app expected names (keeps whatever the dataset provides)
    rename = {
        "ndff_identity": "ndff_identity",
    }
    df = df.rename(columns=rename)
    return df


def _load_df() -> pd.DataFrame:
    global _df
    if CSV_URL_ENV:
        r = requests.get(CSV_URL_ENV, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    else:
        # Fallback mini dataset (als env var niet gezet is)
        csv_data = (
            "naam,wetenschappelijke_naam,inheems,bodem,droogtetolerantie,klimaatzone,zon_min,zon_max,ecowaarde,ellenberg_l,ellenberg_f,ellenberg_n,ellenberg_r,ellenberg_s,nectarwaarde,pollenwaarde,waarde_vogels,waarde_insecten,vocht,standplaats_licht,stikstofbinder,invasief,opmerking,familie,zeldzaamheid,zeldzaamheid_uitleg,rode_lijst,rode_lijst_uitleg,deelgroep,deelgroep_uitleg\n"
            "Zwarte els,Alnus glutinosa,ja,klei;veen,hoog,7a,1,2,8,6.5,7.5,5,6,0,, , , ,vochtig,zon;half,ja,nee,,Betulaceae,z,vrij zeldzaam, , ,S,Tweezaadlobbigen\n"
            "Duindoorn,Hippophae rhamnoides,ja,zand,laag,7a,2,2,7,8,5,4,7,0,,, , ,droog,zon,ja,nee,,Elaeagnaceae,a,algemeen, , ,S,Tweezaadlobbigen\n"
            "Gele lis,Iris pseudacorus,ja,klei;veen,laag,7a,0,2,6,6,8,6,6,0,,, , ,nat,zon;half,nee,nee,,Iridaceae,a,algemeen, , ,E,Eenzaadlobbigen\n"
        )
        df = pd.read_csv(io.StringIO(csv_data))
    df = _norm_cols(df)
    # ensure required columns exist
    for c in ["naam","wetenschappelijke_naam","inheems","zon_min","zon_max","klimaatzone","standplaats_licht","vocht"]:
        if c not in df.columns:
            df[c] = "" if c not in {"zon_min","zon_max"} else 0
    return df


# initial load
_df = _load_df()


# ----------------------- PDOK helpers -----------------------
PDOK_ROOT = (
    "https://service.pdok.nl/ez/fysischgeografischeregios/wfs/v1_0?service=WFS&version=2.0.0"
)
LAYER = "fysischgeografischeregios:fysischgeografischeregios"
FMT_JSON = "application/json;subtype=geojson"
TX_WGS84_RD = Transformer.from_crs(4326, 28992, always_xy=True)


def _wfs(url: str) -> list[dict]:
    r = requests.get(url, headers=HEADERS, timeout=8)
    if r.status_code != 200:
        return []
    if "json" not in r.headers.get("Content-Type", ""):
        return []
    return r.json().get("features", [])


def fgr_from_point(lat: float, lon: float) -> str | None:
    # 0. filter buiten NL
    x, y = TX_WGS84_RD.transform(lon, lat)
    if not (0 < x < 300_000 and 300_000 < y < 620_000):
        return None

    # 1. RD-bbox ±100 m
    b = 100
    x1, y1, x2, y2 = x - b, y - b, x + b, y + b
    url_rd = (
        f"{PDOK_ROOT}&request=GetFeature&typenames={LAYER}&outputFormat={FMT_JSON}&srsName=EPSG:28992"
        f"&bbox={x1},{y1},{x2},{y2}&count=1"
    )
    feats = _wfs(url_rd)
    if feats:
        return feats[0]["properties"].get("fgr")

    # 2. fallback WGS84 POINT
    cql = urllib.parse.quote_plus(f"INTERSECTS(geometry,POINT({lon} {lat}))")
    url_pt = (
        f"{PDOK_ROOT}&request=GetFeature&typenames={LAYER}&outputFormat={FMT_JSON}&srsName=EPSG:4326&cql_filter={cql}&count=1"
    )
    feats = _wfs(url_pt)
    if feats:
        return feats[0]["properties"].get("fgr")
    return None


def soil_from_fgr(fgr: str | None) -> tuple[str, str]:
    txt = (fgr or "").lower()
    if "duin" in txt or "zand" in txt:
        return "zand", "hoog"
    if "klei" in txt:
        return "klei", "middel"
    if "veen" in txt:
        return "veen", "laag"
    if "löss" in txt or "leem" in txt:
        return "leem", "laag"
    return "leem", "middel"


# ----------------------- Selection logic -----------------------

def select_advies(df: pd.DataFrame, zon: int, bodem: str, droogte: str, zone: str, inheems_only: bool) -> list[dict]:
    sel = df.copy()
    # inheems filter
    if inheems_only and "inheems" in sel.columns:
        sel = sel[sel["inheems"].astype(str).str.lower() == "ja"]
    # zon
    if "zon_min" in sel.columns and "zon_max" in sel.columns:
        sel = sel[(sel["zon_min"].astype(int) <= zon) & (sel["zon_max"].astype(int) >= zon)]
    # bodem
    if "bodem" in sel.columns:
        sel = sel[sel["bodem"].astype(str).str.contains(bodem, case=False, na=False) | (sel["bodem"].astype(str) == "")]
    # droogte
    if "droogtetolerantie" in sel.columns:
        sel = sel[sel["droogtetolerantie"].astype(str).str.contains(droogte, case=False, na=False) | (sel["droogtetolerantie"].astype(str) == "")]
    # klimaatzone
    if "klimaatzone" in sel.columns:
        sel = sel[sel["klimaatzone"].astype(str).str.contains(zone, case=False, na=False) | (sel["klimaatzone"].astype[str) == ""]]

    # sorteer op ecowaarde (indien aanwezig), anders op naam
    if "ecowaarde" in sel.columns:
        with pd.option_context('mode.use_inf_as_na', True):
            sel["_eco"] = pd.to_numeric(sel["ecowaarde"], errors="coerce").fillna(-1)
        sel = sel.sort_values(["_eco","naam"], ascending=[False, True])
    else:
        sel = sel.sort_values(["naam"], ascending=True)
    return sel.drop(columns=[c for c in sel.columns if c == "_eco"]).head(200).to_dict(orient="records")


# ----------------------- FastAPI app -----------------------
app = FastAPI(title="PlantWijs API 0.11.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST"],
    allow_headers=["*"]
)


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html lang=nl>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>PlantWijs</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,ui-sans-serif; margin:0;}
    header{padding:10px 14px; border-bottom:1px solid #e5e7eb; display:flex; align-items:center; gap:12px}
    #map{height:45vh;}
    .panel{padding:10px 14px}
    .controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
    .tbl{width:100%; border-collapse:collapse;}
    .tbl th,.tbl td{border-bottom:1px solid #eee; padding:6px 8px; text-align:left;}
    .muted{color:#6b7280; font-size:12px}
    .badge{display:inline-block;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;padding:2px 6px;border-radius:999px;font-size:12px}
    .chip{display:inline-block;background:#f3f4f6;color:#111827;border:1px solid #e5e7eb;padding:2px 6px;border-radius:6px;font-size:12px;margin-right:4px}
  </style>
</head>
<body>
  <header>
    <h3 style="margin:0">🌱 PlantWijs</h3>
    <label style="display:flex;align-items:center;gap:6px">
      <input id="chkInheems" type="checkbox" checked>
      Alleen inheemse planten
    </label>
    <div class="muted">Klik op de kaart om advies te krijgen.</div>
  </header>
  <div id="map"></div>
  <div class="panel">
    <div class="controls">
      <label>Zon: 
        <select id="selZon">
          <option value="0">schaduw</option>
          <option value="1">half</option>
          <option value="2" selected>zon</option>
        </select>
      </label>
      <span id="labBodem" class="chip">bodem: —</span>
      <span id="labDroogte" class="chip">droogte: —</span>
      <span id="labFGR" class="chip">FGR: —</span>
    </div>
    <table class="tbl" id="tbl"><thead></thead><tbody></tbody></table>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([52.09,5.12], 8);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution:'© OSM'}).addTo(map);
    let marker = null;

    const HEADS = [
      ['Naam','naam'], ['Wetenschappelijke naam','wetenschappelijke_naam'], ['Familie','familie'], ['Inheems','inheems'],
      ['Zeldzaamheid','zeldzaamheid'], ['↳ Uitleg','zeldzaamheid_uitleg'],
      ['Rode Lijst','rode_lijst'], ['↳ Uitleg','rode_lijst_uitleg'],
      ['Deelgroep','deelgroep'], ['↳ Uitleg','deelgroep_uitleg'],
      ['Licht','standplaats_licht'], ['Vocht','vocht'], ['Bodem','bodem'], ['Droogte','droogtetolerantie'], ['Zone','klimaatzone'],
      ['Ellenberg L','ellenberg_l'], ['F','ellenberg_f'], ['N','ellenberg_n'], ['R','ellenberg_r'], ['S','ellenberg_s'],
      ['Stikstofbinder','stikstofbinder'], ['Ecowaarde','ecowaarde']
    ];

    function htmlEscape(txt){return (''+txt).replace(/[&<>"']/g, s=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[s]||s));}

    function renderTable(rows){
      const thead = document.querySelector('#tbl thead');
      const tbody = document.querySelector('#tbl tbody');
      thead.innerHTML = '<tr>' + HEADS.map(([h])=>`<th>${h}</th>`).join('') + '</tr>';
      tbody.innerHTML = rows.map(r => {
        return '<tr>' + HEADS.map(([_,k])=>`<td>${htmlEscape(r[k]??'')}</td>`).join('') + '</tr>';
      }).join('');
    }

    async function fetchAdvies(lat, lon){
      const zon = document.getElementById('selZon').value;
      const inheemsOnly = document.getElementById('chkInheems').checked ? '1':'0';
      const url = `/advies/geo?lat=${lat}&lon=${lon}&zon=${zon}&inheems_only=${inheemsOnly}`;
      const r = await fetch(url);
      const j = await r.json();
      document.getElementById('labBodem').textContent = 'bodem: ' + j.bodem;
      document.getElementById('labDroogte').textContent = 'droogte: ' + j.droogte;
      document.getElementById('labFGR').textContent = 'FGR: ' + (j.fgr||'—');
      renderTable(j.advies || []);
    }

    map.on('click', e => {
      const {lat, lng} = e.latlng;
      if (marker) marker.remove();
      marker = L.marker([lat,lng]).addTo(map);
      fetchAdvies(lat, lng);
    });

    // start: pick center of NL
    fetchAdvies(52.09,5.12);
  </script>
</body>
</html>
    """
    return HTMLResponse(html)


@app.get("/dataset/schema")
def dataset_schema():
    df = _df
    return {
        "columns": list(df.columns),
        "preferred_columns": PREF_COLS,
        "allowed_tokens": {
            "bodem": sorted(BODEM_TOKENS),
            "droogtetolerantie": sorted(DROOGTE_TOKENS),
            "standplaats_licht": sorted(LICHT_TOKENS),
            "vocht": sorted(VOCHT_TOKENS),
            "inheems": sorted(INHEEMS_TOKENS),
        }
    }


@app.get("/dataset/validate")
def dataset_validate():
    df = _df
    issues: list[str] = []
    req = ["naam","wetenschappelijke_naam","inheems","zon_min","zon_max","klimaatzone"]
    for c in req:
        if c not in df.columns:
            issues.append(f"Kolom ontbreekt: {c}")
    # token checks
    def bad_tokens(series: pd.Series, allowed: set[str]) -> list[str]:
        bad = set()
        for v in series.fillna(""):
            toks = [t for t in re.split(r"[;|,\\/\s]+", str(v).lower()) if t]
            for t in toks:
                if t not in allowed:
                    bad.add(t)
        return sorted(bad)

    if "standplaats_licht" in df.columns:
        b = bad_tokens(df["standplaats_licht"], LICHT_TOKENS)
        if b: issues.append(f"Onbekende standplaats_licht tokens: {b}")
    if "vocht" in df.columns:
        b = bad_tokens(df["vocht"], VOCHT_TOKENS)
        if b: issues.append(f"Onbekende vocht tokens: {b}")
    if "inheems" in df.columns:
        b = bad_tokens(df["inheems"], INHEEMS_TOKENS)
        if b: issues.append(f"Onbekende inheems tokens: {b}")

    return {"ok": not issues, "issues": issues, "rows": len(df)}


@app.post("/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df = _norm_cols(df)
    global _df
    _df = df
    return {"ok": True, "rows": len(df)}


@app.get("/dataset/download", response_class=PlainTextResponse)
def dataset_download():
    buf = io.StringIO()
    _df.to_csv(buf, index=False)
    return buf.getvalue()


@app.post("/dataset/reload")
def dataset_reload():
    global _df
    _df = _load_df()
    return {"ok": True, "rows": len(_df)}


@app.get("/advies/geo")
def advies_geo(
    lat: float = Query(...),
    lon: float = Query(...),
    zon: int = Query(2, ge=0, le=2),
    inheems_only: bool = Query(True)
):
    t0 = time.time()
    fgr = fgr_from_point(lat, lon) or "Onbekend"
    bodem, droogte = soil_from_fgr(fgr)
    adv = select_advies(_df, zon, bodem, droogte, zone="7a", inheems_only=inheems_only)
    ms = int((time.time()-t0)*1000)
    return {"fgr": fgr, "bodem": bodem, "droogte": droogte, "advies": adv, "ms": ms}
