"""
PlantWijs Dataset Builder — V16 (LONG-sheet met L.min/L.max + fallback)
-----------------------------------------------------------------------
- Leest NDFF/Verspreidingsatlas CSV/TSV (autodetect UTF-16/UTF-8; \t ; , | spatie)
- Leest Ellenberg (XLSX), **voorkeur**: sheet 'Tab-AveragePerDatabase-LONG' met kolommen
  L.min L.max T.average T.median T.min T.max M.average M.median ... R/N/S idem.
  → We maken per taxon **min/max** en **ellenberg_x** (avg) voor L/F/T/N/R/S.
- Fallback: losse tabs (LIGHT/MOISTURE/...) of Tichý-sheet als LONG niet matcht.
- Leest SL2020 (XLSX) en bepaalt **inheems** (1a/1b/2a = ja; anders nee) per taxon.
- Zet min–max om naar **meervoudige labels** (licht/vocht/temperatuur/nutrienten/zuurgraad/zout).
- Schrijft **beide** outputs: `out/plantwijs_full.csv` én `out/plantwijs_full_semicolon.csv`.

Gebruik:
  venv\Scripts\python build_dataset.py
Optioneel forceren:
  set IN_ENC_FORCE=utf-16
  set IN_SEP_FORCE=\t
  set ELLEN_SHEET_FORCE=Tab-AveragePerDatabase-LONG
  set ELLEN_HEADER_ROW=1
"""
from __future__ import annotations
import os, sys, re, unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
pd.options.mode.copy_on_write = True

# --------------------------- Config ---------------------------
IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_XLSX_SL2020 = os.getenv("IN_XLSX_SL2020",  "data/standaardlijst2020.xlsx")
IN_CSV_OVR     = os.getenv("IN_CSV_OVR",      "data/manual_overrides.csv")
OUT_CSV        = os.getenv("OUT_CSV",         "out/plantwijs_full.csv")
OUT_CSV_SEMI   = os.getenv("OUT_CSV_SEMI",    "out/plantwijs_full_semicolon.csv")

# Forceringen (optioneel)
IN_ENC_FORCE   = os.getenv("IN_ENC_FORCE")  # bv. 'utf-16'
IN_SEP_FORCE   = os.getenv("IN_SEP_FORCE")  # bv. '\t' of ';'

APP_COLS = [
    "naam","wetenschappelijke_naam","inheems",
    "bodem","droogtetolerantie","klimaatzone","zon_min","zon_max",
    "ecowaarde",
    "ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s",
    "nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten",
    "vocht","standplaats_licht","temperatuur","voedselrijkdom","zuurgraad","zoutgehalte",
    "stikstofbinder","invasief","opmerking"
]

N_FIX_GENERA = {
    # Actinorhizal
    "alnus", "hippophae", "myrica", "morella",
    # Fabaceae (selection)
    "alhagi","anthyllis","cicer","cytisus","desmodium","genista","glycyrrhiza",
    "galega","gleditsia","laburnum","lathyrus","lotus","lupinus","medicago",
    "melilotus","ononis","ornithopus","oxytropis","prosopis","robinia",
    "spartium","spiraea","trifolium","vicia"
}

# --------------------------- Helpers ---------------------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)  # hybriden
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    m = re.match(r"^(\w+\s+\w+).*$", t)
    if m:
        t = m.group(1)
    return t.lower()


def _peek_bom(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read(4)


def _good_species_header(cols: List[str]) -> bool:
    cols_norm = [re.sub(r"[^0-9a-z_]", "", c.lower().strip()) for c in cols]
    expect = {"wetenschappelijke_naam","nederlandse_naam","familie","soortnummer","soortcode","ndffidentity"}
    hits = sum(c in expect for c in cols_norm)
    unnamed = sum(c.startswith("unnamed") or len(c) <= 2 for c in cols_norm)
    return hits >= 1 and unnamed < max(2, len(cols)//2)


def _read_table_best(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError("Bestand ontbreekt: " + path)

    bom = _peek_bom(path)
    if IN_ENC_FORCE:
        enc_order = [IN_ENC_FORCE]
    elif bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        enc_order = ["utf-16","utf-16-le","utf-16-be","utf-8","cp1252"]
    elif bom.startswith(b"\xef\xbb\xbf"):
        enc_order = ["utf-8-sig","utf-8","cp1252","utf-16"]
    else:
        enc_order = ["utf-8","cp1252","utf-16","utf-16-le","utf-16-be"]

    sep_order = [IN_SEP_FORCE] if IN_SEP_FORCE else ["\t",";",",","|"," "]

    last_err: Optional[Exception] = None
    for enc in enc_order:
        for sep in sep_order:
            try:
                eng = "python" if sep == " " else "c"
                df = pd.read_csv(path, sep=sep, engine=eng, dtype=str,
                                 encoding=enc, encoding_errors="ignore",
                                 on_bad_lines="skip")
                sep_disp = sep.replace("\t","\\t")
                if not _good_species_header([str(c) for c in df.columns]):
                    print(f"[SKIP] enc='{enc}' delim='{sep_disp}': header oogt niet correct → volgende poging")
                    continue
                print(f"[INFO] CSV geladen enc='{enc}' delim='{sep_disp}' engine='{eng}' — {len(df)} rijen, {df.shape[1]} kolommen")
                print("[INFO] Aangetroffen kolommen:", ", ".join([str(c) for c in list(df.columns)[:50]]))
                return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError("Kon CSV niet lezen met fallback-readers: " + str(last_err))


# --------------------------- NDFF soortenlijst ---------------------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_table_best(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_sci = next((c for c in df.columns if c in {"wetenschappelijke_naam","soort","taxon","species","binomium"}), None)
    col_nl  = next((c for c in df.columns if c in {"nederlandse_naam","nederlandse_soortnaam","naam","soortnaam_nl"}), None)
    col_family = next((c for c in df.columns if c in {"familie","family"}), None)
    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")
    df = df.rename(columns={col_sci: "wetenschappelijke_naam"})
    if col_nl: df = df.rename(columns={col_nl: "naam"})
    if col_family: df = df.rename(columns={col_family: "_familie"})
    if "naam" not in df.columns:
        df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns:
        df["_familie"] = ""
    return df[["naam","wetenschappelijke_naam","_familie"]]


# --------------------------- Ellenberg parsing ---------------------------

def _flatten_cols(cols) -> List[str]:
    out: List[str] = []
    if getattr(cols, "levels", None) is not None:  # MultiIndex
        for tup in cols:
            parts = [str(x) for x in tup if x is not None and str(x).strip() and not str(x).startswith("Unnamed")]
            name = "_".join(parts)
            name = re.sub("[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
            out.append(name)
    else:
        for c in cols:
            name = re.sub("[^0-9A-Za-z_]+", "_", str(c).strip().lower()).strip("_")
            out.append(name)
    return out


def _ellen_from_long_sheet(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    # Voorkeur: 'Tab-AveragePerDatabase-LONG' met kolommen L.min/L.max/T.average/... etc.
    force = os.getenv("ELLEN_SHEET_FORCE")
    name = force or next((s for s in xl.sheet_names if re.search(r"average.*long", s, re.I)), None)
    if not name:
        return None
    headers = [int(os.getenv("ELLEN_HEADER_ROW", x)) for x in (0,1,2,3)]
    for header in headers:
        try:
            df = xl.parse(name, header=header)
            df.columns = _flatten_cols(df.columns)
            # Normaliseer punten naar underscores: 'l.min' → 'l_min'
            df.columns = [c.replace('.', '_') for c in df.columns]
            # Zoek taxon kolom
            cand_tax = next((c for c in df.columns if c in {"taxon","taxon_name","species","species_name","name","scientific_name","latin_name"}), None)
            if not cand_tax:
                # probeer genus + species
                cand_genus = next((c for c in df.columns if c in {"genus","geslacht"}), None)
                cand_sp    = next((c for c in df.columns if c in {"species","soort","epithet","species_epithet"}), None)
                if cand_genus and cand_sp:
                    df["taxon"] = (df[cand_genus].astype(str).str.strip() + " " + df[cand_sp].astype(str).str.strip()).str.strip()
                else:
                    raise ValueError("LONG: taxon-kolom niet gevonden")
            else:
                df["taxon"] = df[cand_tax]
            # Pak per metric min/max/avg indien aanwezig
            def pick(prefix: str) -> Tuple[Optional[str],Optional[str],Optional[str]]:
                cmin = next((c for c in df.columns if c == f"{prefix}_min"), None)
                cmax = next((c for c in df.columns if c == f"{prefix}_max"), None)
                cavg = next((c for c in df.columns if c == f"{prefix}_average" or c == f"{prefix}_avg" or c == f"{prefix}_mean"), None)
                return cmin, cmax, cavg
            pairs = {
                "l": pick("l"),
                "f": pick("m"),  # MOISTURE is M.* in LONG
                "t": pick("t"),
                "n": pick("n"),
                "r": pick("r"),
                "s": pick("s"),
            }
            if not any(v[0] or v[1] or v[2] for v in pairs.values()):
                raise ValueError("LONG: geen min/max/avg kolommen gevonden")
            core = pd.DataFrame({"taxon": df["taxon"]})
            core["taxon_norm"] = core["taxon"].map(_norm)
            for k,(cmin,cmax,cavg) in pairs.items():
                if cmin and cmin in df.columns:
                    core[f"{k}_min"] = pd.to_numeric(df[cmin], errors="coerce")
                if cmax and cmax in df.columns:
                    core[f"{k}_max"] = pd.to_numeric(df[cmax], errors="coerce")
                # Avg voor app-kolom ellenberg_k
                if cavg and cavg in df.columns:
                    core[f"ellenberg_{k}"] = pd.to_numeric(df[cavg], errors="coerce")
                else:
                    # Geen average? neem halveweg min/max indien beide aanwezig
                    if f"{k}_min" in core.columns and f"{k}_max" in core.columns:
                        core[f"ellenberg_{k}"] = (core[f"{k}_min"] + core[f"{k}_max"]) / 2.0
            print(f"[ELLEN] LONG-sheet '{name}' header={header} gebruikt")
            return core
        except Exception as e:
            print("[DBG] LONG header=", header, "→", e)
            continue
    return None


def _ellen_from_tabs(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    # Fallback: losse tabs LIGHT/MOISTURE/TEMPERATURE/REACTION/NUTRIENTS/SALINITY
    want = {"l":"LIGHT","f":"MOISTURE","t":"TEMPERATURE","n":"NUTRIENTS","r":"REACTION","s":"SALINITY"}
    def parse_metric(sheet: str, key: str) -> Optional[pd.DataFrame]:
        for header in (0,1,2):
            try:
                df = xl.parse(sheet, header=header)
                df.columns = _flatten_cols(df.columns)
                tax = next((c for c in df.columns if re.fullmatch(r"taxon|taxon_name|species|species_name|name|scientific_name|latin_name", c)), None)
                c_min = next((c for c in df.columns if c.endswith("min") and not c.endswith("_min_min")), None)
                c_max = next((c for c in df.columns if c.endswith("max") and not c.endswith("_max_max")), None)
                c_avg = next((c for c in df.columns if "avg" in c or "average" in c), None)
                if not (tax and c_min and c_max):
                    raise ValueError("tab mist vereiste kolommen")
                d = pd.DataFrame({
                    "taxon": df[tax],
                    f"{key}_min": pd.to_numeric(df[c_min], errors="coerce"),
                    f"{key}_max": pd.to_numeric(df[c_max], errors="coerce"),
                })
                if c_avg:
                    d[f"ellenberg_{key}"] = pd.to_numeric(df[c_avg], errors="coerce")
                d["taxon_norm"] = d["taxon"].map(_norm)
                return d
            except Exception:
                continue
        return None
    pieces: List[pd.DataFrame] = []
    for k, sheet in want.items():
        if sheet in xl.sheet_names:
            p = parse_metric(sheet, k)
            if p is not None:
                pieces.append(p)
    if not pieces:
        return None
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="taxon_norm", how="outer")
    print("[ELLEN] Gevonden via losse tabs (fallback)")
    return out


def _ellen_from_tichy_sheet(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    # Laatste redmiddel: 'Tab-IVs-Tichy-et-al2023' / 'Tab-OriginalNamesValues'
    name = next((s for s in xl.sheet_names if re.search(r"tichy|original", s, re.I)), None)
    if not name:
        return None
    for header in (0,1,2):
        try:
            df = xl.parse(name, header=header)
            df.columns = _flatten_cols(df.columns)
            tax = next((c for c in df.columns if re.fullmatch(r"taxon|taxon_name|species|species_name|name|scientific_name|latin_name", c)), None)
            if not tax:
                raise ValueError("taxon-kolom niet gevonden")
            # Probeer min/max/avg kolommen per metric te vinden
            def grab(prefix: str) -> Tuple[Optional[str],Optional[str],Optional[str]]:
                cmin = next((c for c in df.columns if re.search(rf"^{prefix}.*min", c)), None)
                cmax = next((c for c in df.columns if re.search(rf"^{prefix}.*max", c)), None)
                cavg = next((c for c in df.columns if re.search(rf"^{prefix}.*(avg|aver|average)", c)), None)
                return cmin, cmax, cavg
            pairs = {"l":grab("light"), "f":grab("moisture"), "t":grab("temperature"), "n":grab("nutrients"), "r":grab("reaction"), "s":grab("salinity")}
            core = pd.DataFrame({"taxon": df[tax]})
            core["taxon_norm"] = core["taxon"].map(_norm)
            ok=False
            for k,(cmin,cmax,cavg) in pairs.items():
                if cmin and cmax:
                    core[f"{k}_min"] = pd.to_numeric(df[cmin], errors="coerce")
                    core[f"{k}_max"] = pd.to_numeric(df[cmax], errors="coerce")
                    ok=True
                if cavg:
                    core[f"ellenberg_{k}"] = pd.to_numeric(df[cavg], errors="coerce")
            if not ok:
                raise ValueError("geen min/max kolommen gevonden")
            print(f"[ELLEN] Tichy sheet '{name}' header={header}")
            return core
        except Exception:
            continue
    return None


def load_ellenberg_ranges(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError("Ellenberg-bestand ontbreekt: " + path)
    xl = pd.ExcelFile(path)
    print("[ELLEN] Beschikbare sheets:", ", ".join(xl.sheet_names))
    for fn in (_ellen_from_long_sheet, _ellen_from_tabs, _ellen_from_tichy_sheet):
        df = fn(xl)
        if df is not None:
            # Zorg dat alle ellenberg_* bestaan indien avg/min/max aanwezig is
            for k in ("l","f","t","n","r","s"):
                if f"ellenberg_{k}" not in df.columns:
                    if f"{k}_min" in df.columns and f"{k}_max" in df.columns:
                        df[f"ellenberg_{k}"] = (df.get(f"{k}_min") + df.get(f"{k}_max")) / 2.0
            print("[ELLEN] Structuur kolommen:", ", ".join([c for c in df.columns if c.endswith("_min") or c.endswith("_max")] )[:120])
            return df
    raise RuntimeError("Ellenberg: kon geen geschikte structuur parsen (LONG/tabs/Tichy)")


# --------------------------- SL2020 ---------------------------

def load_sl2020(path_xlsx: str) -> pd.DataFrame:
    if not os.path.exists(path_xlsx):
        raise RuntimeError("SL2020 Excel ontbreekt: " + path_xlsx)
    xl = pd.ExcelFile(path_xlsx, engine="openpyxl")
    def _flat(cols) -> List[str]:
        return _flatten_cols(cols)
    def _normkey(name: str) -> str:
        return re.sub(r"[^a-z]", "", str(name).lower())
    best = None
    for sheet in xl.sheet_names:
        for header in (0,1,2,3):
            try:
                df_try = xl.parse(sheet, header=header)
                df_try.columns = _flat(df_try.columns)
                cols_norm = [_normkey(c) for c in df_try.columns]
                has_sci = any(c in {"wetenschappelijkenaam","wetnaam","scientificname","naamwetenschappelijk","wetenschappelijke_naam","wetenschappelijkenaamvande_soort"} for c in cols_norm)
                score = 5 if has_sci else 0
                score += sum("status" in c and "nsr" in c for c in cols_norm)
                score += sum("indigen" in c for c in cols_norm)
                if best is None or score > best[0]:
                    best = (score, sheet, header, df_try)
            except Exception:
                continue
    if best is None:
        raise RuntimeError("SL2020: kon geen geschikte sheet/header vinden")
    _, sheet, header, sl = best
    print(f"[SL2020] Gekozen sheet='{sheet}' header={header}")
    sl.columns = _flat(sl.columns)
    cols_norm = { re.sub(r"[^a-z]","", c): c for c in sl.columns }
    def pick(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in cols_norm: return cols_norm[k]
        return None
    col_sci = pick(["wetenschappelijkenaam","wetnaam","scientificname","naamwetenschappelijk","wetenschappelijke_naam","wetenschappelijkenaamvande_soort"]) or \
              pick(["taxon","name","species","speciesname","latinname"])
    if not col_sci:
        raise RuntimeError("SL2020: kon kolom met wetenschappelijke naam niet vinden")
    col_stat = pick(["statusnsr","nsrstatus","status_nsr"])  # optioneel
    col_ind  = pick(["indigeniteit","indigen","indigeneit"])  # optioneel

    base = pd.DataFrame({
        "wetenschappelijke_naam": sl[col_sci].astype(str),
        "status_nsr": sl[col_stat].astype(str) if col_stat else "",
        "indigeniteit": sl[col_ind].astype(str) if col_ind else "",
    })
    base["taxon_norm"] = base["wetenschappelijke_naam"].map(_norm)

    def map_inheems(status_nsr: str, indigen: str) -> str:
        s = str(status_nsr or "").strip().lower()
        i = str(indigen or "").strip().lower()
        if i.startswith("i") or s.startswith("1a") or s.startswith("1b") or s.startswith("2a"):
            return "ja"
        return "nee"

    base["inheems"] = [map_inheems(s, i) for s, i in zip(base["status_nsr"], base["indigeniteit"]) ]
    base = base.drop_duplicates(subset=["taxon_norm"])  # 1 record per taxon_norm
    return base[["taxon_norm","inheems","status_nsr","indigeniteit"]]


# --------------------------- Labels uit min/max ---------------------------

def _labels_from_range(vmin: Optional[float], vmax: Optional[float], edges: List[float], labels: List[str]) -> str:
    try:
        a = float(vmin) if vmin is not None else None
        b = float(vmax) if vmax is not None else None
    except Exception:
        a = b = None
    if a is None and b is None:
        return ""
    if a is None: a = b
    if b is None: b = a
    if a > b: a, b = b, a
    def idx(x: float) -> int:
        for i, e in enumerate(edges):
            if x <= e: return i
        return len(edges)
    i1 = idx(a); i2 = idx(b)
    i1 = max(0, min(i1, len(labels)-1)); i2 = max(0, min(i2, len(labels)-1))
    return " / ".join(labels[i1:i2+1])

EDGES_L = [3.5, 6.0]
LAB_L   = ["schaduw","halfschaduw","zon"]

EDGES_F = [2.5, 4.5, 6.5, 8.0]
LAB_F   = ["zeer droog","droog","vochtig","nat","zeer nat"]

EDGES_T = [3.0, 4.5, 6.5, 8.0]
LAB_T   = ["zeer koel","koel","gematigd","warm","zeer warm"]

EDGES_N = [2.5, 4.5, 6.5, 8.0]
LAB_N   = ["zeer voedselarm","voedselarm","matig","voedselrijk","zeer voedselrijk"]

EDGES_R = [3.5, 5.5, 7.0, 8.0]
LAB_R   = ["zuur","zwak zuur","neutraal","zwak basisch","basisch"]

EDGES_S = [0.5, 1.5, 3.5, 6.0]
LAB_S   = ["zoet","zwak brak","brak","zout","zeer zout"]


# --------------------------- Build ---------------------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # NDFF soortenlijst
    sp = load_species_csv(IN_CSV_SPECIES).copy()
    sp.loc[:, "taxon_norm"] = sp["wetenschappelijke_naam"].map(_norm)
    # Uniek per soort
    sp = sp.drop_duplicates(subset=["wetenschappelijke_naam"])  # voorkom 32k records

    # Ellenberg
    el = load_ellenberg_ranges(IN_XLSX_ELLEN)

    # SL2020
    try:
        sl = load_sl2020(IN_XLSX_SL2020)
    except Exception as e:
        print("[WARN] SL2020 niet geladen:", e)
        sl = pd.DataFrame(columns=["taxon_norm","inheems","status_nsr","indigeniteit"])  # leeg

    merged = sp.merge(el, how="left", on="taxon_norm")
    if not sl.empty:
        merged = merged.merge(sl, how="left", on="taxon_norm")

    # Collapse naar 1 rij per wetenschappelijke_naam (soms meerdere records per taxon_norm)
    def any_ja(s: pd.Series) -> str:
        return "ja" if (s.astype(str).str.lower() == "ja").any() else "nee"

    agg: Dict[str, Any] = {
        "naam": "first",
        "wetenschappelijke_naam": "first",
        "_familie": "first",
        "inheems": any_ja,
        # averages (ellenberg_x)
        "ellenberg_l": "mean", "ellenberg_f": "mean", "ellenberg_t": "mean", "ellenberg_n": "mean", "ellenberg_r": "mean", "ellenberg_s": "mean",
        # ranges
        "l_min": "min", "l_max": "max",
        "f_min": "min", "f_max": "max",
        "t_min": "min", "t_max": "max",
        "n_min": "min", "n_max": "max",
        "r_min": "min", "r_max": "max",
        "s_min": "min", "s_max": "max",
    }
    use_cols = {k:v for k,v in agg.items() if k in merged.columns}
    collapsed = merged.groupby("wetenschappelijke_naam", as_index=False).agg(use_cols)

    out = pd.DataFrame(columns=APP_COLS)
    out["naam"] = collapsed["naam"].fillna(collapsed["wetenschappelijke_naam"])  # fallback
    out["wetenschappelijke_naam"] = collapsed["wetenschappelijke_naam"]
    out["inheems"] = collapsed.get("inheems", "nee")

    # map ellenberg averages naar app-kolommen
    for src,dst in [("ellenberg_l","ellenberg_l"),("ellenberg_f","ellenberg_f"),("ellenberg_t","ellenberg_tTEMP"),("ellenberg_n","ellenberg_n"),("ellenberg_r","ellenberg_r"),("ellenberg_s","ellenberg_s")]:
        if src in collapsed.columns:
            out[dst if dst!="ellenberg_tTEMP" else "ellenberg_t"] = pd.to_numeric(collapsed[src], errors="coerce").round(2)

    # labels uit min/max
    out["standplaats_licht"] = collapsed.apply(lambda r: _labels_from_range(r.get("l_min"), r.get("l_max"), EDGES_L, LAB_L), axis=1)
    out["vocht"]              = collapsed.apply(lambda r: _labels_from_range(r.get("f_min"), r.get("f_max"), EDGES_F, LAB_F), axis=1)
    out["temperatuur"]        = collapsed.apply(lambda r: _labels_from_range(r.get("t_min"), r.get("t_max"), EDGES_T, LAB_T), axis=1)
    out["voedselrijkdom"]     = collapsed.apply(lambda r: _labels_from_range(r.get("n_min"), r.get("n_max"), EDGES_N, LAB_N), axis=1)
    out["zuurgraad"]          = collapsed.apply(lambda r: _labels_from_range(r.get("r_min"), r.get("r_max"), EDGES_R, LAB_R), axis=1)
    out["zoutgehalte"]        = collapsed.apply(lambda r: _labels_from_range(r.get("s_min"), r.get("s_max"), EDGES_S, LAB_S), axis=1)

    # zon_min/max afgeleid uit standplaats_licht
    def zon_minmax_from_labels(lbl: str) -> Tuple[int,int]:
        lbl = str(lbl)
        has_schaduw = "schaduw" in lbl and "half" not in lbl and "zon" not in lbl
        has_half    = "halfschaduw" in lbl
        has_zon     = (" " + lbl).endswith(" zon") or lbl == "zon" or "/ zon" in lbl or " zon /" in lbl
        if has_schaduw and not has_half and not has_zon: return 0,0
        if has_zon and not has_half and not has_schaduw: return 2,2
        if has_half and not has_schaduw and not has_zon: return 1,1
        if has_schaduw and has_half and not has_zon: return 0,1
        if has_half and has_zon and not has_schaduw: return 1,2
        return 0,2
    zminmax = out["standplaats_licht"].map(zon_minmax_from_labels)
    out["zon_min"] = [a for a,b in zminmax]
    out["zon_max"] = [b for a,b in zminmax]

    # overige
    out["klimaatzone"] = ""  # nog niet betrouwbaar — later via kaart
    out["ecowaarde"]   = ""
    out["bodem"] = ""
    out["droogtetolerantie"] = ""

    # familie meenemen voor stikstofbinding
    fam_map = merged.drop_duplicates("wetenschappelijke_naam").set_index("wetenschappelijke_naam").get("_familie", pd.Series(index=[]))
    out["_familie"] = out["wetenschappelijke_naam"].map(fam_map)
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie", "")).lower()
        sci = str(row.get("wetenschappelijke_naam", ""))
        genus = sci.split(" ")[0].lower() if sci else ""
        if "fabaceae" in fam or genus in N_FIX_GENERA:
            return "ja"
        return "nee"
    out["stikstofbinder"] = out.apply(is_nfix, axis=1)
    out.drop(columns=["_familie"], inplace=True)

    out["invasief"] = "nee"
    out["nectarwaarde"] = ""
    out["pollenwaarde"] = ""
    out["waarde_vogels"] = ""
    out["waarde_insecten"] = ""
    out["opmerking"] = ""

    # Handmatige overrides (optioneel)
    if os.path.exists(IN_CSV_OVR) and os.path.getsize(IN_CSV_OVR) > 5:
        ovr = _read_table_best(IN_CSV_OVR)
        ovr.columns = [c.strip().lower().replace(" ", "_") for c in ovr.columns]
        if "wetenschappelijke_naam" in ovr.columns:
            ovr["_key"] = ovr["wetenschappelijke_naam"].map(_norm)
        elif "taxon" in ovr.columns:
            ovr["_key"] = ovr["taxon"].map(_norm)
        else:
            ovr = None
            print("[WARN] overrides mist kolom 'wetenschappelijke_naam' of 'taxon' — overslaan")
        if ovr is not None:
            out["_key"] = out["wetenschappelijke_naam"].map(_norm)
            ovr = ovr.drop_duplicates(subset=["_key"], keep="last")
            out = out.merge(
                ovr.drop(columns=[c for c in ["wetenschappelijke_naam","taxon"] if c in ovr.columns]),
                how="left", left_on="_key", right_on="_key", suffixes=("", "__ovr")
            )
            for c in APP_COLS:
                oc = c + "__ovr"
                if oc in out.columns:
                    out[c] = out[oc].where(out[oc].notna() & (out[oc] != ""), out[c])
            out = out[[c for c in out.columns if not c.endswith("__ovr") and c != "_key"]]
    elif os.path.exists(IN_CSV_OVR):
        print("[WARN] Overrides-bestand is leeg — overslaan")

    # Zorg dat alle kolommen bestaan en in juiste volgorde staan
    for c in APP_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[APP_COLS]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    out.to_csv(OUT_CSV_SEMI, index=False, encoding="utf-8", sep=";", decimal=",")

    print(f"[OK] Geschreven: {OUT_CSV}  ({len(out)} rijen)")
    print(f"[OK] Geschreven: {OUT_CSV_SEMI}  ({len(out)} rijen)")

    return out


if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as e:
        print("[ERR]", repr(e))
        sys.exit(2)
