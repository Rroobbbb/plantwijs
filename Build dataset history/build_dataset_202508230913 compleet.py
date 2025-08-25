"""
PlantWijs Dataset Builder — V22 (robust)
----------------------------------------
Fixes:
- Lost recurring error: "Error tokenizing data … Expected 1 fields …" → robuuste CSV-lezer die BOM/encodering en scheidingsteken (TAB/;/,/spatie) detecteert met fallback-engines.
- Vult **inheems** uit SL2020 (1a/1b/2a of indigeniteit die met 'i' begint → ja).
- Merge **TreeEbb** (data/treeebb_planten.csv; sep=';') op *naam* (TreeEbb) ↔ *wetenschappelijke_naam* (soortenlijst) na normalisatie → **hoogte, breedte, winterhardheidszone, grondsoorten**.
- Leest **Ellenberg** uit LONG of losse tabs; levert **ellenberg_l/f/t/n/r/s** + labels over klassen (meervoud mogelijk) + min/max indien beschikbaar.
- Schrijft **twee** uitvoerbestanden: comma-CSV en semicolon-CSV.
- Selftest aanwezig: `python build_dataset.py --selftest`.

Benodigde bestanden (standaardpaden):
- data/verspreidingsatlas_planten.csv
- data/ellenberg.xlsx
- data/standaardlijst2020.xlsx
- data/treeebb_planten.csv (sep=';')
- optioneel: data/invasieve_soorten.csv (sep=';')
"""
from __future__ import annotations
import os, re, sys, io, unicodedata, tempfile
from typing import Any, List, Optional, Tuple

import pandas as pd
pd.options.mode.copy_on_write = True

# --------------------------- Bestanden ---------------------------
IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_XLSX_SL2020 = os.getenv("IN_XLSX_SL2020",  "data/standaardlijst2020.xlsx")
IN_CSV_TREEEBB = os.getenv("IN_CSV_TREEEBB",  "data/treeebb_planten.csv")
IN_CSV_INVAS   = os.getenv("IN_CSV_INVAS",    "data/invasieve_soorten.csv")

OUT_CSV        = os.getenv("OUT_CSV",         "out/plantwijs_full.csv")
OUT_CSV_SEMI   = os.getenv("OUT_CSV_SEMI",    "out/plantwijs_full_semicolon.csv")

# --------------------------- App kolommen ---------------------------
APP_COLS = [
    "naam","wetenschappelijke_naam","inheems",
    "bodem","droogtetolerantie","klimaatzone","zon_min","zon_max",
    "ecowaarde",
    # Ellenberg (gemiddelden)
    "ellenberg_l","ellenberg_f","ellenberg_t","ellenberg_n","ellenberg_r","ellenberg_s",
    # range (indien beschikbaar)
    "ellenberg_l_min","ellenberg_l_max",
    "ellenberg_f_min","ellenberg_f_max",
    "ellenberg_t_min","ellenberg_t_max",
    "ellenberg_n_min","ellenberg_n_max",
    "ellenberg_r_min","ellenberg_r_max",
    "ellenberg_s_min","ellenberg_s_max",
    # labels over meerdere klasses
    "standplaats_licht","vocht","temperatuur","voedselrijkdom","zuurgraad","zoutgehalte",
    # fauna/overig
    "nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten",
    "stikstofbinder","invasief",
    # TreeEbb
    "hoogte","breedte","winterhardheidszone","grondsoorten",
    "opmerking"
]

# stikstofbinders (Fabaceae + actinorhizaal)
N_FIX_GENERA = {
    "alnus","hippophae","myrica","morella",
    "alhagi","anthyllis","cicer","cytisus","desmodium","genista","glycyrrhiza",
    "galega","gleditsia","laburnum","lathyrus","lotus","lupinus","medicago",
    "melilotus","ononis","ornithopus","oxytropis","prosopis","robinia",
    "spartium","spiraea","trifolium","vicia"
}

# --------------------------- Helpers ---------------------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    m = re.match(r"^([A-Za-z-]+)\s+([A-Za-z-]+)", t)
    if m:
        t = f"{m.group(1)} {m.group(2)}"
    return t.lower()

# --------------------------- Robuuste CSV lezer ---------------------------

def _peek_bom(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read(4)

def _good_species_header(cols: List[str]) -> bool:
    c = [re.sub(r"[^0-9a-z_]", "", str(x).lower()) for x in cols]
    exp = {"wetenschappelijke_naam","nederlandse_naam","naam","familie","soortnummer","soortcode","ndffidentity"}
    return any(x in exp for x in c)

def _read_table_best(path: str) -> pd.DataFrame:
    encs = ["utf-8","cp1252","utf-16"]
    bom = _peek_bom(path)
    if bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        encs = ["utf-16","utf-8","cp1252"]
    seps = ["\t",";",",","|"," "]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                eng = "python" if sep == " " else "c"
                df = pd.read_csv(path, sep=sep, engine=eng, dtype=str,
                                 encoding=enc, encoding_errors="ignore",
                                 on_bad_lines="skip")
                if not _good_species_header(list(df.columns)):
                    continue
                disp = 'TAB' if sep == '\t' else ('SPACE' if sep == ' ' else sep)
                print(f"[INFO] CSV geladen enc='{enc}' delim='{disp}' engine='{eng}' — {len(df)} rijen, {df.shape[1]} kolommen")
                print("[INFO] Aangetroffen kolommen:", ", ".join([str(c) for c in list(df.columns)[:50]]))
                return df
            except Exception as e:
                last = e
                continue
    raise RuntimeError("Kon CSV niet lezen met fallback-readers: " + str(last))

# --------------------------- Loaders ---------------------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_table_best(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_sci = next((c for c in df.columns if c in {"wetenschappelijke_naam","soort","taxon","species","binomium"}), None)
    col_nl  = next((c for c in df.columns if c in {"nederlandse_naam","naam","nederlandse_soortnaam"}), None)
    col_fam = next((c for c in df.columns if c in {"familie","family"}), None)
    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")
    df = df.rename(columns={col_sci: "wetenschappelijke_naam"})
    if col_nl:  df = df.rename(columns={col_nl:  "naam"})
    if col_fam: df = df.rename(columns={col_fam: "_familie"})
    if "naam" not in df.columns:
        df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns:
        df["_familie"] = ""
    df["taxon_norm"] = df["wetenschappelijke_naam"].map(_norm)
    return df[["naam","wetenschappelijke_naam","_familie","taxon_norm"]]

# ---- Ellenberg uit LONG sheet (Tab-AveragePerDatabase-LONG) ----

def _flatten_cols(cols) -> List[str]:
    out: List[str] = []
    if getattr(cols, "levels", None) is not None:
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

def _ellen_from_long(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    name = next((s for s in xl.sheet_names if re.search(r"average.*long", s, re.I)), None)
    if not name:
        return None
    for header in (0,1,2,3):
        try:
            df = xl.parse(name, header=header)
            df.columns = _flatten_cols(df.columns)
            tax = next((c for c in df.columns if re.fullmatch(r"taxon|species|name|taxon_name|scientific_name|latin_name", c)), None)
            if not tax:
                continue
            # zoek min/max/avg per letter
            def pick(prefix: str):
                cmin = next((c for c in df.columns if re.fullmatch(rf"{prefix}_(min|min_.*)", c)), None)
                cmax = next((c for c in df.columns if re.fullmatch(rf"{prefix}_(max|max_.*)", c)), None)
                cavg = next((c for c in df.columns if re.fullmatch(rf"{prefix}_(avg|average|mean)", c)), None)
                return cmin, cmax, cavg
            pairs = {"l":pick("l"),"f":pick("f"),"t":pick("t"),"n":pick("n"),"r":pick("r"),"s":pick("s")}
            core = pd.DataFrame({"taxon": df[tax]})
            core["taxon_norm"] = core["taxon"].map(_norm)
            ok = False
            for k,(cmin,cmax,cavg) in pairs.items():
                if cmin and cmax:
                    core[f"{k}_min"] = pd.to_numeric(df[cmin], errors="coerce")
                    core[f"{k}_max"] = pd.to_numeric(df[cmax], errors="coerce")
                    ok = True
                if cavg:
                    core[f"ellenberg_{k}"] = pd.to_numeric(df[cavg], errors="coerce")
            if ok:
                print(f"[ELLEN] LONG-sheet '{name}' header={header} gebruikt")
                return core
        except Exception:
            continue
    return None

# ---- Ellenberg uit losse tabs (LIGHT/MOISTURE/...) ----

def _ellen_from_tabs(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    want = {"l":"LIGHT","f":"MOISTURE","t":"TEMPERATURE","n":"NUTRIENTS","r":"REACTION","s":"SALINITY"}
    parts: List[pd.DataFrame] = []
    for k, sheet in want.items():
        if sheet not in xl.sheet_names:
            continue
        for header in (0,1,2):
            try:
                df = xl.parse(sheet, header=header)
                df.columns = _flatten_cols(df.columns)
                tax = next((c for c in df.columns if re.fullmatch(r"taxon|species|name|taxon_name|scientific_name|latin_name", c)), None)
                cmin = next((c for c in df.columns if c.endswith("_min") or c.endswith("min")), None)
                cmax = next((c for c in df.columns if c.endswith("_max") or c.endswith("max")), None)
                cavg = next((c for c in df.columns if ("avg" in c or "average" in c or "mean" in c)), None)
                if not (tax and cmin and cmax):
                    continue
                d = pd.DataFrame({
                    "taxon": df[tax],
                    f"{k}_min": pd.to_numeric(df[cmin], errors="coerce"),
                    f"{k}_max": pd.to_numeric(df[cmax], errors="coerce"),
                })
                if cavg:
                    d[f"ellenberg_{k}"] = pd.to_numeric(df[cavg], errors="coerce")
                d["taxon_norm"] = d["taxon"].map(_norm)
                parts.append(d)
                break
            except Exception:
                continue
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="taxon_norm", how="outer")
    print("[ELLEN] Gevonden via losse tabs (fallback)")
    return out

# ---- Tichy/Original fallback ----

def _ellen_from_tichy(xl: pd.ExcelFile) -> Optional[pd.DataFrame]:
    name = next((s for s in xl.sheet_names if re.search(r"tichy|original", s, re.I)), None)
    if not name:
        return None
    for header in (0,1,2):
        try:
            df = xl.parse(name, header=header)
            df.columns = _flatten_cols(df.columns)
            tax = next((c for c in df.columns if re.fullmatch(r"taxon|species|name|taxon_name|scientific_name|latin_name", c)), None)
            if not tax:
                continue
            core = pd.DataFrame({"taxon": df[tax]})
            core["taxon_norm"] = core["taxon"].map(_norm)
            def grab(prefix: str):
                cmin = next((c for c in df.columns if re.search(rf"^{prefix}.*min", c)), None)
                cmax = next((c for c in df.columns if re.search(rf"^{prefix}.*max", c)), None)
                cavg = next((c for c in df.columns if re.search(rf"^{prefix}.*(avg|average|mean)", c)), None)
                return cmin, cmax, cavg
            pairs = {"l":grab("light"),"f":grab("moisture"),"t":grab("temperature"),"n":grab("nutrients"),"r":grab("reaction"),"s":grab("salinity")}
            ok = False
            for k,(cmin,cmax,cavg) in pairs.items():
                if cmin and cmax:
                    core[f"{k}_min"] = pd.to_numeric(df[cmin], errors="coerce")
                    core[f"{k}_max"] = pd.to_numeric(df[cmax], errors="coerce")
                    ok = True
                if cavg:
                    core[f"ellenberg_{k}"] = pd.to_numeric(df[cavg], errors="coerce")
            if ok:
                print(f"[ELLEN] Tichy sheet '{name}' header={header}")
                return core
        except Exception:
            continue
    return None


def load_ellenberg(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError("Ellenberg-bestand ontbreekt: " + path)
    xl = pd.ExcelFile(path)
    print("[ELLEN] Beschikbare sheets:", ", ".join(xl.sheet_names))
    for fn in (_ellen_from_long, _ellen_from_tabs, _ellen_from_tichy):
        df = fn(xl)
        if df is not None:
            for k in ("l","f","t","n","r","s"):
                if f"ellenberg_{k}" not in df.columns and f"{k}_min" in df.columns and f"{k}_max" in df.columns:
                    df[f"ellenberg_{k}"] = (df[f"{k}_min"] + df[f"{k}_max"]) / 2.0
            return df
    raise RuntimeError("Ellenberg: kon geen geschikte structuur parsen (LONG/tabs/Tichy)")

# ---- SL2020 (Excel) → inheems ----

def load_sl2020(path_xlsx: str) -> pd.DataFrame:
    if not os.path.exists(path_xlsx):
        raise RuntimeError("SL2020 Excel ontbreekt: " + path_xlsx)
    xl = pd.ExcelFile(path_xlsx, engine="openpyxl")
    best = None
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
            df.columns = [re.sub(r"[^a-z]", "", str(c).lower()) for c in df.columns]
            col_sci = next((c for c in df.columns if "wetenschappelijkenaam" in c or "scientificname" in c or c=="wetnaam"), None)
            col_stat = next((c for c in df.columns if "statusnsr" in c or c=="status"), None)
            col_ind  = next((c for c in df.columns if "indigen" in c), None)
            if col_sci:
                best = (sheet, col_sci, col_stat, col_ind, df)
                break
        except Exception:
            continue
    if not best:
        raise RuntimeError("SL2020: kon kolom met wetenschappelijke naam niet vinden")
    sheet, col_sci, col_stat, col_ind, df = best
    base = pd.DataFrame({
        "wetenschappelijke_naam": df[col_sci].astype(str),
        "status_nsr": df[col_stat].astype(str) if col_stat else "",
        "indigeniteit": df[col_ind].astype(str) if col_ind else "",
    })
    base["taxon_norm"] = base["wetenschappelijke_naam"].map(_norm)
    def map_inheems(s: str, i: str) -> str:
        s = (s or '').strip().lower()
        i = (i or '').strip().lower()
        return "ja" if i.startswith('i') or s.startswith('1a') or s.startswith('1b') or s.startswith('2a') else "nee"
    base["inheems"] = [map_inheems(s, i) for s, i in zip(base["status_nsr"], base["indigeniteit"])]
    base = base.drop_duplicates("taxon_norm")
    print(f"[SL2020] sheet='{sheet}' geladen — {len(base)} rijen")
    return base[["taxon_norm","inheems","status_nsr","indigeniteit"]]

# ---- TreeEbb (CSV ;)

def load_treeebb(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print("[TREEEBB] Bestand ontbreekt — overslaan")
        return pd.DataFrame(columns=["taxon_norm","hoogte","breedte","winterhardheidszone","grondsoorten"])
    df = pd.read_csv(path, sep=';', dtype=str, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'naam' not in df.columns:
        raise RuntimeError("treeebb_planten.csv mist kolom 'naam'")
    def _to_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip().replace(',', '.')
        s = re.sub(r"[^0-9.]+", "", s)
        try:
            return float(s) if s != '' else None
        except Exception:
            return None
    df['taxon_norm'] = df['naam'].map(_norm)
    for c in ('hoogte','breedte'):
        if c in df.columns:
            df[c] = df[c].map(_to_float)
    keep = [k for k in ['taxon_norm','hoogte','breedte','winterhardheidszone','grondsoorten'] if k in df.columns]
    print(f"[TREEEBB] geladen: {len(df)} rijen — kolommen: {', '.join(keep)}")
    return df[keep]

# ---- invasieve lijst (optioneel)

def load_invasief(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["taxon_norm","invasief"])
    df = pd.read_csv(path, sep=';', dtype=str, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
    df.columns = [c.strip().lower() for c in df.columns]
    key = 'wetenschappelijke_naam' if 'wetenschappelijke_naam' in df.columns else ('taxon' if 'taxon' in df.columns else None)
    if not key:
        raise RuntimeError("invasieve_soorten.csv mist kolom 'wetenschappelijke_naam' of 'taxon'")
    df['taxon_norm'] = df[key].map(_norm)
    df['invasief'] = df.get('invasief','ja').fillna('ja').str.strip().str.lower().map(lambda v: 'ja' if v in {'ja','yes','y','true','1'} else 'nee')
    return df[['taxon_norm','invasief']].drop_duplicates('taxon_norm')

# --------------------------- Labeling (meerdere klasses) ---------------------------

def _labels_from_range(vmin: Optional[float], vmax: Optional[float], edges: List[float], labels: List[str]) -> str:
    if vmin is None and vmax is None:
        return ""
    try:
        a = float(vmin) if vmin is not None else float(vmax)
        b = float(vmax) if vmax is not None else float(vmin)
    except Exception:
        return ""
    if a > b:
        a, b = b, a
    def idx(x: float) -> int:
        for i, e in enumerate(edges):
            if x <= e:
                return i
        return len(edges)
    i1, i2 = idx(a), idx(b)
    i1 = max(0, min(i1, len(labels)-1))
    i2 = max(0, min(i2, len(labels)-1))
    return " / ".join(labels[i1:i2+1])

EDGES_L = [3.5, 6.0]; LAB_L = ["schaduw","halfschaduw","zon"]
EDGES_F = [2.0, 4.0, 6.0, 8.0]; LAB_F = ["zeer droog","droog","vochtig","nat","zeer nat"]
EDGES_T = [3.0, 4.5, 6.5, 8.0]; LAB_T = ["zeer koel","koel","gematigd","warm","zeer warm"]
EDGES_N = [2.5, 4.5, 6.5, 8.0]; LAB_N = ["zeer voedselarm","voedselarm","matig","voedselrijk","zeer voedselrijk"]
EDGES_R = [3.5, 5.5, 7.0, 8.0]; LAB_R = ["zuur","zwak zuur","neutraal","zwak basisch","basisch"]
EDGES_S = [0.5, 1.5, 3.5, 6.0]; LAB_S = ["zoet","zwak brak","brak","zout","zeer zout"]

# --------------------------- Build ---------------------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    sp  = load_species_csv(IN_CSV_SPECIES)
    el  = load_ellenberg(IN_XLSX_ELLEN)
    try:
        sl = load_sl2020(IN_XLSX_SL2020)
    except Exception as e:
        print("[WARN] SL2020 niet geladen:", e)
        sl = pd.DataFrame(columns=["taxon_norm","inheems"])  # leeg → default 'nee'
    tr  = load_treeebb(IN_CSV_TREEEBB)
    inv = load_invasief(IN_CSV_INVAS)

    # Merge op taxon_norm
    merged = sp.merge(el,  how="left", on="taxon_norm")
    if not sl.empty:  merged = merged.merge(sl,  how="left", on="taxon_norm")
    if not tr.empty:  merged = merged.merge(tr,  how="left", on="taxon_norm")
    if not inv.empty: merged = merged.merge(inv, how="left", on="taxon_norm")

    # Aggregeer naar 1 rij per wetenschappelijke_naam
    agg = {
        "naam":"first","wetenschappelijke_naam":"first","taxon_norm":"first","_familie":"first",
        # avg
        "ellenberg_l":"mean","ellenberg_f":"mean","ellenberg_t":"mean","ellenberg_n":"mean","ellenberg_r":"mean","ellenberg_s":"mean",
        # ranges
        "l_min":"min","l_max":"max","f_min":"min","f_max":"max","t_min":"min","t_max":"max",
        "n_min":"min","n_max":"max","r_min":"min","r_max":"max","s_min":"min","s_max":"max",
        # TreeEbb
        "hoogte":"max","breedte":"max","winterhardheidszone":"first","grondsoorten":"first",
        # ja/nee
        "inheems":  lambda s: "ja" if (s.astype(str).str.lower()=="ja").any() else "nee",
        "invasief": lambda s: "ja" if (s.astype(str).str.lower()=="ja").any() else "",
    }
    use_cols = {k:v for k,v in agg.items() if k in merged.columns}
    collapsed = merged.groupby("wetenschappelijke_naam", as_index=False).agg(use_cols)

    out = pd.DataFrame(columns=APP_COLS)
    out["naam"] = collapsed["naam"].fillna(collapsed["wetenschappelijke_naam"])  # fallback
    out["wetenschappelijke_naam"] = collapsed["wetenschappelijke_naam"]
    out["inheems"] = collapsed.get("inheems", "nee")

    # Ellenberg avg
    for k in ("l","f","t","n","r","s"):
        src = f"ellenberg_{k}"
        if src in collapsed.columns:
            out[src] = pd.to_numeric(collapsed[src], errors="coerce").round(2)

    # min/max overzetten
    for k in ("l","f","t","n","r","s"):
        if f"{k}_min" in collapsed.columns:
            out[f"ellenberg_{k}_min"] = pd.to_numeric(collapsed[f"{k}_min"], errors="coerce")
        if f"{k}_max" in collapsed.columns:
            out[f"ellenberg_{k}_max"] = pd.to_numeric(collapsed[f"{k}_max"], errors="coerce")

    # labels uit ranges
    out["standplaats_licht"] = collapsed.apply(lambda r: _labels_from_range(r.get("l_min"), r.get("l_max"), EDGES_L, LAB_L), axis=1)
    out["vocht"]              = collapsed.apply(lambda r: _labels_from_range(r.get("f_min"), r.get("f_max"), EDGES_F, LAB_F), axis=1)
    out["temperatuur"]        = collapsed.apply(lambda r: _labels_from_range(r.get("t_min"), r.get("t_max"), EDGES_T, LAB_T), axis=1)
    out["voedselrijkdom"]     = collapsed.apply(lambda r: _labels_from_range(r.get("n_min"), r.get("n_max"), EDGES_N, LAB_N), axis=1)
    out["zuurgraad"]          = collapsed.apply(lambda r: _labels_from_range(r.get("r_min"), r.get("r_max"), EDGES_R, LAB_R), axis=1)
    out["zoutgehalte"]        = collapsed.apply(lambda r: _labels_from_range(r.get("s_min"), r.get("s_max"), EDGES_S, LAB_S), axis=1)

    # zon-min/max heuristiek uit label
    def zon_minmax(lbl: str) -> Tuple[int,int]:
        txt = str(lbl)
        has_s = "schaduw" in txt and "half" not in txt and "zon" not in txt
        has_h = "halfschaduw" in txt
        has_z = txt.endswith(" zon") or txt == "zon" or "/ zon" in txt or " zon /" in txt
        if has_s and not has_h and not has_z: return 0,0
        if has_z and not has_h and not has_s: return 2,2
        if has_h and not has_s and not has_z: return 1,1
        if has_s and has_h and not has_z:     return 0,1
        if has_h and has_z and not has_s:     return 1,2
        return 0,2
    z = out["standplaats_licht"].map(zon_minmax)
    out["zon_min"] = [a for a,b in z]
    out["zon_max"] = [b for a,b in z]

    # familie→stikstofbinder
    fam_map = collapsed.set_index("wetenschappelijke_naam").get("_familie", pd.Series(index=[]))
    out["_familie"] = out["wetenschappelijke_naam"].map(fam_map)
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie",""))
        sci = str(row.get("wetenschappelijke_naam",""))
        genus = sci.split(" ")[0].lower() if sci else ""
        return "ja" if ("fabaceae" in fam.lower() or genus in N_FIX_GENERA) else "nee"
    out["stikstofbinder"] = out.apply(is_nfix, axis=1)
    out.drop(columns=["_familie"], inplace=True)

    # invasief (alleen gevuld als lijst aanwezig is)
    out["invasief"] = collapsed.get("invasief", "")

    # TreeEbb velden
    for c in ("hoogte","breedte","winterhardheidszone","grondsoorten"):
        if c in collapsed.columns:
            out[c] = collapsed[c]

    # placeholders
    out["bodem"] = ""
    out["droogtetolerantie"] = ""
    out["klimaatzone"] = ""
    out["ecowaarde"] = ""
    out["nectarwaarde"] = ""
    out["pollenwaarde"] = ""
    out["waarde_vogels"] = ""
    out["waarde_insecten"] = ""
    out["opmerking"] = ""

    # ensure kolommen & volgorde
    for c in APP_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[APP_COLS]

    # schrijven
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8')
    out.to_csv(OUT_CSV_SEMI, index=False, encoding='utf-8', sep=';', decimal=',')

    matched_treeebb = collapsed[[c for c in ("hoogte","breedte","winterhardheidszone","grondsoorten") if c in collapsed.columns]].notna().any(axis=1).sum() if not collapsed.empty else 0
    print(f"[MERGE] TreeEbb velden gevuld voor ~{matched_treeebb} taxa")
    print(f"[OK] Geschreven: {OUT_CSV}  ({len(out)} rijen)")
    print(f"[OK] Geschreven: {OUT_CSV_SEMI}  ({len(out)} rijen)")
    return out

# --------------------------- Selftest ---------------------------

def _selftest() -> None:
    print("[SELFTEST] start")
    tmp = tempfile.mkdtemp()
    # maak een UTF-16 TSV soortenlijst
    species_path = os.path.join(tmp, 'soorten.tsv')
    tsv = "\t".join(["Wetenschappelijke naam","Nederlandse naam","Familie"]) + "\n"
    tsv += "Crataegus monogyna\tEenstijlige meidoorn\tRosaceae\n"
    with open(species_path, 'wb') as f:
        f.write("\ufeff".encode('utf-16le'))  # BOM voor utf-16
        f.write(tsv.encode('utf-16le'))
    # mini Ellenberg (tabs fallback)
    ellen_xlsx = os.path.join(tmp, 'ellenberg.xlsx')
    with pd.ExcelWriter(ellen_xlsx) as xw:
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[4.0],"Max":[8.0],"Average":[6.0]}).to_excel(xw, sheet_name='LIGHT', index=False)
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[3.0],"Max":[7.5],"Average":[5.0]}).to_excel(xw, sheet_name='MOISTURE', index=False)
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[4.0],"Max":[7.0],"Average":[5.5]}).to_excel(xw, sheet_name='TEMPERATURE', index=False)
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[3.5],"Max":[7.0],"Average":[5.0]}).to_excel(xw, sheet_name='REACTION', index=False)
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[3.0],"Max":[7.0],"Average":[5.0]}).to_excel(xw, sheet_name='NUTRIENTS', index=False)
        pd.DataFrame({"Taxon":["Crataegus monogyna"],"Min":[0.5],"Max":[2.0],"Average":[1.0]}).to_excel(xw, sheet_name='SALINITY', index=False)
    # mini SL2020
    sl_path = os.path.join(tmp, 'sl2020.xlsx')
    pd.DataFrame({"Wetenschappelijke naam":["Crataegus monogyna"],"Status NSR":["1b"],"Indigeniteit":["i"]}).to_excel(sl_path, index=False)
    # mini TreeEbb
    treeebb_path = os.path.join(tmp, 'treeebb_planten.csv')
    with open(treeebb_path, 'w', encoding='utf-8') as f:
        f.write("naam;url;hoogte;breedte;winterhardheidszone;grondsoorten\n")
        f.write("Crataegus monogyna;http://x;10;5;6A;Zand | Leem\n")

    # run build met env override
    os.environ['IN_CSV_SPECIES'] = species_path
    os.environ['IN_XLSX_ELLEN'] = ellen_xlsx
    os.environ['IN_XLSX_SL2020'] = sl_path
    os.environ['IN_CSV_TREEEBB'] = treeebb_path
    os.environ['OUT_CSV'] = os.path.join(tmp, 'out.csv')
    os.environ['OUT_CSV_SEMI'] = os.path.join(tmp, 'out_semicolon.csv')

    out = build_dataset()
    assert len(out) == 1, "Selftest: verwacht 1 rij"
    row = out.iloc[0]
    assert row['inheems'] == 'ja', "Selftest: inheems moet 'ja' zijn via SL2020 1b"
    assert 'halfschaduw' in str(row['standplaats_licht']) or 'zon' in str(row['standplaats_licht']), "Selftest: lichtlabels uit min/max"
    assert float(row['hoogte']) == 10.0 and float(row['breedte']) == 5.0, "Selftest: TreeEbb merge (hoogte/breedte)"
    print("[SELFTEST] OK")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    if '--selftest' in sys.argv:
        _selftest()
        sys.exit(0)
    try:
        build_dataset()
    except Exception as e:
        print("[ERR]", e)
        sys.exit(2)
