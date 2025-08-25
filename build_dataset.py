# build_dataset.py — PlantWijs Dataset Builder V25
# Fixes tov V23/V24:
# - Soortenlijst inlezen nóg robuuster (BOM-detectie, delimiter-detectie, header-detectie)
# - Exploderende merges voorkomen (aggregeren per taxon_norm + dedupe op wetenschappelijke_naam)
# - Strikte exportkolommen, geen ruisvelden
# - CSV als UTF‑8 met BOM (Excel toont “Löss” correct)
# - Behoudt: Ellenberg LONG (M→F), SL2020 inheems (1a/1b/2a), TreeEbb merge
from __future__ import annotations
import os, re, codecs
from typing import Optional, Dict, Tuple
import pandas as pd

DATA_DIR = "data"
OUT_DIR  = "out"
PATH_SOORTEN   = os.path.join(DATA_DIR, "verspreidingsatlas_planten.csv")
PATH_ELLEN     = os.path.join(DATA_DIR, "ellenberg.xlsx")
PATH_SL2020    = os.path.join(DATA_DIR, "standaardlijst2020.xlsx")
PATH_TREEEBB   = os.path.join(DATA_DIR, "treeebb_planten.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- utils

def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[’'`´]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _flatten_cols(cols) -> list[str]:
    out=[]
    for c in cols:
        c = str(c or "")
        c = c.replace("\n"," ").replace("\r"," ")
        c = re.sub(r"\s+", " ", c)
        c = c.strip().lower()
        c = c.replace(".", "_").replace("-", "_").replace(" ", "_").replace("/", "_")
        out.append(c)
    return out

# ---------- soortenlijst detectie & lezen

def _peek_bytes(path: str, n: int = 65536) -> bytes:
    with open(path, 'rb') as f:
        return f.read(n)

def _detect_encoding_and_sep(raw: bytes) -> Tuple[str, str, int]:
    # Encoding-kandidaten op basis van BOM
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        enc_candidates = ["utf-16", "utf-8-sig", "utf-8", "cp1252"]
    elif raw.startswith(codecs.BOM_UTF8):
        enc_candidates = ["utf-8-sig", "utf-8", "cp1252", "utf-16"]
    else:
        enc_candidates = ["utf-16", "utf-8-sig", "utf-8", "cp1252"]

    chosen_enc = None
    text = None
    for enc in enc_candidates:
        try:
            text = raw.decode(enc)
            chosen_enc = enc
            break
        except Exception:
            continue
    if text is None:
        # laatste redmiddel
        chosen_enc = "utf-8"
        text = raw.decode("utf-8", errors="replace")

    # Delimiter-detectie op basis van eerste niet-lege 5 regels
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = "\n".join(lines[:5])
    counts = {"\t": head.count("\t"), ";": head.count(";"), ",": head.count(",")}
    sep = max(counts, key=counts.get) if any(counts.values()) else "\t"

    # Header-rij index (0..4)
    header_idx = 0
    for i, ln in enumerate(lines[:5]):
        fields = [f.strip().lower() for f in ln.split(sep)]
        if any("wetenschappelijke" in f for f in fields) or any("nederlandse" in f for f in fields):
            header_idx = i
            break
    return chosen_enc, sep, header_idx


def _read_soorten() -> pd.DataFrame:
    """Lees soortenlijst uiterst robuust met detectie voor encoding/delimiter/header.
    Houdt alleen kernkolommen en forceert 1 rij per wetenschappelijke_naam.
    """
    raw = _peek_bytes(PATH_SOORTEN)
    enc, sep, hdr = _detect_encoding_and_sep(raw)
    try:
        df = pd.read_csv(PATH_SOORTEN, encoding=enc, sep=sep, engine="python", header=hdr, dtype=str, on_bad_lines="skip")
    except Exception:
        # fallback combinaties
        df = None
        for enc2 in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
            for sep2 in ("\t", ";", ",", " "):
                for hdr2 in (0,1):
                    try:
                        df = pd.read_csv(PATH_SOORTEN, encoding=enc2, sep=sep2, engine="python", header=hdr2, dtype=str, on_bad_lines="skip")
                        break
                    except Exception:
                        df = None
                        continue
                if df is not None: break
            if df is not None: break
        if df is None:
            raise SystemExit("[ERR] Kon soortenlijst niet robuust lezen. Check encoding/delimiter.")

    df.columns = _flatten_cols(df.columns)
    # rename naar kern
    ren: Dict[str,str] = {}
    for c in df.columns:
        if c.startswith("wetenschappelijke"):
            ren[c] = "wetenschappelijke_naam"
        elif c in ("nederlandse_naam","nederlandse","naam"):
            ren[c] = "naam"
        elif c == "familie" or ("familie" in c and "_" not in c):
            ren[c] = "familie"
    if ren:
        df = df.rename(columns=ren)
    # alleen relevante kolommen
    keep = [c for c in ("wetenschappelijke_naam","naam","familie") if c in df.columns]
    if not keep:
        raise SystemExit("[ERR] Kon kolommen 'Wetenschappelijke naam' / 'Naam' niet vinden in soortenlijst.")
    df = df[keep].copy()
    if "naam" not in df.columns:
        df["naam"] = ""
    if "familie" not in df.columns:
        df["familie"] = ""

    df = df.dropna(subset=["wetenschappelijke_naam"]).drop_duplicates(subset=["wetenschappelijke_naam"])  # 1 rij per soort
    df["taxon_norm"] = df["wetenschappelijke_naam"].map(_norm)

    delim_disp = 'TAB' if sep == '\t' else sep
    print(f"[INFO] CSV geladen enc='{enc}' delim={delim_disp!r} engine='python' header={hdr} — {len(df)} rijen, {df.shape[1]} kolommen")
    return df

# ---------- Ellenberg

def _read_xlsx_headings(path:str) -> list[str]:
    xl = pd.ExcelFile(path, engine="openpyxl")
    return xl.sheet_names


def _read_ellenberg() -> pd.DataFrame:
    """Lees Ellenberg uit LONG (voorkeur) of tabs; map 'M'→'F' (moisture); aggregeer naar 1 rij per taxon_norm."""
    sheets = _read_xlsx_headings(PATH_ELLEN)
    print("[ELLEN] Beschikbare sheets:", ", ".join(sheets))

    # ---- LONG pad (Tab-AveragePerDatabase-LONG)
    if "Tab-AveragePerDatabase-LONG" in sheets:
        for header in (0,1,2):
            try:
                d = pd.read_excel(PATH_ELLEN, sheet_name="Tab-AveragePerDatabase-LONG", header=header, engine="openpyxl")
                d.columns = _flatten_cols(d.columns)
                tax = next((c for c in ("taxon","species","name","scientific_name","sciname","taxon_name") if c in d.columns), None)
                if not tax:
                    continue
                d = d.rename(columns={tax:"taxon"})
                d["taxon_norm"] = d["taxon"].map(_norm)
                alias = {
                    "l": ["l", "light"],
                    "f": ["f", "m", "moisture"],  # M→F mapping
                    "t": ["t", "temperature"],
                    "n": ["n", "nutrients"],
                    "r": ["r", "reaction"],
                    "s": ["s", "salinity"],
                }
                def pick(cols: list[str], al: list[str], suf: str) -> Optional[str]:
                    for a in al:
                        for cand in (f"{a}_{suf}", f"{a}.{suf}", f"{a}{suf}", f"{a}_{suf}_average"):
                            if cand in cols: return cand
                    return None
                cols = list(d.columns)
                sel = {}
                for key, al in alias.items():
                    mn = pick(cols, al, "min"); mx = pick(cols, al, "max"); av = pick(cols, al, "average") or pick(cols, al, "avg") or pick(cols, al, "median")
                    if mn: sel[f"{key}_min"] = mn
                    if mx: sel[f"{key}_max"] = mx
                    if av: sel[f"{key}_avg"] = av
                if not sel:
                    continue
                e = d[["taxon_norm"] + sorted(set(sel.values()))].copy()
                for k,v in sel.items():
                    e[k] = pd.to_numeric(e[v], errors="coerce")
                agg = {}
                for c in e.columns:
                    if c == "taxon_norm": continue
                    if c.endswith("_min"): agg[c] = "min"
                    elif c.endswith("_max"): agg[c] = "max"
                    else: agg[c] = "mean"
                g = e.groupby("taxon_norm", as_index=False).agg(agg)
                out = pd.DataFrame({"taxon_norm": g["taxon_norm"]})
                for letter in ("l","f","t","n","r","s"):
                    if f"{letter}_avg" in g.columns: out[f"ellenberg_{letter}"] = g[f"{letter}_avg"]
                    if f"{letter}_min" in g.columns: out[f"ellenberg_{letter}_min"] = g[f"{letter}_min"]
                    if f"{letter}_max" in g.columns: out[f"ellenberg_{letter}_max"] = g[f"{letter}_max"]
                print("[ELLEN] LONG-sheet gebruikt (geaggregeerd)")
                return out
            except Exception:
                continue

    # ---- Fallback: losse tabs (LIGHT/MOISTURE/…)
    got = {}
    for tab, key in (("LIGHT","l"),("MOISTURE","f"),("TEMPERATURE","t"),("REACTION","r"),("NUTRIENTS","n"),("SALINITY","s")):
        if tab not in sheets: continue
        for header in (0,1,2):
            try:
                d = pd.read_excel(PATH_ELLEN, sheet_name=tab, header=header, engine="openpyxl")
                d.columns = _flatten_cols(d.columns)
                tax = next((c for c in ("taxon","species","name","scientific_name","sciname","taxon_name") if c in d.columns), None)
                if not tax: continue
                d = d.rename(columns={tax:"taxon"}); d["taxon_norm"] = d["taxon"].map(_norm)
                cand = {"min":None, "max":None, "avg":None}
                for c in d.columns:
                    if re.search(r"(^|_)min$", c): cand["min"]=c
                    if re.search(r"(^|_)max$", c): cand["max"]=c
                    if re.search(r"(average|avg|median)$", c): cand["avg"]=c
                if not (cand["min"] or cand["avg"]):
                    continue
                e = d[["taxon_norm"] + [v for v in cand.values() if v]].copy()
                for k in ("min","max","avg"):
                    if cand[k]: e[f"{key}_{k}"] = pd.to_numeric(e[cand[k]], errors="coerce")
                got.setdefault(key, []).append(e[["taxon_norm"] + [f"{key}_{k}" for k in ("min","max","avg") if f"{key}_{k}" in e.columns]])
                break
            except Exception:
                continue
    if got:
        res = None
        for key, parts in got.items():
            merged = pd.concat(parts, ignore_index=True)
            agg = {}
            for c in merged.columns:
                if c == "taxon_norm": continue
                if c.endswith("_min"): agg[c] = "min"
                elif c.endswith("_max"): agg[c] = "max"
                else: agg[c] = "mean"
            got[key] = merged.groupby("taxon_norm", as_index=False).agg(agg)
            res = got[key] if res is None else res.merge(got[key], on="taxon_norm", how="outer")
        final = pd.DataFrame({"taxon_norm": res["taxon_norm"]})
        for letter in ("l","f","t","n","r","s"):
            if f"{letter}_avg" in res.columns: final[f"ellenberg_{letter}"] = res[f"{letter}_avg"]
            if f"{letter}_min" in res.columns: final[f"ellenberg_{letter}_min"] = res[f"{letter}_min"]
            if f"{letter}_max" in res.columns: final[f"{letter}_max"] = res[f"{letter}_max"]
        print("[ELLEN] Gevonden via losse tabs (geaggregeerd)")
        return final

    raise SystemExit("[ERR] Ellenberg: geen geschikte structuur")

# ---------- SL2020 → inheems

def _read_sl2020_inheems() -> Optional[pd.DataFrame]:
    if not os.path.exists(PATH_SL2020):
        print("[WARN] SL2020 niet gevonden — sla inheems over")
        return None
    try:
        xl = pd.ExcelFile(PATH_SL2020, engine="openpyxl")
    except Exception as e:
        print("[WARN] SL2020 niet geladen:", e)
        return None
    target = None
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=0)
        df.columns = _flatten_cols(df.columns)
        if any("wetenschappelijke" in c for c in df.columns) and any("status" in c or "indigen" in c for c in df.columns):
            target = df; break
    if target is None:
        print("[WARN] SL2020 niet geladen: kolommen niet gevonden")
        return None
    wcol = next((c for c in target.columns if "wetenschappelijke" in c), None)
    scol = next((c for c in target.columns if "status" in c or "indigen" in c), None)
    m = target[[wcol, scol]].copy().rename(columns={wcol:"wetenschappelijke_naam", scol:"status"})
    m["taxon_norm"] = m["wetenschappelijke_naam"].map(_norm)
    def _inh(s):
        t = str(s or "").strip().lower()
        return "ja" if any(t.startswith(x) for x in ("1a","1b","2a")) else "nee"
    m["inheems"] = m["status"].map(_inh)
    return m[["taxon_norm","inheems"]].drop_duplicates("taxon_norm")

# ---------- TreeEbb (naam ↔ wetenschappelijke_naam), aggregeren

def _read_treeebb() -> Optional[pd.DataFrame]:
    if not os.path.exists(PATH_TREEEBB):
        print("[TREEEBB] bestand ontbreekt — overslaan")
        return None
    try:
        df = pd.read_csv(PATH_TREEEBB, sep=";", dtype=str, encoding="utf-8")
        df.columns = _flatten_cols(df.columns)
        need = {"naam","hoogte","breedte","winterhardheidszone","grondsoorten"}
        if not need.issubset(set(df.columns)):
            print("[TREEEEB] kolommen ontbreken — gevonden:", ", ".join(df.columns))
            return None
        df["taxon_norm"] = df["naam"].map(_norm)
        def to_num(x):
            try:
                return float(str(x).replace(",",".").strip())
            except Exception:
                return None
        if "hoogte" in df.columns: df["hoogte_num"] = df["hoogte"].map(to_num)
        if "breedte" in df.columns: df["breedte_num"] = df["breedte"].map(to_num)
        agg = {"hoogte_num":"max","breedte_num":"max","winterhardheidszone":"first","grondsoorten":"first"}
        g = df.groupby("taxon_norm", as_index=False).agg(agg).rename(columns={"hoogte_num":"hoogte","breedte_num":"breedte"})
        return g
    except Exception as e:
        print("[TREEEBB] fout bij lezen:", e)
        return None

# ---------- labels/klassen
F_EDGES = [(1,2.5,"zeer droog"), (2.5,4.5,"droog"), (4.5,6.5,"vochtig"), (6.5,8,"nat"), (8,12.5,"zeer nat")]
L_EDGES = [(0,3.5,"schaduw"), (3.5,6.5,"halfschaduw"), (6.5,12.5,"zon")]

def _labels_from_range(vmin: float|None, vmax: float|None, edges) -> str:
    if vmin is None and vmax is None: return ""
    a = vmin if vmin is not None else vmax
    b = vmax if vmax is not None else vmin
    if a is None: return ""
    a, b = float(a), float(b)
    labs=[]
    for lo,hi,name in edges:
        if b >= lo and a <= hi:
            labs.append(name)
    return " / ".join(labs)

# ---------- main

def main():
    sp = _read_soorten()                                  # ~6k rijen, 1 per soort
    sp_count = len(sp)

    ellen = _read_ellenberg()                             # geaggregeerd per taxon_norm
    sl    = _read_sl2020_inheems()                        # uniek per taxon_norm
    te    = _read_treeebb()                               # geaggregeerd per taxon_norm

    df = sp.merge(ellen, on="taxon_norm", how="left")
    if sl is not None:
        df = df.merge(sl, on="taxon_norm", how="left")
    if te is not None:
        df = df.merge(te, on="taxon_norm", how="left")

    # forceer 1 rij per wetenschappelijke_naam
    df = df.drop_duplicates(subset=["wetenschappelijke_naam"])  # anti-explosie

    # labels uit min/max
    df["standplaats_licht"] = df.apply(lambda r: _labels_from_range(r.get("ellenberg_l_min"), r.get("ellenberg_l_max"), L_EDGES), axis=1)
    df["vocht"]              = df.apply(lambda r: _labels_from_range(r.get("ellenberg_f_min"), r.get("ellenberg_f_max"), F_EDGES), axis=1)
    if "bodem" not in df.columns: df["bodem"] = ""
    if "inheems" not in df.columns: df["inheems"] = ""
    if "naam" not in df.columns: df["naam"] = ""

    # strikte kolomvolgorde
    EXACT_COLS = [
        "naam","wetenschappelijke_naam","inheems",
        "standplaats_licht","vocht","bodem",
        "ellenberg_l","ellenberg_l_min","ellenberg_l_max",
        "ellenberg_f","ellenberg_f_min","ellenberg_f_max",
        "ellenberg_t","ellenberg_t_min","ellenberg_t_max",
        "ellenberg_n","ellenberg_n_min","ellenberg_n_max",
        "ellenberg_r","ellenberg_r_min","ellenberg_r_max",
        "ellenberg_s","ellenberg_s_min","ellenberg_s_max",
        "hoogte","breedte","winterhardheidszone","grondsoorten",
    ]
    for c in EXACT_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[EXACT_COLS]

    # schrijf (UTF‑8 BOM i.v.m. “Löss” in Excel)
    out_csv = os.path.join(OUT_DIR, "plantwijs_full.csv")
    out_sc  = os.path.join(OUT_DIR, "plantwijs_full_semicolon.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_csv(out_sc, index=False, sep=";", encoding="utf-8-sig", decimal=",")

    print(f"[COUNT] soortenlijst: {sp_count} → output: {len(df)} rijen (1 per wetenschappelijke_naam)")
    print(f"[OK] Geschreven: {out_csv}")
    print(f"[OK] Geschreven: {out_sc}")

if __name__ == "__main__":
    main()
