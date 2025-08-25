"""
Build a comprehensive PlantWijs dataset for the web/app from open sources.

Inputs (you download once):
- data/verspreidingsatlas_planten.csv   # Soortenlijst (CSV/TSV; kan UTF-8/UTF-16; delimiter variabel)
- data/ellenberg.xlsx                   # Ellenberg-achtige indicatoren (Zenodo DOI 10.5281/zenodo.7427088)
- data/manual_overrides.csv             # (optioneel) handmatige correcties/aanvullingen

Outputs:
- out/plantwijs_full.csv                # komma-gescheiden (voor app/Sheets)
- out/plantwijs_full_semicolon.csv      # puntkomma-gescheiden (handig voor NL-Excel)
- out/unmatched_sample.csv              # eerste 200 taxa zonder Ellenberg-match (debug)

Usage (PowerShell/CMD):
  venv\Scripts\python build_dataset.py
"""
from __future__ import annotations
import os, sys, re, unicodedata
from typing import Any, List, Optional, Tuple
from difflib import SequenceMatcher

try:
    import pandas as pd
except ModuleNotFoundError:
    print("[ERR] 'pandas' ontbreekt. Gebruik de venv!", file=sys.stderr)
    print("Fix:", file=sys.stderr)
    print("  1) cd C:\\PlantWijs", file=sys.stderr)
    print("  2) venv\\Scripts\\python -m pip install pandas openpyxl", file=sys.stderr)
    print("  3) venv\\Scripts\\python build_dataset.py", file=sys.stderr)
    print("Interpreter:", sys.executable, file=sys.stderr)
    sys.exit(1)

# --------------------------- Config ---------------------------
IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_CSV_OVR     = os.getenv("IN_CSV_OVR",      "data/manual_overrides.csv")

OUT_CSV         = os.getenv("OUT_CSV",         "out/plantwijs_full.csv")
OUT_CSV_SEMI    = os.getenv("OUT_CSV_SEMI",    "out/plantwijs_full_semicolon.csv")
OUT_CSV_UNMATCH = os.getenv("OUT_CSV_UNMATCH", "out/unmatched_sample.csv")

APP_COLS = [
    "naam","wetenschappelijke_naam","inheems",
    "bodem","droogtetolerantie","klimaatzone","zon_min","zon_max",
    "ecowaarde",
    "ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s",
    "nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten",
    "vocht","standplaats_licht",
    "stikstofbinder","invasief",
    "opmerking"
]

N_FIX_GENERA = {
    # actinorhizal
    "alnus","hippophae","myrica","morella",
    # fabaceae (selectie)
    "alhagi","anthyllis","cicer","cytisus","desmodium","genista","glycyrrhiza",
    "galega","gleditsia","laburnum","lathyrus","lotus","lupinus","medicago",
    "melilotus","ononis","ornithopus","oxytropis","prosopis","robinia",
    "spartium","spiraea","trifolium","vicia"
}

VOCHT_TOKENS = {"droog","fris","vochtig","nat"}
LICHT_TOKENS = {"zon","half","halfschaduw","schaduw"}

# --------------------------- Robust CSV helpers ---------------------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    # vermijd backrefs in broncode i.v.m. canvas → lambda
    t = re.sub(r"^(\w+\s+\w+).*$", lambda m: m.group(1), t)
    return t.lower()

def _detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" in head:
        return "utf-16"
    return "utf-8"

def _read_csv_best(path: str) -> pd.DataFrame:
    """Probeer diverse encodings/delimiters/engines; val terug op autodetect/whitespace.
       Zet on_bad_lines='skip' om tokenizing errors te omzeilen.
    """
    enc0 = _detect_encoding(path)
    encs: List[str] = []
    def add_enc(e: str):
        if e not in encs:
            encs.append(e)
    add_enc(enc0); add_enc("utf-8-sig"); add_enc("utf-8"); add_enc("latin1")

    delims = ["\t", ";", ",", "|"]

    best_df: Optional[pd.DataFrame] = None
    best = {"cols": -1, "enc": None, "delim": None, "engine": None}
    last_err: Optional[Exception] = None

    for enc in encs:
        for d in delims:
            engines = ("python",) if enc.startswith("utf-16") else ("c", "python")
            for eng in engines:
                try:
                    kw = dict(sep=d, dtype=str, encoding=enc)
                    if eng == "python":
                        kw.update(dict(engine="python", on_bad_lines="skip"))
                    df = pd.read_csv(path, **kw)
                    ncols = df.shape[1]
                    if ncols >= 8:
                        print(f"[INFO] CSV geladen enc='{enc}' delim='{d}' engine='{eng}' — {len(df)} rijen, {ncols} kolommen")
                        return df
                    if ncols > best["cols"]:
                        best_df = df; best.update({"cols": ncols, "enc": enc, "delim": d, "engine": eng})
                except Exception as e:
                    last_err = e
                    continue
    # autodetect via python-engine
    for enc in encs:
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=enc, on_bad_lines="skip")
            ncols = df.shape[1]
            if ncols >= 8:
                print(f"[INFO] CSV autodetected enc='{enc}' — {len(df)} rijen, {ncols} kolommen")
                return df
            if ncols > best["cols"]:
                best_df = df; best.update({"cols": ncols, "enc": enc, "delim": "auto", "engine": "python"})
        except Exception as e:
            last_err = e
            continue
    # whitespace fallback
    for enc in encs:
        try:
            df = pd.read_csv(path, delim_whitespace=True, engine="python", dtype=str, encoding=enc, on_bad_lines="skip")
            ncols = df.shape[1]
            if ncols >= 8:
                print(f"[INFO] CSV whitespace enc='{enc}' — {len(df)} rijen, {ncols} kolommen")
                return df
            if ncols > best["cols"]:
                best_df = df; best.update({"cols": ncols, "enc": enc, "delim": "whitespace", "engine": "python"})
        except Exception as e:
            last_err = e
            continue
    if best_df is not None:
        print(f"[WARN] Beste parse had {best['cols']} kolom(men) enc='{best['enc']}' delim='{best['delim']}' engine='{best['engine']}'.")
        return best_df
    raise last_err if last_err else RuntimeError("Kon CSV niet parsen (encoding/delimiter)")

# --------------------------- Load species ---------------------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_csv_best(path)
    print("[INFO] Aangetroffen kolommen:", ", ".join(map(str, df.columns))[:2000])
    df.columns = [re.sub(r"\W+", "_", c.strip().lower()).strip("_") for c in df.columns]

    def pick_first(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_sci = pick_first([
        "wetenschappelijke_naam","soortcode","soort","taxon","species","binomium",
        "soortnaam_wetenschappelijk","wetenschappelijke_soortnaam","latijnse_naam",
        "scientific_name","latin_name","taxonnaam"
    ])
    col_nl  = pick_first(["nederlandse_naam","nederlandse_soortnaam","naam","soortnaam_nl","nl_naam","dutch_name","nederlandse"])
    col_family = pick_first(["familie","family"]) or next((c for c in df.columns if re.search(r"familie|family", c)), None)
    col_status = pick_first(["status","herkomst","inheems","inheems_of_niet","soortstatus","oorspronkelijk"])  # optioneel

    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")

    rename = {col_sci: "wetenschappelijke_naam"}
    if col_nl: rename[col_nl] = "naam"
    if col_family: rename[col_family] = "_familie"
    if col_status: rename[col_status] = "_status"
    df = df.rename(columns=rename)

    if "_status" in df.columns:
        mask_inh = df["_status"].astype(str).str.contains("inheems", case=False, na=False)
        df = df.loc[mask_inh].copy()

    df["wetenschappelijke_naam"] = df["wetenschappelijke_naam"].astype(str)
    if "naam" not in df.columns:
        df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns:
        df["_familie"] = ""

    return df[["naam","wetenschappelijke_naam","_familie"]]

# --------------------------- Ellenberg (robust) ---------------------------

def _flatten_cols(cols) -> List[str]:
    out: List[str] = []
    if getattr(cols, "levels", None) is not None:
        for tup in cols:
            parts = [str(x) for x in tup if x is not None and str(x).strip() and not str(x).startswith("Unnamed")]
            name = "_".join(parts)
            name = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
            out.append(name)
    else:
        for c in cols:
            name = re.sub(r"\W+", "_", str(c).strip().lower()).strip("_")
            out.append(name)
    return out

def load_ellenberg_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"Ellenberg-bestand ontbreekt: {path}")
    xl = pd.ExcelFile(path)

    def _score_columns(cols: List[str]) -> int:
        keys_tax = ("taxon","species","name","latin","scientific","genus")
        keys_env = ("light","moist","moisture","react","ph","nutr","nutrient","salin","salinity")
        s_tax = sum(any(k in c for k in keys_tax) for c in cols)
        s_env = sum(any(k in c for k in keys_env) for c in cols)
        return 10*s_tax + s_env

    best: Optional[Tuple[str,object,pd.DataFrame,int]] = None
    header_opts: List[object] = [0,1,2,[0,1],[1,2]]
    for sheet in xl.sheet_names:
        for hdr in header_opts:
            try:
                df_try = xl.parse(sheet, header=hdr)
                df_try.columns = _flatten_cols(df_try.columns)
                sc = _score_columns(df_try.columns)
                if re.search(r"harmon", sheet, re.I):
                    sc += 2
                if best is None or sc > best[3]:
                    best = (sheet, hdr, df_try, sc)
            except Exception:
                continue

    if best is None:
        raise RuntimeError("Kon geen geschikte sheet/headers vinden in Ellenberg-bestand.")

    sheet, hdr, df, sc = best
    print(f"[INFO] Ellenberg: gekozen sheet='{sheet}' header={hdr} (score={sc})")

    def _find_col(patterns: List[str]) -> Optional[str]:
        for p in patterns:
            for c in df.columns:
                if c == p:
                    return c
        for p in patterns:
            rx = re.compile(p)
            for c in df.columns:
                if rx.search(c):
                    return c
        return None

    col_tax = _find_col([r"^taxon$", r"^taxon_name$", r"^species$", r"^species_name$", r"^name$", r"^scientific_name$", r"^latin_name$"])
    if not col_tax:
        col_genus   = _find_col([r"^genus$", r"^geslacht$"])
        col_species = _find_col([r"^species$", r"^soort$", r"^epithet$", r"^species_epithet$", r"^soortepitheton$"])
        if col_genus and col_species:
            df["taxon"] = (df[col_genus].astype(str).str.strip() + " " + df[col_species].astype(str).str.strip()).str.strip()
        else:
            pattern = r"^[A-Z][A-Za-z\-]+(?:\s+[x×]\s+)?[a-z][A-Za-z\-]+$"
            def is_binom(val: str) -> bool:
                if not isinstance(val, str):
                    return False
                v = val.strip()
                return bool(re.match(pattern, v))
            best_col, best_score = None, 0.0
            for c in df.columns:
                s = df[c].dropna().astype(str).head(500)
                if s.empty:
                    continue
                score = sum(is_binom(x) for x in s) / max(1, len(s))
                if score > best_score:
                    best_col, best_score = c, score
            if best_col and best_score >= 0.10:
                df["taxon"] = df[best_col]
            else:
                raise RuntimeError("Kon taxon-kolom in Ellenberg-bestand niet vinden (ook niet via genus+species of heuristiek).")
    else:
        df["taxon"] = df[col_tax]

    def pick_metric(key: str, name_variants: List[str]) -> Optional[str]:
        col = _find_col([fr"^{key}$", fr"^ellenberg_{key}$"])  # korte code exact
        if col:
            return col
        return _find_col(name_variants)

    map_src = {}
    mL = pick_metric("l", [r"^light", r"_light_", r"light_", r"^l_light", r"licht"])
    mF = pick_metric("f", [r"^moist|moisture|water|vocht"])
    mN = pick_metric("n", [r"^nutr|nutrient|nutrients|nitrogen"])
    mR = pick_metric("r", [r"^react|reaction|ph|soil_reaction"])
    mS = pick_metric("s", [r"^salin|salinity|salt|zout"])
    if mL: map_src["ellenberg_l"] = mL
    if mF: map_src["ellenberg_f"] = mF
    if mN: map_src["ellenberg_n"] = mN
    if mR: map_src["ellenberg_r"] = mR
    if mS: map_src["ellenberg_s"] = mS

    rename = {v:k for k,v in map_src.items()}
    df = df.rename(columns=rename)
    print("[MAP] Ellenberg kolommen:", {k: map_src.get(k, None) for k in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]})

    for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["taxon_norm"] = df["taxon"].map(_norm)
    keep = ["taxon","taxon_norm","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]
    return df[keep]

# --------------------------- Build ---------------------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    sp = load_species_csv(IN_CSV_SPECIES)
    el = load_ellenberg_xlsx(IN_XLSX_ELLEN)

    sp["taxon_norm"] = sp["wetenschappelijke_naam"].map(_norm)
    merged = sp.merge(el, how="left", on="taxon_norm")

    # Fuzzy fallback per genus (alleen hoge zekerheid)
    unmatched = merged[merged["ellenberg_l"].isna()].copy()
    if not unmatched.empty:
        el_by_genus = el.groupby(el["taxon_norm"].str.split().str[0])
        filled = 0
        for idx, row in unmatched.iterrows():
            tn = str(row["taxon_norm"])
            parts = tn.split()
            if len(parts) < 2: 
                continue
            genus = parts[0]
            if genus not in el_by_genus.groups: 
                continue
            pool = el_by_genus.get_group(genus)
            sims = pool["taxon_norm"].apply(lambda x: SequenceMatcher(None, tn, x).ratio())
            j = sims.idxmax(); best = float(sims.loc[j])
            if best >= 0.96:
                for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]:
                    merged.at[idx, c] = pool.at[j, c]
                filled += 1
        print(f"[INFO] Fuzzy matches toegevoegd: {filled}")

    out = pd.DataFrame(index=merged.index, columns=APP_COLS)
    out["naam"] = merged["naam"].fillna(merged["wetenschappelijke_naam"])
    out["wetenschappelijke_naam"] = merged["wetenschappelijke_naam"]
    out["inheems"] = "ja"

    def licht_tokens(L: float | None) -> str:
        try: L = float(L)
        except Exception: return ""
        toks: List[str] = []
        if L <= 4.0: toks.append("schaduw")
        if 4.0 < L < 7.0: toks.append("half")
        if L >= 6.0: toks.append("zon")
        return ";".join(dict.fromkeys(toks))

    def vocht_token(F: float | None) -> str:
        try: F = float(F)
        except Exception: return ""
        if F <= 3.5: return "droog"
        if F >= 7.0: return "vochtig"
        return "fris"

    def droogte_tol(F: float | None) -> str:
        try: F = float(F)
        except Exception: return ""
        if F <= 3.5: return "hoog"
        if F >= 6.0: return "laag"
        return "middel"

    out["ellenberg_l"] = merged.get("ellenberg_l")
    out["ellenberg_f"] = merged.get("ellenberg_f")
    out["ellenberg_n"] = merged.get("ellenberg_n")
    out["ellenberg_r"] = merged.get("ellenberg_r")
    out["ellenberg_s"] = merged.get("ellenberg_s")

    out["standplaats_licht"] = [licht_tokens(v) for v in out["ellenberg_l"]]
    out["vocht"] = [vocht_token(v) for v in out["ellenberg_f"]]
    out["droogtetolerantie"] = [droogte_tol(v) for v in out["ellenberg_f"]]

    def zon_minmax(L: Any) -> Tuple[int,int]:
        try: L = float(L)
        except Exception: return (0, 2)
        if L <= 4.0: return (0, 1)
        if L >= 7.0: return (1, 2)
        return (0, 2)

    zm = out["ellenberg_l"].map(zon_minmax)
    out["zon_min"] = [a for (a, b) in zm]
    out["zon_max"] = [b for (a, b) in zm]

    out["klimaatzone"] = "7a"
    out["ecowaarde"]   = ""

    merged["_familie"] = merged.get("_familie", "")
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie", "")).lower()
        sci = str(row.get("wetenschappelijke_naam", ""))
        genus = sci.split(" ")[0].lower() if sci else ""
        return "ja" if ("fabaceae" in fam or genus in N_FIX_GENERA) else "nee"

    out["stikstofbinder"] = merged.apply(is_nfix, axis=1)
    out["invasief"] = "nee"

    for c in ["bodem","nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten","opmerking"]:
        out[c] = ""

    # overrides (optioneel)
    if os.path.exists(IN_CSV_OVR) and os.path.getsize(IN_CSV_OVR) >= 4:
        ovr = _read_csv_best(IN_CSV_OVR)
        ovr.columns = [re.sub(r"\W+", "_", c.strip().lower()).strip("_") for c in ovr.columns]
        if "wetenschappelijke_naam" in ovr.columns:
            ovr["_key"] = ovr["wetenschappelijke_naam"].map(_norm)
        elif "taxon" in ovr.columns:
            ovr["_key"] = ovr["taxon"].map(_norm)
        else:
            raise RuntimeError("manual_overrides.csv mist kolom 'wetenschappelijke_naam' of 'taxon'")
        out["_key"] = out["wetenschappelijke_naam"].map(_norm)
        out = out.merge(
            ovr.drop(columns=[c for c in ["wetenschappelijke_naam","taxon"] if c in ovr.columns]),
            how="left", left_on="_key", right_on="_key", suffixes=("", "__ovr")
        )
        for c in list(out.columns):
            if c.endswith("__ovr"):
                base = c[:-5]
                out[base] = out[c].where(out[c].notna() & (out[c] != ""), out.get(base))
        out = out[[c for c in out.columns if not c.endswith("__ovr") and c != "_key"]]

    for c in APP_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[APP_COLS]

    cov = out[["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]].notna().mean()*100
    print(f"[COVERAGE] L:{cov['ellenberg_l']:.1f}% F:{cov['ellenberg_f']:.1f}% N:{cov['ellenberg_n']:.1f}% R:{cov['ellenberg_r']:.1f}% S:{cov['ellenberg_s']:.1f}%")

    unmatched_taxa = merged.loc[merged["ellenberg_l"].isna(), ["wetenschappelijke_naam"]].head(200)
    if not unmatched_taxa.empty:
        os.makedirs(os.path.dirname(OUT_CSV_UNMATCH), exist_ok=True)
        unmatched_taxa.to_csv(OUT_CSV_UNMATCH, index=False, encoding="utf-8")
        print(f"[INFO] Unmatched voorbeeld geschreven: {OUT_CSV_UNMATCH} ({len(unmatched_taxa)} rijen)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    out.to_csv(OUT_CSV_SEMI, index=False, encoding="utf-8", sep=";")
    print(f"[OK] Geschreven: {OUT_CSV}  ({len(out)} rijen)")
    print(f"[OK] Geschreven: {OUT_CSV_SEMI}  ({len(out)} rijen)")
    return out

if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as e:
        print("[ERR]", e)
        sys.exit(2)
