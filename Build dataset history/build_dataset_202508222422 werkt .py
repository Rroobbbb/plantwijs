"""
Build Dataset — PlantWijs v1.6.1
• FIX: 'float' object has no attribute 'upper' → 'Rode Lijst'-waarde wordt nu altijd als tekst behandeld
• Inheems uit NDFF 'Rode Lijst' (E/OV‑1/OV‑2 ⇒ niet inheems)
• Invasief via lijst met EU/NL-problemsoorten (te overriden)
• Klimaatzone: inheemsen → '7a;7b;8a' (env: PW_HARDINESS_NATIVE), exoten → leeg
• Leest Ellenberg IVs met twee header-rijen ('LIGHT'/'Average' → L/F/N/R/S)
• CSV-lezer blijft robuust; fuzzy match per genus; uitgebreide logging

Gebruik
  cd C:\\PlantWijs
  venv\\Scripts\\python build_dataset.py

Benodigd in ./data
- verspreidingsatlas_planten.csv  (NDFF export)
- ellenberg.xlsx                  (Tichý et al. 2023 — IVs)
- manual_overrides.csv            (optioneel)
"""
from __future__ import annotations
import os, sys, re, unicodedata
from typing import Any, List, Optional, Tuple
from difflib import SequenceMatcher

try:
    import pandas as pd
except ModuleNotFoundError:
    print("[ERR] pandas ontbreekt; draai vanuit venv of installeer het.", file=sys.stderr)
    sys.exit(1)

IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_CSV_OVR     = os.getenv("IN_CSV_OVR",      "data/manual_overrides.csv")
OUT_CSV        = os.getenv("OUT_CSV",         "out/plantwijs_full.csv")
OUT_CSV_SEMI   = os.getenv("OUT_CSV_SEMI",    "out/plantwijs_full_semicolon.csv")
OUT_CSV_UNMATCH= os.getenv("OUT_CSV_UNMATCH", "out/unmatched_sample.csv")
HARDINESS_NATIVE_DEFAULT = os.getenv("PW_HARDINESS_NATIVE", "7a;7b;8a")

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
    "alnus","hippophae","myrica","morella",
    "alhagi","anthyllis","cicer","cytisus","desmodium","genista","glycyrrhiza",
    "galega","gleditsia","laburnum","lathyrus","lotus","lupinus","medicago",
    "melilotus","ononis","ornithopus","oxytropis","prosopis","robinia",
    "spartium","spiraea","trifolium","vicia"
}

INVASIVE_TAXA = {
    "ailanthus altissima","prunus serotina","acer negundo","quercus rubra","rhododendron ponticum",
    "heracleum mantegazzianum","impatiens glandulifera","impatiens parviflora","fallopia japonica",
    "lysichiton americanus","solidago canadensis","solidago gigantea","rudbeckia laciniata",
    "crassula helmsii","azolla filiculoides","lemna minuta","ludwigia grandiflora","ludwigia peploides",
}

# ---------------- Helpers ----------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    t = re.sub(r"^(\w+\s+\w+).*$", lambda m: m.group(1), t)
    return t.lower()


def _detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _read_csv_best(path: str) -> pd.DataFrame:
    enc0 = _detect_encoding(path)
    encs: List[str] = []
    for e in (enc0, "utf-8-sig", "utf-8", "latin1"):
        if e not in encs:
            encs.append(e)
    delims = ["\t", ";", ",", "|"]

    last_err: Optional[Exception] = None
    best_df: Optional[pd.DataFrame] = None
    best_cols = -1

    for enc in encs:
        for d in delims:
            engines = ("python",) if enc.startswith("utf-16") else ("c", "python")
            for eng in engines:
                try:
                    kw = dict(sep=d, dtype=str, encoding=enc)
                    if eng == "python":
                        kw.update(dict(engine="python", on_bad_lines="skip"))
                    df = pd.read_csv(path, **kw)
                    if df.shape[1] >= 8:
                        print(f"[INFO] CSV geladen enc='{enc}' delim='{d}' engine='{eng}' — {len(df)} rijen, {df.shape[1]} kolommen")
                        return df
                    if df.shape[1] > best_cols:
                        best_cols = df.shape[1]; best_df = df
                except Exception as e:
                    last_err = e
    for enc in encs:
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=enc, on_bad_lines="skip")
            if df.shape[1] >= 8:
                print(f"[INFO] CSV autodetected enc='{enc}' — {len(df)} rijen, {df.shape[1]} kolommen")
                return df
            if df.shape[1] > best_cols:
                best_cols = df.shape[1]; best_df = df
        except Exception as e:
            last_err = e
    for enc in encs:
        try:
            df = pd.read_csv(path, delim_whitespace=True, engine="python", dtype=str, encoding=enc, on_bad_lines="skip")
            if df.shape[1] >= 8:
                print(f"[INFO] CSV whitespace enc='{enc}' — {len(df)} rijen, {df.shape[1]} kolommen")
                return df
            if df.shape[1] > best_cols:
                best_cols = df.shape[1]; best_df = df
        except Exception as e:
            last_err = e
    if best_df is not None:
        print(f"[WARN] Beste parse had {best_cols} kolommen; ga daarmee verder.")
        return best_df
    raise last_err if last_err else RuntimeError("Kon soortenlijst niet parsen")

# -------------- Load sources --------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_csv_best(path)
    print("[INFO] Aangetroffen kolommen:", ", ".join(str(c) for c in list(df.columns)[:50]))
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]

    def pick_first(names: List[str]) -> Optional[str]:
        for n in names:
            if n in df.columns: return n
        return None

    col_sci   = pick_first(["wetenschappelijke_naam","soort","taxon","species","binomium","scientific_name","latin_name"]) or pick_first([c for c in df.columns if "naam" in c and "wet" in c])
    col_nl    = pick_first(["nederlandse_naam","nederlandse_soortnaam","naam","soortnaam_nl","dutch_name"]) or pick_first([c for c in df.columns if "nederlands" in c])
    col_family= pick_first(["familie","family"]) or next((c for c in df.columns if re.search(r"famil", c)), None)
    col_rl    = pick_first(["rode_lijst","red_list","rodelijst","rl","rl_status"])  # kan ontbreken

    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")

    ren = {col_sci: "wetenschappelijke_naam"}
    if col_nl:    ren[col_nl] = "naam"
    if col_family:ren[col_family] = "_familie"
    if col_rl:    ren[col_rl] = "_rode_lijst"
    df = df.rename(columns=ren)

    df["wetenschappelijke_naam"] = df["wetenschappelijke_naam"].astype(str)
    if "naam" not in df.columns:
        df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns:
        df["_familie"] = ""

    # --- Inheems bepalen uit Rode Lijst ---
    def _infer_inheems(rl: Any) -> str:
        txt = str(rl or "").strip().upper()
        if txt in {"", "NAN", "NA"}:  # geen info ⇒ conservatief 'ja'
            return "ja"
        if re.search(r"\bE\b|EXOOT", txt):  # E = Exoot in NDFF
            return "nee"
        if re.search(r"OV-?1|OV-?2", txt):  # onbestendig
            return "nee"
        return "ja"

    df["_inheems"] = df["_rode_lijst"].map(_infer_inheems) if "_rode_lijst" in df.columns else "ja"

    return df[["naam","wetenschappelijke_naam","_familie","_inheems"]]

# -------------- Ellenberg (IVs averages; 2 header-rijen) --------------

def load_ellenberg_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"Ellenberg-bestand ontbreekt: {path}")

    xl = pd.ExcelFile(path)

    # kandidaat-sheets met IVs/averages
    sheet_candidates: List[str] = []
    pref = [r"Tab-IVs-Tichy-et-al2023", r"ivs", r"tichy", r"average", r"harmon"]
    for pat in pref:
        for s in xl.sheet_names:
            if re.search(pat, s, re.I) and s not in sheet_candidates:
                sheet_candidates.append(s)
    for s in xl.sheet_names:
        if s not in sheet_candidates:
            sheet_candidates.append(s)

    def _flatten(cols) -> List[str]:
        cols = list(cols)
        out: List[str] = []
        if cols and isinstance(cols[0], tuple):
            for tup in cols:
                parts = [str(x) for x in tup if x is not None and str(x).strip() and not str(x).startswith("Unnamed")]
                name = "_".join(parts)
                name = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
                out.append(name)
        else:
            for c in cols:
                name = re.sub(r"[^0-9A-Za-z_]+", "_", str(c).strip().lower()).strip("_")
                out.append(name)
        return out

    parsed: Optional[Tuple[str,Tuple[int,...],pd.DataFrame]] = None
    for sheet in sheet_candidates:
        for header in ([0,1],[1,2],1,0,2):
            try:
                df_try = xl.parse(sheet, header=header)
                df_try.columns = _flatten(df_try.columns)
                if not df_try.empty:
                    parsed = (sheet, (tuple(header) if isinstance(header, list) else (header,) if isinstance(header,int) else header), df_try)
                    break
            except Exception:
                continue
        if parsed is not None:
            break

    if parsed is None:
        raise RuntimeError("Kon geen bruikbare sheet/header combinatie parsen uit Ellenberg-bestand.")

    sheet, header_rows, df = parsed
    print(f"[ELLEN] Gekozen sheet='{sheet}' header={list(header_rows)}")
    print("[ELLEN] Kolommen na flatten:", ", ".join(list(df.columns)[:30]))

    def _find_col(patterns: List[str]) -> Optional[str]:
        for p in patterns:
            for c in df.columns:
                if c == p: return c
        for p in patterns:
            rx = re.compile(p)
            for c in df.columns:
                if rx.search(c): return c
        return None

    col_tax = _find_col([r"^taxon$", r"^taxon_name$", r"^species$", r"^species_name$", r"^name$", r"^scientific_name$", r"^latin_name$"])
    if not col_tax:
        pattern = r"^[A-Z][A-Za-z\-]+(?:\s+[x×]\s+)?[a-z][A-Za-z\-]+$"
        best_col, best_score = None, 0.0
        for c in df.columns:
            s = df[c].dropna().astype(str).head(500)
            if s.empty: continue
            score = sum(bool(re.match(pattern, x.strip())) for x in s) / max(1, len(s))
            if score > best_score:
                best_col, best_score = c, score
        if best_col and best_score >= 0.10:
            col_tax = best_col
        else:
            raise RuntimeError("Kon taxon-kolom in Ellenberg-bestand niet vinden (ook niet via heuristiek).")

    def pick_metric(key: str, variants: List[str]) -> Optional[str]:
        return _find_col([fr"^{key}$", fr"^ellenberg_{key}$"]) or _find_col(variants)

    mL = pick_metric("l", [r"^light_average$", r"^light$", r"_light_", r"licht"]) 
    mF = pick_metric("f", [r"^moisture_average$", r"^moist$", r"moisture", r"vocht"]) 
    mN = pick_metric("n", [r"^nutrients_average$", r"^nutr", r"nutrients", r"nitrogen"]) 
    mR = pick_metric("r", [r"^reaction_average$", r"^react", r"reaction", r"ph", r"soil_reaction"]) 
    mS = pick_metric("s", [r"^salinity_average$", r"^salin", r"salinity", r"salt", r"zout"]) 

    rename = {}
    if mL: rename[mL] = "ellenberg_l"
    if mF: rename[mF] = "ellenberg_f"
    if mN: rename[mN] = "ellenberg_n"
    if mR: rename[mR] = "ellenberg_r"
    if mS: rename[mS] = "ellenberg_s"
    df = df.rename(columns=rename)

    print("[MAP] Ellenberg bron→kolom:", rename)

    keep = ["taxon","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]
    if not any(c in df.columns for c in keep[1:]):
        raise RuntimeError("Geen L/F/N/R/S-kolommen gevonden in gekozen sheet. Controleer of koppen 'LIGHT Average' etc. aanwezig zijn.")

    df = df[[c for c in keep if c in df.columns]].dropna(subset=["taxon"]).copy()
    for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["taxon_norm"] = df["taxon"].map(_norm)

    print("[DBG] Eerste 5 Ellenberg-rijen:")
    try:
        print(df.head()[[c for c in ["taxon","taxon_norm","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in df.columns]].to_string(index=False))
    except Exception:
        pass

    return df[[c for c in ["taxon","taxon_norm","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in df.columns]]

# -------------- Build --------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    sp = load_species_csv(IN_CSV_SPECIES)
    el = load_ellenberg_xlsx(IN_XLSX_ELLEN)

    sp["taxon_norm"] = sp["wetenschappelijke_naam"].map(_norm)
    merged = sp.merge(el, how="left", on="taxon_norm")

    # Fuzzy per genus voor Ellenberg-missers
    unmatched = merged[merged["ellenberg_l"].isna()].copy()
    if not unmatched.empty and not el.empty:
        el_by_genus = el.groupby(el["taxon_norm"].str.split().str[0])
        filled = 0
        for idx, row in unmatched.iterrows():
            tn = str(row["taxon_norm"])  
            parts = tn.split()
            if len(parts) < 2: continue
            genus = parts[0]
            if genus not in el_by_genus.groups: continue
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

    # inheems
    out["inheems"] = merged.get("_inheems", "ja")

    # ellenberg + afgeleiden
    out["ellenberg_l"] = merged.get("ellenberg_l")
    out["ellenberg_f"] = merged.get("ellenberg_f")
    out["ellenberg_n"] = merged.get("ellenberg_n")
    out["ellenberg_r"] = merged.get("ellenberg_r")
    out["ellenberg_s"] = merged.get("ellenberg_s")

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
    out["zon_min"], out["zon_max"] = [a for a,b in zm], [b for a,b in zm]

    # klimaat
    out["klimaatzone"] = out["inheems"].map(lambda x: HARDINESS_NATIVE_DEFAULT if str(x).lower()=="ja" else "")

    # stikstofbinder
    merged["_familie"] = merged.get("_familie", "")
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie", "")).lower()
        sci = str(row.get("wetenschappelijke_naam", ""))
        genus = sci.split(" ")[0].lower() if sci else ""
        return "ja" if ("fabaceae" in fam or genus in N_FIX_GENERA) else "nee"
    out["stikstofbinder"] = merged.apply(is_nfix, axis=1)

    # invasief
    out["_norm_taxon"] = out["wetenschappelijke_naam"].map(_norm)
    out["invasief"] = out["_norm_taxon"].map(lambda t: "ja" if t in INVASIVE_TAXA else "nee")

    # defaults
    for c in ["bodem","ecowaarde","nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten","opmerking"]:
        out[c] = out.get(c, "")

    # overrides
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
        out = out.merge(ovr.drop(columns=[c for c in ["wetenschappelijke_naam","taxon"] if c in ovr.columns]), how="left", left_on="_key", right_on="_key", suffixes=("", "__ovr"))
        for c in list(out.columns):
            if c.endswith("__ovr"):
                base = c[:-5]
                out[base] = out[c].where(out[c].notna() & (out[c] != ""), out.get(base))
        out = out[[c for c in out.columns if not c.endswith("__ovr") and c not in {"_key","_norm_taxon"}]]

    # reorder
    for c in APP_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[APP_COLS]

    # coverage
    if {"ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"}.issubset(out.columns):
        cov = out[["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]].notna().mean()*100
        print(f"[COVERAGE] L:{cov['ellenberg_l']:.1f}% F:{cov['ellenberg_f']:.1f}% N:{cov['ellenberg_n']:.1f}% R:{cov['ellenberg_r']:.1f}% S:{cov['ellenberg_s']:.1f}%")

    # unmatched sample
    if "ellenberg_l" in merged.columns:
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
