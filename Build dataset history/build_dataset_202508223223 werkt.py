"""
PlantWijs Dataset Builder — v2.1

- Ellenberg: ondersteunt MultiIndex headers (LIGHT/MOISTURE/... + Average) met header=[0,1] en fallbacks (0|1|2).
  Kiest per metric (L/F/N/R/S) de kolom met de meeste numerieke waarden.
- SL2020: scant alle sheets, vindt kolommen heuristisch, en zet inheems = ja voor indigeniteit die met 'i' begint
  of NSR in {1a,1b,2a}.
- CSV-reader is robuust voor encoding/delimiter.
- Geen KeyErrors als Ellenberg-kolommen ontbreken; lege kolommen zijn toegestaan.
- Schrijft beide exports: komma en puntkomma.
"""
from __future__ import annotations
import os, sys, re, unicodedata
from typing import Any, List, Optional, Tuple, Dict

try:
    import pandas as pd
except ModuleNotFoundError:
    print("[ERR] pandas ontbreekt; draai vanuit venv of installeer het.", file=sys.stderr)
    print("Huidige interpreter:", sys.executable, file=sys.stderr)
    sys.exit(1)

# --------------------------- Config ---------------------------
IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_XLSX_SL2020 = os.getenv("IN_XLSX_SL2020",  "data/standaardlijst2020.xlsx")
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

# Nitrogen-fixing genera (Fabaceae + actinorhizal) — NL-relevant subset
N_FIX_GENERA = {
    # Actinorhizal
    "alnus", "hippophae", "myrica", "morella",
    # Fabaceae (selection)
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

LICHT_TOKENS = {"schaduw","half","zon"}
VOCHT_TOKENS = {"droog","fris","vochtig"}

# --------------------------- Helpers ---------------------------

def _norm(s: Any) -> str:
    """Normaliseer naam: ascii fold, lower, verwijder auteurs & infraspecifieke rangen."""
    if s is None:
        return ""
    t = str(s)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    m = re.match(r"^(\w+\s+\w+).*$", t)  # pak alleen genus + soort
    if m:
        t = m.group(1)
    return t.lower()

def _detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"): return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"): return "utf-8-sig"
    return "utf-8"

def _read_csv_best(path: str) -> pd.DataFrame:
    enc0 = _detect_encoding(path)
    encs: List[str] = []
    for e in (enc0, "utf-8-sig", "utf-8", "latin1"):
        if e not in encs: encs.append(e)
    delims = ["\t", ";", ",", "|", None]

    last_err: Optional[Exception] = None
    best_df: Optional[pd.DataFrame] = None
    best_cols = -1

    for enc in encs:
        for d in delims:
            engines = ("python",) if enc.startswith("utf-16") or d is None else ("c","python")
            for eng in engines:
                try:
                    kw = dict(dtype=str, encoding=enc)
                    if d is None:
                        kw.update(dict(sep=None, engine="python", on_bad_lines="skip"))
                    else:
                        kw.update(dict(sep=d))
                        if eng == "python": kw.update(dict(engine="python", on_bad_lines="skip"))
                    df = pd.read_csv(path, **kw)
                    if df.shape[1] >= 2:
                        used = d if d is not None else "auto"
                        print(f"[INFO] CSV geladen enc='{enc}' delim='{used}' engine='{eng}' — {len(df)} rijen, {df.shape[1]} kolommen")
                        return df
                    if df.shape[1] > best_cols:
                        best_cols = df.shape[1]; best_df = df
                except Exception as e:
                    last_err = e
    if best_df is not None:
        print(f"[WARN] Beste parse had {best_cols} kolommen; ga daarmee verder.")
        return best_df
    raise last_err if last_err else RuntimeError(f"Kon CSV niet parsen: {path}")

# --------------------------- Load sources ---------------------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_csv_best(path)
    print("[INFO] Aangetroffen kolommen:", ", ".join([str(c) for c in list(df.columns)[:50]]))
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]

    def pick_first(names: List[str]) -> Optional[str]:
        for n in names:
            if n in df.columns: return n
        return None

    col_sci   = pick_first(["wetenschappelijke_naam","soort","taxon","species","binomium","scientific_name","latin_name"]) or \
                pick_first([c for c in df.columns if "naam" in c and "wet" in c])
    col_nl    = pick_first(["nederlandse_naam","nederlandse_soortnaam","naam","soortnaam_nl","dutch_name"]) or \
                pick_first([c for c in df.columns if "nederlands" in c])
    col_family= pick_first(["familie","family"]) or next((c for c in df.columns if re.search(r"famil", c)), None)

    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")

    ren = {col_sci: "wetenschappelijke_naam"}
    if col_nl:     ren[col_nl] = "naam"
    if col_family: ren[col_family] = "_familie"
    df = df.rename(columns=ren)

    df["wetenschappelijke_naam"] = df["wetenschappelijke_naam"].astype(str)
    if "naam" not in df.columns: df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns: df["_familie"] = ""
    return df[["naam","wetenschappelijke_naam","_familie"]]

# --------------------------- Ellenberg (robust, MultiIndex) ---------------------------

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

def load_ellenberg_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"Ellenberg-bestand ontbreekt: {path}")

    xl = pd.ExcelFile(path)
    header_opts: List[Any] = [ [0,1], [1,2], 0, 1, 2 ]

    best_df: Optional[pd.DataFrame] = None
    chosen = None

    def find_taxon_and_metrics(df0: pd.DataFrame) -> Optional[pd.DataFrame]:
        df = df0.copy()
        df.columns = _flatten_cols(df.columns)

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
            col_genus   = _find_col([r"^genus$", r"^geslacht$"])
            col_species = _find_col([r"^species$", r"^soort$", r"^epithet$", r"^species_epithet$", r"^soortepitheton$"])
            if col_genus and col_species:
                df["taxon"] = (df[col_genus].astype(str).str.strip() + " " + df[col_species].astype(str).str.strip()).str.strip()
            else:
                pattern = r"^[A-Z][A-Za-z\-]+(?:\s+[x×]\s+)?[a-z][A-Za-z\-]+$"
                best_col, best_score = None, 0.0
                for c in df.columns:
                    s = df[c].dropna().astype(str).head(1000)
                    if s.empty: continue
                    score = sum(bool(re.match(pattern, x.strip())) for x in s) / max(1, len(s))
                    if score > best_score:
                        best_col, best_score = c, score
                if best_col and best_score >= 0.05:
                    df["taxon"] = df[best_col]
                else:
                    return None
        else:
            df["taxon"] = df[col_tax]

        metric_map: Dict[str, List[str]] = {
            "ellenberg_l": [r"\blight\b", r"\blight_", r"_light\b", r"^l$", r"ellenberg_l"],
            "ellenberg_f": [r"\bmoist|moisture\b", r"^f$", r"ellenberg_f"],
            "ellenberg_n": [r"\bnutr|nutrient|nitrogen\b", r"^n$", r"ellenberg_n"],
            "ellenberg_r": [r"\breac|reaction|soil_reaction|ph\b", r"^r$", r"ellenberg_r"],
            "ellenberg_s": [r"\bsalin|salinity|salt\b", r"^s$", r"ellenberg_s"],
        }

        chosen_cols: Dict[str, str] = {}
        for out_name, patterns in metric_map.items():
            candidates = [c for c in df.columns if any(re.search(p, c) for p in patterns)]
            if not candidates:
                continue
            best_c, best_score = None, -1.0
            for c in candidates:
                s = pd.to_numeric(df[c], errors="coerce")
                score = s.notna().mean()
                if score > best_score:
                    best_c, best_score = c, score
            if best_c is not None and best_score > 0:
                chosen_cols[out_name] = best_c

        df_out = df[["taxon"]].copy()
        for out_name, src_col in chosen_cols.items():
            df_out[out_name] = pd.to_numeric(df[src_col], errors="coerce")

        if "taxon" not in df_out.columns:
            return None
        df_out = df_out.dropna(subset=["taxon"])
        df_out["taxon_norm"] = df_out["taxon"].map(_norm)
        return df_out

    for sheet in xl.sheet_names:
        for header in header_opts:
            try:
                df_try = xl.parse(sheet, header=header)
            except Exception:
                continue
            cand = find_taxon_and_metrics(df_try)
            if cand is not None:
                best_df = cand; chosen = (sheet, header)
                if re.search("harmon|ivs|tichy", sheet, re.I):
                    break
        if best_df is not None and re.search("harmon|ivs|tichy", sheet, re.I):
            break

    if best_df is None:
        raise RuntimeError("Kon taxon-kolom in Ellenberg-bestand niet vinden (na alle fallback-pogingen).")

    print(f"[ELLEN] Gekozen sheet='{chosen[0]}' header={chosen[1]}")
    print("[ELLEN] Kolommen:", ", ".join([c for c in best_df.columns if c != "taxon_norm"]))
    return best_df

# --------------------------- SL2020 — inheems/ingeburgerd ---------------------------

def load_sl2020(path_xlsx: str) -> pd.DataFrame:
    if not os.path.exists(path_xlsx):
        raise RuntimeError(
            f"SL2020 Excel ontbreekt: {path_xlsx} — plaats de Excel-appendix als data/standaardlijst2020.xlsx"
        )

    xl = pd.ExcelFile(path_xlsx, engine="openpyxl")
    frames: List[pd.DataFrame] = []

    def pick_cols(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        d = df.copy()
        d.columns = [str(c).strip().lower() for c in d.columns]
        candidates = [c for c in d.columns if any(k in c for k in ["wet","scientific","latijn","latin"]) and "naam" in c]
        col_sci = candidates[0] if candidates else None
        if not col_sci:
            pattern = r"^[A-Z][A-Za-z\-]+(?:\s+[x×]\s+)?[a-z][A-Za-z\-]+$"
            best_col, best_score = None, 0.0
            for c in d.columns:
                s = d[c].dropna().astype(str).head(1000)
                score = sum(bool(re.match(pattern, x.strip())) for x in s) / max(1, len(s)) if len(s) else 0.0
                if score > best_score:
                    best_col, best_score = c, score
            if best_col and best_score >= 0.05:
                col_sci = best_col
        if not col_sci:
            return None

        col_stat = next((c for c in d.columns if "status" in c and "nsr" in c), None)
        col_ind  = next((c for c in d.columns if "indigen" in c), None)

        out = pd.DataFrame({
            "wetenschappelijke_naam": d[col_sci].astype(str),
            "status_nsr": d[col_stat].astype(str) if col_stat else "",
            "indigeniteit": d[col_ind].astype(str) if col_ind else "",
        })
        out["taxon_norm"] = out["wetenschappelijke_naam"].map(_norm)
        return out

    for sheet in xl.sheet_names:
        for header in ([0,1], [1,2], 0, 1, 2):
            try:
                df_try = xl.parse(sheet, header=header)
            except Exception:
                continue
            got = pick_cols(df_try)
            if got is not None:
                frames.append(got)
                break

    if not frames:
        raise RuntimeError("SL2020: kon kolommen niet vinden (geen sheet met wetenschappelijke namen)")

    sl = pd.concat(frames, ignore_index=True)
    sl = sl.drop_duplicates(subset=["taxon_norm"], keep="first")

    def map_inheems(status_nsr: str, indigen: str) -> str:
        s = str(status_nsr or "").strip().lower()
        i = str(indigen or "").strip().lower()
        if i.startswith("i") or s.startswith("1a") or s.startswith("1b") or s.startswith("2a"):
            return "ja"
        return "nee"

    sl["inheems"] = [map_inheems(s, i) for s, i in zip(sl["status_nsr"], sl["indigeniteit"])]
    return sl[["taxon_norm","inheems","status_nsr","indigeniteit"]]

# --------------------------- Build ---------------------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    sp = load_species_csv(IN_CSV_SPECIES)
    el = load_ellenberg_xlsx(IN_XLSX_ELLEN)

    try:
        sl = load_sl2020(IN_XLSX_SL2020)
    except Exception as e:
        print("[WARN] SL2020 niet geladen:", e)
        sl = pd.DataFrame(columns=["taxon_norm","inheems","status_nsr","indigeniteit"])

    sp["taxon_norm"] = sp["wetenschappelijke_naam"].map(_norm)
    merged = sp.merge(el, how="left", on="taxon_norm")
    if not sl.empty:
        merged = merged.merge(sl, how="left", on="taxon_norm")

    out = pd.DataFrame(index=merged.index, columns=APP_COLS)
    out["naam"] = merged["naam"].fillna(merged["wetenschappelijke_naam"])
    out["wetenschappelijke_naam"] = merged["wetenschappelijke_naam"]
    out["inheems"] = merged["inheems"] if "inheems" in merged.columns else "nee"

    def getcol(name: str):
        return merged[name] if name in merged.columns else pd.Series([None]*len(merged), index=merged.index)

    out["ellenberg_l"] = getcol("ellenberg_l")
    out["ellenberg_f"] = getcol("ellenberg_f")
    out["ellenberg_n"] = getcol("ellenberg_n")
    out["ellenberg_r"] = getcol("ellenberg_r")
    out["ellenberg_s"] = getcol("ellenberg_s")

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

    out["klimaatzone"] = out["inheems"].map(lambda x: HARDINESS_NATIVE_DEFAULT if str(x).lower()=="ja" else "")

    merged["_familie"] = merged.get("_familie", "")
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie", "")).lower()
        sci = str(row.get("wetenschappelijke_naam", ""))
        genus = sci.split(" ")[0].lower() if sci else ""
        return "ja" if ("fabaceae" in fam or genus in N_FIX_GENERA) else "nee"
    out["stikstofbinder"] = merged.apply(is_nfix, axis=1)

    out["_norm_taxon"] = out["wetenschappelijke_naam"].map(_norm)
    out["invasief"] = out["_norm_taxon"].map(lambda t: "ja" if t in INVASIVE_TAXA else "nee")

    for c in ["bodem","ecowaarde","nectarwaarde","pollenwaarde","waarde_vogels","waarde_insecten","opmerking"]:
        out[c] = out.get(c, "")

    if os.path.exists(IN_CSV_OVR) and os.path.getsize(IN_CSV_OVR) >= 4:
        try:
            ovr = _read_csv_best(IN_CSV_OVR)
        except Exception as e:
            print(f"[WARN] Kon overrides niet lezen ({IN_CSV_OVR}): {e}")
            ovr = pd.DataFrame()
        if not ovr.empty:
            ovr.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in ovr.columns]
            if "wetenschappelijke_naam" in ovr.columns:
                ovr["_key"] = ovr["wetenschappelijke_naam"].map(_norm)
            elif "taxon" in ovr.columns:
                ovr["_key"] = ovr["taxon"].map(_norm)
            else:
                print("[WARN] overrides mist kolom 'wetenschappelijke_naam' of 'taxon' — overslaan.")
                ovr = pd.DataFrame()
        if not ovr.empty:
            out["_key"] = out["wetenschappelijke_naam"].map(_norm)
            out = out.merge(
                ovr.drop(columns=[c for c in ["wetenschappelijke_naam","taxon"] if c in ovr.columns]),
                how="left", left_on="_key", right_on="_key", suffixes=("", "__ovr")
            )
            for c in APP_COLS:
                oc = c + "__ovr"
                if oc in out.columns:
                    out[c] = out[oc].where(out[oc].notna() & (out[oc] != ""), out[c])
            out = out[[c for c in out.columns if not c.endswith("__ovr") and c not in {"_key","_norm_taxon"}]]

    for c in APP_COLS:
        if c not in out.columns: out[c] = ""
    out = out[APP_COLS]

    if {"ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"}.issubset(out.columns):
        cov = out[["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]].notna().mean()*100
        print(f"[COVERAGE] L:{cov['ellenberg_l']:.1f}% F:{cov['ellenberg_f']:.1f}% N:{cov['ellenberg_n']:.1f}% R:{cov['ellenberg_r']:.1f}% S:{cov['ellenberg_s']:.1f}%")

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

def _selftest():
    assert _norm("Quercus robur L.") == "quercus robur"

if __name__ == "__main__":
    try:
        _selftest()
        build_dataset()
    except Exception as e:
        print("[ERR]", e)
        sys.exit(2)
