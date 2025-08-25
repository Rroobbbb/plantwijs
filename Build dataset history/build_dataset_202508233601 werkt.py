"""
Build a comprehensive PlantWijs dataset for the web/app from open sources.

Inputs (you download once):
- data/verspreidingsatlas_planten.csv   # Soortenlijst (CSV/TSV) van NDFF Verspreidingsatlas (planten)
- data/ellenberg.xlsx                   # Ellenberg-type indicator values (Tichý et al. / Harmonized)
- data/standaardlijst2020.xlsx          # (aanbevolen) SL2020 Excel-appendix met status/indigeniteit
- data/manual_overrides.csv            # (optioneel) Handmatige correcties / aanvullingen

Output:
- out/plantwijs_full.csv               # komma, punt als decimaal (voor programmatisch gebruik)
- out/plantwijs_full_semicolon.csv    # puntkomma, komma als decimaal (Excel NL‑vriendelijk)

Usage (PowerShell/CMD from project root):
  venv\Scripts\python build_dataset.py

Afspraken:
- standplaats_licht = één label uit Ellenberg L: schaduw / halfschaduw / zon
- vocht = 5 klassen uit Ellenberg F: zeer droog, droog, vochtig, nat, zeer nat (F ≥ 8.0 ⇒ zeer nat)
- inheems = ja bij SL2020 NSR-status 1a/1b/2a of indigeniteit die met 'i' begint; anders nee.
- Dedup: eerst Ellenberg aggregeren per taxon_norm (mean), daarna na merge nogmaals per wetenschappelijke_naam collapsen.
"""
from __future__ import annotations
import os, sys, re, unicodedata
from typing import Any, List, Optional, Tuple

import pandas as pd
pd.options.mode.copy_on_write = True  # vermijdt chained-assign waarschuwingen

# --------------------------- Config ---------------------------
IN_CSV_SPECIES = os.getenv("IN_CSV_SPECIES", "data/verspreidingsatlas_planten.csv")
IN_XLSX_ELLEN  = os.getenv("IN_XLSX_ELLEN",   "data/ellenberg.xlsx")
IN_XLSX_SL2020 = os.getenv("IN_XLSX_SL2020",  "data/standaardlijst2020.xlsx")
IN_CSV_OVR     = os.getenv("IN_CSV_OVR",      "data/manual_overrides.csv")
OUT_CSV        = os.getenv("OUT_CSV",         "out/plantwijs_full.csv")
OUT_CSV_SEMI   = os.getenv("OUT_CSV_SEMI",    "out/plantwijs_full_semicolon.csv")

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
    # Actinorhizal
    "alnus", "hippophae", "myrica", "morella",
    # Fabaceae (selection)
    "alhagi","anthyllis","cicer","cytisus","desmodium","genista","glycyrrhiza",
    "galega","gleditsia","laburnum","lathyrus","lotus","lupinus","medicago",
    "melilotus","ononis","ornithopus","oxytropis","prosopis","robinia",
    "spartium","spiraea","trifolium","vicia"
}

VOCHT_TOKENS = {"zeer droog","droog","vochtig","nat","zeer nat"}
LICHT_TOKENS = {"zon","halfschaduw","schaduw"}

# --------------------------- Helpers ---------------------------

def _norm(s: Any) -> str:
    """Normaliseer naar 'genus soort' (ASCII, lowercase, zonder auteurs/varianten)."""
    if s is None:
        return ""
    t = str(s)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.strip()
    t = re.sub(r"\s+[x×]\s+", " ", t, flags=re.I)  # hybrides
    t = re.sub(r"\s+(subsp\.|ssp\.|var\.|subvar\.|f\.|forma)\b.*$", "", t, flags=re.I)
    m = re.match(r"^(\w+\s+\w+).*$", t)
    if m:
        t = m.group(1)
    return t.lower()


def _read_table_best(path: str) -> pd.DataFrame:
    """Robuuste reader voor CSV/TSV met verschillende encodings en delimiters."""
    encs  = ["utf-8","utf-16","utf-16-le","utf-16-be","cp1252"]
    seps  = ["\t",";",",","|"," "]
    last_err: Optional[Exception] = None
    for enc in encs:
        for sep in seps:
            try:
                eng = "python" if sep == " " else "c"
                df = pd.read_csv(path, sep=sep, engine=eng, dtype=str, encoding=enc, on_bad_lines="skip")
                if df.shape[1] == 1 and sep != " ":
                    continue
                sep_disp = sep.replace("\t", "\\t")
                print("[INFO] CSV geladen enc='" + enc + "' delim='" + sep_disp + "' engine='" + eng +
                      "' — " + str(len(df)) + " rijen, " + str(df.shape[1]) + " kolommen")
                return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError("Kon CSV niet lezen met fallback-readers: " + str(last_err))

# --------------------------- Load sources ---------------------------

def load_species_csv(path: str) -> pd.DataFrame:
    df = _read_table_best(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_sci = next((c for c in df.columns if c in {"wetenschappelijke_naam","soort","taxon","species","binomium"}), None)
    col_nl  = next((c for c in df.columns if c in {"nederlandse_naam","nederlandse_soortnaam","naam","soortnaam_nl"}), None)
    col_status = next((c for c in df.columns if c in {"status","herkomst","inheems","inheems_of_niet"}), None)
    col_family = next((c for c in df.columns if c in {"familie","family"}), None)
    if not col_sci:
        raise RuntimeError("Kon kolom met wetenschappelijke_naam niet vinden in soortenlijst.")
    df = df.rename(columns={col_sci: "wetenschappelijke_naam"})
    if col_nl: df = df.rename(columns={col_nl: "naam"})
    if col_status: df = df.rename(columns={col_status: "_status"})
    if col_family: df = df.rename(columns={col_family: "_familie"})
    if "_status" in df.columns:
        mask_inh = df["_status"].astype(str).str.contains("inheems", case=False, na=False)
        df = df.loc[mask_inh].copy()
    df["wetenschappelijke_naam"] = df["wetenschappelijke_naam"].astype(str)
    if "naam" not in df.columns:
        df["naam"] = df["wetenschappelijke_naam"]
    if "_familie" not in df.columns:
        df["_familie"] = ""
    print("[INFO] Aangetroffen kolommen:", ", ".join([str(c) for c in list(df.columns)[:50]]))
    return df[["naam","wetenschappelijke_naam","_familie"]]


def load_ellenberg_xlsx(path: str) -> pd.DataFrame:
    """Ellenberg-lezer: detecteert sheet/header, normaliseert kolommen, en **dedupliceert per taxon_norm**."""
    if not os.path.exists(path):
        raise RuntimeError("Ellenberg-bestand ontbreekt: " + path)
    xl = pd.ExcelFile(path)

    def _flatten(cols) -> List[str]:
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

    def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
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

    def _pick_metric(df: pd.DataFrame, key: str, variants: List[str]) -> Optional[str]:
        col = _find_col(df, ["^" + key + "$", "^ellenberg_" + key + "$"])
        if col:
            return col
        return _find_col(df, variants)

    def _try_parse(sheet: str, header_row: int):
        df_try = xl.parse(sheet, header=header_row)
        df_try.columns = _flatten(df_try.columns)
        col_tax = _find_col(df_try, ["^taxon$","^taxon_name$","^species$","^species_name$","^name$","^scientific_name$","^latin_name$"])
        mL = _pick_metric(df_try, "l", ["^light","_light_","licht"]) 
        mF = _pick_metric(df_try, "f", ["^moist","moisture","vocht","water"]) 
        mN = _pick_metric(df_try, "n", ["^nutr","nutrient","nutrients","nitrogen"]) 
        mR = _pick_metric(df_try, "r", ["^react","reaction","ph","soil_reaction"]) 
        mS = _pick_metric(df_try, "s", ["^salin","salinity","salt","zout"]) 
        rename = {}
        if mL: rename[mL] = "ellenberg_l"
        if mF: rename[mF] = "ellenberg_f"
        if mN: rename[mN] = "ellenberg_n"
        if mR: rename[mR] = "ellenberg_r"
        if mS: rename[mS] = "ellenberg_s"
        dfx = df_try.rename(columns=rename)
        metrics_found = [c for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in dfx.columns]
        return dfx, col_tax, len(metrics_found)

    candidates = []
    for sheet in xl.sheet_names:
        for header_row in (0,1,2):
            try:
                dfx, col_tax, mcount = _try_parse(sheet, header_row)
                score = mcount + (3 if re.search("harmon|tichy|iv", sheet, re.I) else 0) + (5 if col_tax else 0)
                candidates.append((score, sheet, header_row, dfx, col_tax, mcount))
            except Exception:
                continue

    if not candidates:
        raise RuntimeError("Geen parsebare sheets/headers in Ellenberg-bestand.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, sheet, header_row, df, col_tax, _ = candidates[0]
    print("[ELLEN] Gekozen sheet='" + str(sheet) + "' header=" + str(header_row))

    if not col_tax:
        # heuristiek: kies kolom die vaak binomina bevat
        pattern = r"^[A-Z][A-Za-z-]+(?:[ ]+[x×][ ]+)?[a-z][A-Za-z-]+$"
        best_col, best_score = None, 0.0
        for c in df.columns:
            s = df[c].dropna().astype(str).head(500)
            if s.empty:
                continue
            score = sum(bool(re.match(pattern, v.strip())) for v in s) / max(1, len(s))
            if score > best_score:
                best_col, best_score = c, score
        if best_col and best_score >= 0.10:
            df["taxon"] = df[best_col]
        else:
            raise RuntimeError("Kon taxon-kolom in Ellenberg-bestand niet vinden (ook niet via heuristiek).")
    else:
        df["taxon"] = df[col_tax]

    keep = [c for c in ["taxon","ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in df.columns]
    df = df[keep].dropna(subset=["taxon"]).copy()

    for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["taxon_norm"] = df["taxon"].map(_norm)

    # >>> DEDUPE: per taxon_norm middelen <<<
    num_cols = [c for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in df.columns]
    agg = {c: "mean" for c in num_cols}
    df = df.groupby("taxon_norm", as_index=False).agg(agg)
    return df

# --------------------------- SL2020 — inheems ---------------------------

def load_sl2020(path_xlsx: str) -> pd.DataFrame:
    if not os.path.exists(path_xlsx):
        raise RuntimeError("SL2020 Excel ontbreekt: " + path_xlsx)
    xl = pd.ExcelFile(path_xlsx, engine="openpyxl")

    def _flat(cols) -> List[str]:
        out = []
        if getattr(cols, "levels", None) is not None:
            for tup in cols:
                parts = [str(x) for x in tup if x is not None and str(x).strip() and not str(x).startswith("Unnamed")]
                name = "_".join(parts)
                out.append(re.sub("[^0-9A-Za-z_]+","_", name.strip().lower()).strip("_"))
        else:
            for c in cols:
                out.append(re.sub("[^0-9A-Za-z_]+","_", str(c).strip().lower()).strip("_"))
        return out

    def _normkey(name: str) -> str:
        return re.sub(r"[^a-z]", "", str(name).lower())

    best = None
    for sheet in xl.sheet_names:
        for header in (0,1,2,3):
            try:
                df_try = xl.parse(sheet, header=header)
                df_try.columns = _flat(df_try.columns)
                cols_norm = [_normkey(c) for c in df_try.columns]
                sci_candidates = {"wetenschappelijkenaam","wetnaam","scientificname","naamwetenschappelijk","wetenschappelijke_naam","wetenschappelijkenaamvande_soort"}
                has_sci = any(c in sci_candidates for c in cols_norm)
                score = 0
                score += 5 if has_sci else 0
                score += 1 * sum("status" in c and "nsr" in c for c in cols_norm)
                score += 1 * sum("indigen" in c for c in cols_norm)
                score += 2 if re.search("sl2020|appendix|tab", sheet, re.I) else 0
                if best is None or score > best[0]:
                    best = (score, sheet, header, df_try)
            except Exception:
                continue

    if best is None:
        raise RuntimeError("SL2020: kon geen geschikte sheet/header vinden")

    _, sheet, header, sl = best
    print("[SL2020] Gekozen sheet='" + str(sheet) + "' header=" + str(header))

    sl.columns = _flat(sl.columns)
    cols_norm = { _normkey(c): c for c in sl.columns }

    def pick(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in cols_norm: return cols_norm[k]
        return None

    col_sci = pick(["wetenschappelijkenaam","wetnaam","scientificname","naamwetenschappelijk","wetenschappelijke_naam","wetenschappelijkenaamvande_soort"]) or \
              pick(["taxon","name","species","speciesname","latinname"])  # fallback
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

# --------------------------- Build ---------------------------

def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    sp = load_species_csv(IN_CSV_SPECIES).copy()
    sp.loc[:, "taxon_norm"] = sp["wetenschappelijke_naam"].map(_norm)  # .loc voorkomt SettingWithCopyWarning
    base_rows = len(sp)

    el = load_ellenberg_xlsx(IN_XLSX_ELLEN)

    try:
        sl = load_sl2020(IN_XLSX_SL2020)
    except Exception as e:
        print("[WARN] SL2020 niet geladen:", e)
        sl = pd.DataFrame(columns=["taxon_norm","inheems","status_nsr","indigeniteit"])  # leeg

    merged = sp.merge(el, how="left", on="taxon_norm")
    if not sl.empty:
        merged = merged.merge(sl, how="left", on="taxon_norm")

    print(f"[DBG] Na merge: {len(merged)} rijen (base {base_rows})")

    # ---- COLLAPSEN NA MERGE ----
    num_cols = [c for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"] if c in merged.columns]

    def any_ja(s: pd.Series) -> str:
        return "ja" if (s.astype(str).str.lower() == "ja").any() else "nee"

    agg = {
        "naam": "first",
        "wetenschappelijke_naam": "first",
        "_familie": "first",
        "inheems": any_ja,
    }
    for c in num_cols:
        agg[c] = "mean"

    collapsed = (merged
                 .groupby("wetenschappelijke_naam", as_index=False)
                 .agg(agg))

    print(f"[DBG] Na collapse: {len(collapsed)} rijen (verwacht ≈ {base_rows})")

    # Defaults + derived fields
    out = pd.DataFrame(columns=APP_COLS)
    out["naam"] = collapsed["naam"].fillna(collapsed["wetenschappelijke_naam"])  # fallback
    out["wetenschappelijke_naam"] = collapsed["wetenschappelijke_naam"]
    out["inheems"] = collapsed.get("inheems").fillna("nee")

    for c in ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]:
        out[c] = collapsed.get(c)

    # ROND AF op 1 decimaal (stabiele export) en houd floats
    num_cols_ellen = ["ellenberg_l","ellenberg_f","ellenberg_n","ellenberg_r","ellenberg_s"]
    for c in num_cols_ellen:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    # standplaats_licht: enkel label
    def licht_label(L: float | None) -> str:
        try:
            L = float(L)
        except Exception:
            return ""
        if L <= 3.5: return "schaduw"
        if L < 6.0:  return "halfschaduw"
        return "zon"

    out["standplaats_licht"] = [licht_label(v) for v in out["ellenberg_l"]]

    # zon_min/max (0/1/2) afgeleid van label
    def zon_minmax_single(L: Any) -> Tuple[int,int]:
        lab = licht_label(L)
        if lab == "schaduw":      return 0, 0
        if lab == "halfschaduw":  return 1, 1
        if lab == "zon":          return 2, 2
        return 0, 2

    zm = out["ellenberg_l"].map(zon_minmax_single)
    out["zon_min"] = [a for (a, b) in zm]
    out["zon_max"] = [b for (a, b) in zm]

    # vocht: 5 klassen
    def vocht_label(F: float | None) -> str:
        try:
            F = float(F)
        except Exception:
            return ""
        if F <= 2.5: return "zeer droog"
        if F <= 4.5: return "droog"
        if F <= 6.5: return "vochtig"
        if F < 8.0:  return "nat"
        return "zeer nat"

    out["vocht"] = [vocht_label(v) for v in out["ellenberg_f"]]

    # (voor nu) leeg/heuristisch
    out["klimaatzone"] = ""
    out["ecowaarde"] = ""
    out["bodem"] = ""
    out["droogtetolerantie"] = ""

    # stikstofbinder
    def is_nfix(row: pd.Series) -> str:
        fam = str(row.get("_familie", "")).lower()
        sci = str(row.get("wetenschappelijke_naam", ""))
        genus = sci.split(" ")[0].lower() if sci else ""
        if "fabaceae" in fam or genus in N_FIX_GENERA:
            return "ja"
        return "nee"

    collapsed["_familie"] = collapsed.get("_familie", "")
    out["stikstofbinder"] = collapsed.apply(is_nfix, axis=1)
    out["invasief"] = "nee"

    out["nectarwaarde"] = ""
    out["pollenwaarde"] = ""
    out["waarde_vogels"] = ""
    out["waarde_insecten"] = ""
    out["opmerking"] = ""

    # Handmatige overrides (optioneel)
    if os.path.exists(IN_CSV_OVR) and os.path.getsize(IN_CSV_OVR) > 5:
        ovr = _read_table_best(IN_CSV_OVR)
        if not ovr.empty:
            ovr.columns = [c.strip().lower().replace(" ", "_") for c in ovr.columns]
            if "wetenschappelijke_naam" in ovr.columns:
                ovr["_key"] = ovr["wetenschappelijke_naam"].map(_norm)
            elif "taxon" in ovr.columns:
                ovr["_key"] = ovr["taxon"].map(_norm)
            else:
                print("[WARN] overrides mist kolom 'wetenschappelijke_naam' of 'taxon' — overslaan")
                ovr = None
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

    # Reorder & write
    for c in APP_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[APP_COLS]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # Programmeervriendelijke export (punt als decimaal)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    # Excel NL‑vriendelijke export (komma als decimaal)
    out.to_csv(OUT_CSV_SEMI, index=False, encoding="utf-8", sep=";", decimal=",")

    print("[OK] Geschreven: " + OUT_CSV + "  (" + str(len(out)) + " rijen)")
    print("[OK] Geschreven: " + OUT_CSV_SEMI + "  (" + str(len(out)) + " rijen)")

    return out


if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as e:
        print("[ERR]", e)
        sys.exit(2)