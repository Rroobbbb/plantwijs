# treeebb_scraper.py
# Scraper voor Boomkwekerij Ebben – TreeEbb
# Haalt per plant: grondsoorten, winterhardheidszone, hoogte en breedte
# Schrijft naar: C:\PlantWijs\data\treeebb_planten.csv
#
# CSV-output:
# - delimiter: ';'  (voor Excel NL)
# - encoding: 'utf-8-sig' (met BOM, zodat tekens goed worden weergegeven)
# - grondsoorten samengevoegd met ' | ' binnen één cel

import os
import re
import time
import csv
import random
import pathlib
import argparse
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager


# ====== Instellingen ======
OUTPUT_DIR = r"C:\PlantWijs\data"  # <- standaardmap voor output
TREEEBB_START_URL = "https://www.ebben.nl/nl/treeebb/"
OUT_CSV = os.path.join(OUTPUT_DIR, "treeebb_planten.csv")
URLS_TXT = os.path.join(OUTPUT_DIR, "treeebb_urls.txt")

HEADLESS_DEFAULT = True   # zet op False als je de browser wil zien
REQUEST_DELAY = 1.0       # wachttijd tussen detail-requests (seconden)

# CSV-instellingen voor Excel NL
CSV_DELIM = ";"           # Excel NL verwacht ;
CSV_ENCODING = "utf-8-sig"  # met BOM
GROUND_JOIN = " | "       # binnen één cel

TREEEBB_PATH_FRAGMENT = "/treeebb/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TreeEbbScraper/1.3; +https://example.local)"
}

# Canonieke grondsoorten + herkenningspatronen
GROUND_PATTERNS = [
    (r"\blöss\b", "Löss"),
    (r"\bloess\b", "Löss"),
    (r"\bzavel\b", "Zavel"),
    (r"\bveen(grond)?\b", "Veen"),
    (r"\bzware\s*klei\b", "Zware klei"),
    (r"\blicht(e)?\s*klei\b", "Lichte klei"),
    (r"\bzand\b", "Zand"),
    (r"\blemige?\s*grond\b", "Leem"),
    (r"\bleem\b", "Leem"),
    (r"\bloam(y)?\b", "Leem"),
    (r"\balle\s*grondsoorten\b", "Alle grondsoorten"),
    (r"\ball\s*soil\s*types\b", "Alle grondsoorten"),
]


# ====== Selenium setup ======
def init_driver(headless: bool = True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,10000")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


def accept_cookies(driver):
    """Klik robuust verschillende cookietoestemmingen weg (best effort)."""
    keywords = [
        "Accepteer", "Accepteren", "Akkoord", "OK", "Oke", "Oké", "Alles toestaan",
        "Accept", "Allow", "Agree", "Continue", "Save", "Opslaan"
    ]
    try:
        time.sleep(1.0)
        for kw in keywords:
            xpath = f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{kw.lower()}')]"
            btns = driver.find_elements(By.XPATH, xpath)
            for b in btns:
                if b.is_displayed():
                    try:
                        b.click()
                        time.sleep(0.6)
                        return
                    except ElementClickInterceptedException:
                        pass
        # plan B: verberg mogelijke cookiebanners
        driver.execute_script("""
            const nodes = [...document.querySelectorAll('[class*="cookie"],[id*="cookie"]')];
            nodes.forEach(n => n.style.display = 'none');
        """)
    except Exception:
        pass  # niet kritisch


# ====== URL-detectie & verzamelen ======
def is_species_url(href: str) -> bool:
    if not href:
        return False
    try:
        u = urlparse(href)
    except Exception:
        return False
    if not u.netloc.endswith("ebben.nl"):
        return False
    if TREEEBB_PATH_FRAGMENT not in u.path:
        return False
    if "/pdf/" in u.path:
        return False
    last = u.path.rstrip("/").split("/")[-1]
    bad_parts = {"over-treeebb", "contact", "about-treeebb", "over-ebben"}
    if last in bad_parts:
        return False
    return "-" in last


def collect_species_urls(driver, max_scroll_rounds=90, sleep_s=1.2):
    driver.get(TREEEBB_START_URL)
    accept_cookies(driver)
    time.sleep(1.5)

    seen = set()
    stagnation_rounds = 0
    last_count = 0

    load_more_xpaths = [
        "//button[contains(., 'Meer') or contains(., 'meer') or contains(., 'Toon')]",
        "//a[contains(., 'Meer') or contains(., 'meer') or contains(., 'Toon')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'more')]",
    ]

    for _ in range(max_scroll_rounds):
        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href]")
        for a in anchors:
            href = a.get_attribute("href")
            if is_species_url(href):
                seen.add(href.split("#")[0])

        clicked = False
        for xp in load_more_xpaths:
            try:
                elems = driver.find_elements(By.XPATH, xp)
                for e in elems:
                    if e.is_displayed():
                        e.click()
                        clicked = True
                        time.sleep(0.8)
                        break
                if clicked:
                    break
            except Exception:
                pass

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_s + random.uniform(0.0, 0.7))

        if len(seen) == last_count:
            stagnation_rounds += 1
        else:
            stagnation_rounds = 0
            last_count = len(seen)

        if stagnation_rounds >= 5:
            break

    return sorted(seen)


# ====== Parsing helpers ======
def text_after_label(soup: BeautifulSoup, labels):
    if isinstance(labels, str):
        labels = [labels]
    for lab in labels:
        el = soup.find(string=re.compile(rf"^\s*{re.escape(lab)}\s*$", flags=re.I))
        if el:
            nxt = el.find_next(string=True)
            while nxt and (
                not nxt.strip() or re.match(
                    r"^\s*(Hoogte|Breedte|Winterhardheidszone|Winter hardiness zone|Grondsoort|Grondsoorten|Soil|Standplaats|Aspects|Eigenschappen|Characteristics)\s*$",
                    nxt, flags=re.I
                )
            ):
                nxt = nxt.find_next(string=True)
            if nxt:
                return " ".join(nxt.split())
    return None


def _collect_candidate_texts_for_soil(section):
    texts = []
    if not section:
        return texts
    for el in section.select(
        ".enumeration span, .enumeration li, .enumeration a, "
        "span.on, li.on, a.on, "
        "span, li, a, div"
    ):
        t = el.get_text(" ", strip=True)
        if t:
            texts.append(t)
    for sel in ['[title]', '[aria-label]', '[data-original-title]', 'img[alt]']:
        for el in section.select(sel):
            for attr in ('title', 'aria-label', 'data-original-title', 'alt'):
                val = el.get(attr)
                if val:
                    texts.append(val)
    return texts


def extract_ground_types(soup: BeautifulSoup):
    label_node = (
        soup.find(string=re.compile(r"^\s*Grondsoort(en)?\s*$", re.I))
        or soup.find(string=re.compile(r"^\s*Soil\s*$", re.I))
    )
    if not label_node:
        return []

    box = label_node.parent
    for _ in range(4):
        if not box:
            break
        texts = _collect_candidate_texts_for_soil(box)
        found = []
        for raw in texts:
            for pat, canon in GROUND_PATTERNS:
                if re.search(pat, raw, flags=re.I):
                    found.append(canon)
                    break
        if found:
            uniq, seen = [], set()
            for v in found:
                if v not in seen:
                    uniq.append(v)
                    seen.add(v)
            return uniq
        box = box.parent
    return []


def parse_species_page_html(html: str, url: str):
    soup = BeautifulSoup(html, "lxml")

    name = None
    h1 = soup.find("h1")
    if h1:
        name = " ".join(h1.get_text().split())
    if not name and soup.title and soup.title.string:
        name = soup.title.string.strip().split("|")[0].strip()

    hoogte = text_after_label(soup, ["Hoogte", "Height"])
    breedte = text_after_label(soup, ["Breedte", "Width"])
    winter = text_after_label(soup, ["Winterhardheidszone", "Winter hardiness zone"])
    grondsoorten = extract_ground_types(soup)

    return {
        "naam": name or "",
        "url": url,
        "hoogte": hoogte or "",
        "breedte": breedte or "",
        "winterhardheidszone": winter or "",
        "grondsoorten": GROUND_JOIN.join(grondsoorten) if grondsoorten else "",
    }


def parse_species_page_requests(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return parse_species_page_html(r.text, url)


def parse_species_page_selenium(driver, url: str):
    driver.get(url)
    time.sleep(1.0)
    accept_cookies(driver)
    time.sleep(0.5)
    return parse_species_page_html(driver.page_source, url)


# ====== Hoofdlogica ======
def ensure_parent_dir(path_str: str):
    pathlib.Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def main(
    headless=True,
    max_plants=None,
    delay=REQUEST_DELAY,
    resume=True,
    out_csv=OUT_CSV,
    use_detail_render_fallback=True,
    csv_delim=CSV_DELIM,
    csv_encoding=CSV_ENCODING,
):
    ensure_parent_dir(out_csv)
    ensure_parent_dir(URLS_TXT)

    if resume and pathlib.Path(URLS_TXT).exists():
        urls = [line.strip() for line in pathlib.Path(URLS_TXT).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"[info] {len(urls)} URL's geladen uit {URLS_TXT}")
    else:
        driver = init_driver(headless=headless)
        try:
            print("[info] Soort-URL's verzamelen (dit kan even duren)...")
            urls = collect_species_urls(driver)
            print(f"[ok] {len(urls)} soortpagina's gevonden")
            pathlib.Path(URLS_TXT).write_text("\n".join(urls), encoding="utf-8")
            print(f"[ok] URL-lijst opgeslagen in {URLS_TXT}")
        finally:
            driver.quit()

    if max_plants:
        urls = urls[:max_plants]

    rows = []
    detail_driver = None
    for i, u in enumerate(urls, 1):
        try:
            data = parse_species_page_requests(u)
            if use_detail_render_fallback and not data["grondsoorten"]:
                if detail_driver is None:
                    detail_driver = init_driver(headless=headless)
                data = parse_species_page_selenium(detail_driver, u)
            rows.append(data)
            print(f"[{i}/{len(urls)}] {data['naam']}")
        except Exception as e:
            print(f"[fout] {u} -> {e}")
        time.sleep(delay + random.uniform(0.0, 0.6))

    if detail_driver:
        detail_driver.quit()

    fieldnames = ["naam", "url", "hoogte", "breedte", "winterhardheidszone", "grondsoorten"]
    with open(out_csv, "w", newline="", encoding=csv_encoding) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=csv_delim,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[klaar] {len(rows)} records opgeslagen in {out_csv} (delimiter='{csv_delim}', encoding='{csv_encoding}')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TreeEbb scraper (grondsoorten, winterhardheidszone, hoogte, breedte)."
    )
    ap.add_argument("--no-headless", action="store_true", help="Toon browservenster (standaard headless).")
    ap.add_argument("--max", type=int, default=None, help="Maximaal aantal planten (voor testen).")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Pauze (s) tussen detail-requests.")
    ap.add_argument("--fresh", action="store_true", help="Negeer bestaande URL-cache en verzamel opnieuw.")
    ap.add_argument("--no-render-fallback", action="store_true",
                    help="Gebruik géén Selenium-fallback op detailpagina's.")
    ap.add_argument("--out", type=str, default=OUT_CSV, help=f"Pad/naam van output CSV (default: {OUT_CSV})")
    ap.add_argument("--sep", type=str, default=CSV_DELIM, help="CSV delimiter (default ';').")
    ap.add_argument("--encoding", type=str, default=CSV_ENCODING, help="CSV encoding (default 'utf-8-sig').")

    args = ap.parse_args()

    main(
        headless=not args.no_headless,
        max_plants=args.max,
        delay=args.delay,
        resume=not args.fresh,
        out_csv=args.out,
        use_detail_render_fallback=not args.no_render_fallback,
        csv_delim=args.sep,
        csv_encoding=args.encoding,
    )
