@echo off
setlocal
title TreeEbb Scraper (eenvoudige start)

rem === Basislocaties ===
set BASE=C:\PlantWijs
set DATA=%BASE%\data
if not exist "%DATA%" mkdir "%DATA%"

rem === Check/installeer Google Chrome (nodig voor Selenium) ===
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" goto chrome_ok
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" goto chrome_ok
echo [info] Google Chrome niet gevonden. Probeer te installeren via winget...
winget install -e --id Google.Chrome
:chrome_ok

rem === Kies python launcher: 'py' (Windows) of 'python' ===
where py >nul 2>&1
if %errorlevel%==0 (set PY=py) else (set PY=python)

rem === Check/installeer Python als nodig ===
where %PY% >nul 2>&1
if %errorlevel% neq 0 (
  echo [info] Python niet gevonden. Probeer te installeren via winget...
  winget install -e --id Python.Python.3.12
  set "PATH=%LocalAppData%\Microsoft\WindowsApps;%PATH%"
)

rem === Virtuele omgeving + vereisten ===
cd /d "%BASE%"
%PY% -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install requests beautifulsoup4 lxml selenium webdriver-manager

rem === Scraper draaien (standaard onzichtbare browser) ===
echo [info] Scraper starten...
python "%BASE%\treeebb_scraper.py"

echo.
echo [klaar] Als alles goed ging staat je bestand hier:
echo         %DATA%\treeebb_planten.csv
if exist "%DATA%\treeebb_planten.csv" start "" "%DATA%\treeebb_planten.csv"
pause
