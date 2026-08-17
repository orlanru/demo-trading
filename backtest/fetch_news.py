"""Descarga el tono histórico de noticias de GDELT para cada activo.

GDELT DOC 2.0 es la única fuente gratuita que permite BACKTESTING de sentimiento
de noticias: publica el tono medio diario de la cobertura que casa con una
consulta, desde 2017. Scrapear titulares de hoy (Finviz, buscadores) sirve para
operar en vivo, pero no para medir si la idea funcionó en el pasado.

Limita a 1 petición cada 5 s por IP, así que esto va despacio y reintenta.

Uso:
    python -m backtest.fetch_news
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

from backtest.news import NEWS_DIR, QUERIES

START, END = "20170101000000", "20260801000000"
MODES = ["timelinetone", "timelinevol"]


def fetch(query: str, mode: str, tries: int = 8) -> dict | None:
    url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(query)}"
           f"&mode={mode}&startdatetime={START}&enddatetime={END}&format=json")
    for intento in range(tries):
        r = subprocess.run(["curl", "-sS", "--max-time", "120", url],
                           capture_output=True, text=True)
        txt = r.stdout.strip()
        if txt.startswith("{"):
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                pass
        espera = 15 + 15 * intento
        print(f"    reintento {intento + 1}/{tries} en {espera}s", flush=True)
        time.sleep(espera)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(QUERIES))
    ap.add_argument("--pause", type=float, default=12.0)
    args = ap.parse_args()

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    for tk in args.tickers.split(","):
        if tk not in QUERIES:
            print(f"{tk}: sin consulta definida en backtest/news.py, se omite")
            continue
        for mode in MODES:
            path = NEWS_DIR / f"{tk}_{mode}.json"
            if path.exists():
                print(f"{tk} {mode}: ya está")
                continue
            print(f"{tk} {mode}: pidiendo...", flush=True)
            d = fetch(QUERIES[tk], mode)
            if d is None:
                print(f"{tk} {mode}: FALLO")
                continue
            tl = d.get("timeline", [])
            n = len(tl[0]["data"]) if tl else 0
            json.dump(d, open(path, "w"))
            print(f"{tk} {mode}: {n} puntos")
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
