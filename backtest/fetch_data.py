"""Descarga histórico largo ajustado por dividendos y lo guarda mensualizado.

yfinance usa curl_cffi, que no atraviesa proxies con CA propia; aquí se llama a
la API de Yahoo con `curl`, que sí funciona en cualquier entorno.

Uso:
    python -m backtest.fetch_data                 # universo por defecto
    python -m backtest.fetch_data --tickers SPY,QQQ
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "monthly"

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "SLV", "NVDA", "MSFT",       # cesta original
    "DIA", "TLT", "EFA", "AAPL",                              # extras
    "XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "XLI", "XLY", "XLB",  # sectores
]


def fetch_daily(ticker: str, tries: int = 4) -> pd.DataFrame | None:
    """OHLC diario ajustado por dividendos y splits."""
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1=630720000&period2=9999999999&interval=1d&events=div%2Csplit")
    for attempt in range(tries):
        r = subprocess.run(
            ["curl", "-sS", "--max-time", "90", "-H", f"User-Agent: {UA}", url],
            capture_output=True, text=True,
        )
        try:
            res = json.loads(r.stdout)["chart"]["result"]
        except Exception:
            res = None
        if res:
            break
        time.sleep(2 ** attempt)
    else:
        return None
    if not res:
        return None

    res = res[0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"],
    })
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is not None:
        df["Adj"] = adj
        ratio = df["Adj"] / df["Close"]
        for c in ("Open", "High", "Low"):
            df[c] = df[c] * ratio
        df["Close"] = df["Adj"]
        df = df.drop(columns=["Adj"])
    return df.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")


def to_monthly(dfd: pd.DataFrame) -> pd.DataFrame:
    g = dfd.resample("ME")
    m = pd.DataFrame({
        "Open": g["Open"].first(),      # precio de compra del mes
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
        "Sessions": g["Close"].count(),
    })
    return m[m["Sessions"] > 0].round(6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--pause", type=float, default=1.2, help="segundos entre peticiones")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fallos = []
    for t in args.tickers.split(","):
        d = fetch_daily(t)
        if d is None or d.empty:
            fallos.append(t)
            print(f"{t:5s} FALLO")
            continue
        m = to_monthly(d)
        m.to_csv(OUT_DIR / f"{t}.csv")
        print(f"{t:5s} {len(m):4d} meses  {m.index.min():%Y-%m} → {m.index.max():%Y-%m}")
        time.sleep(args.pause)

    if fallos:
        print(f"\nNo descargados: {', '.join(fallos)}")


if __name__ == "__main__":
    main()
