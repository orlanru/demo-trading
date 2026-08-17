"""Carga de OHLC diario para el backtest.

Intenta yfinance (misma fuente que la app). Si falla (rate limit, sin red),
cae a un CSV cacheado en `data/cache/{TICKER}.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _from_cache(ticker: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close"]].dropna()


def _from_yfinance(ticker: str, start_date: str) -> pd.DataFrame | None:
    try:
        from services.market_data import get_daily_history

        df = get_daily_history(ticker, start_date=start_date)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df[["Open", "High", "Low", "Close"]].dropna()


def load_ohlc(ticker: str, start_date: str = "2000-01-01", prefer_cache: bool = False) -> pd.DataFrame:
    """Devuelve OHLC diario con índice de fechas.

    prefer_cache=True evita la llamada de red (útil en CI o con rate limits).
    """
    loaders = [_from_cache, lambda t: _from_yfinance(t, start_date)]
    if not prefer_cache:
        loaders.reverse()

    for loader in loaders:
        df = loader(ticker)
        if df is not None and not df.empty:
            return df.loc[df.index >= start_date]

    raise RuntimeError(
        f"Sin datos para {ticker}: yfinance falló y no hay cache en {CACHE_DIR}/{ticker}.csv"
    )


def monthly_ohlc(dfd: pd.DataFrame) -> pd.DataFrame:
    """Agrega el diario a mensual conservando lo que necesita el simulador:

    first_open  -> precio de la primera sesión del mes (entrada del DCA)
    low/high    -> extremos del mes (para saber si una orden límite se ejecuta)
    close       -> cierre del mes (fallback y valoración)
    """
    g = dfd.resample("ME")
    m = pd.DataFrame(
        {
            "first_open": g["Open"].first(),
            "high": g["High"].max(),
            "low": g["Low"].min(),
            "close": g["Close"].last(),
            "sessions": g["Close"].count(),
        }
    )
    return m[m["sessions"] > 0]
