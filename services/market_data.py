import datetime
import pandas as pd
import yfinance as yf

def _fix_yfinance_columns(dfd: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Arregla MultiIndex si yfinance devuelve columnas multi."""
    if isinstance(dfd.columns, pd.MultiIndex):
        try:
            dfd = dfd.xs(ticker, axis=1, level=1)
        except Exception:
            dfd.columns = [c[0] for c in dfd.columns]
    return dfd

def get_daily_history(ticker: str, start_date: str) -> pd.DataFrame:
    dfd = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    dfd = _fix_yfinance_columns(dfd, ticker)
    dfd = dfd.sort_index().dropna()
    return dfd

def get_intraday_price(ticker: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    start_d = start - datetime.timedelta(days=2)
    end_d = end + datetime.timedelta(days=1)
    try:
        df = yf.download(ticker, start=start_d, end=end_d, interval="30m", progress=False, auto_adjust=True)
        if df.empty:
            df = yf.download(ticker, start=start_d, end=end_d, interval="1d", progress=False, auto_adjust=True)
        return df
    except Exception:
        return pd.DataFrame()
