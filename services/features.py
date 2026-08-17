import numpy as np
import pandas as pd

from services.market_data import get_daily_history


def build_monthly_features(dfd: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el DataFrame de features mensuales a partir de OHLC diario.

    Función pura (sin red ni Streamlit) para que el backtest use exactamente
    las mismas features que la app.
    """
    dfd = dfd.copy()

    # Features mensuales base
    m = pd.DataFrame(index=dfd["Close"].resample("ME").last().index)
    m["High"] = dfd["High"].resample("ME").max()
    m["Low"] = dfd["Low"].resample("ME").min()
    m["Close"] = dfd["Close"].resample("ME").last()

    # Returns / vol
    dfd["ret"] = dfd["Close"].pct_change()
    dfd["logret"] = np.log(dfd["Close"]).diff()
    m["RV_1m"] = dfd["logret"].resample("ME").std()

    # Range / down days
    daily_range = (dfd["High"] - dfd["Low"]) / dfd["Close"]
    m["RangeMean_1m"] = daily_range.resample("ME").mean()
    m["RangeStd_1m"] = daily_range.resample("ME").std()
    m["DownDaysFrac_1m"] = (dfd["ret"] < 0).resample("ME").mean()

    # Max drawdown mensual
    def max_dd(s):
        rm = s.cummax()
        dd = (s / rm) - 1
        return float(dd.min()) if len(dd) > 0 else 0.0

    m["MaxDD_1m"] = dfd["Close"].resample("ME").apply(max_dd)

    # Retornos multi-horizonte
    for p in [1, 3, 6, 12]:
        m[f"Retorno_{p}m"] = m["Close"].pct_change(p)

    # SMA/Dist
    m["SMA_12"] = m["Close"].rolling(12).mean()
    m["Dist_SMA12"] = (m["Close"] - m["SMA_12"]) / m["SMA_12"]

    # RSI 14 (sobre cierres mensuales)
    delta = m["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    m["RSI_14"] = 100 - (100 / (1 + rs))

    # Range pct mensual
    m["Range_Pct"] = (m["High"] - m["Low"]) / m["Close"]

    # Estacionalidad
    m["Mes"] = m.index.month
    m["Mes_sin"] = np.sin(2 * np.pi * m["Mes"] / 12)
    m["Mes_cos"] = np.cos(2 * np.pi * m["Mes"] / 12)

    return m


def get_features_complete(ticker: str, features_list: list[str], n_months: int = 3):
    """
    Genera features mensuales y devuelve 'ventanas' para predecir:
      - mes calendario actual
      - mes -1
      - mes -2
    usando como input siempre el último mes CERRADO previo a cada mes objetivo.

    Devuelve:
      (windows, df_daily)
    donde windows es list[dict]:
      {
        "label": "YYYY-MM",
        "x_input": X_final (1 row),
        "prev_close": float,
        "month_start": Timestamp,
        "month_end": Timestamp
      }
    """
    start_date = "2015-01-01" if "BTC" in ticker else "2000-01-01"
    dfd = get_daily_history(ticker, start_date=start_date)

    m = build_monthly_features(dfd)
    m_clean = m.dropna()
    if m_clean.empty or len(m_clean.index) < (n_months + 3):
        return None, dfd

    # =========================================================
    # CLAVE: anclar el mes objetivo al mes calendario actual,
    # aunque todavía no existan velas del mes en el dataset.
    # =========================================================
    today = pd.Timestamp.today().normalize()
    this_month_start = today.replace(day=1)

    windows: list[dict] = []

    for k in range(n_months):
        # Mes objetivo: actual, -1, -2
        month_start = (this_month_start - pd.DateOffset(months=k)).normalize()
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()

        # Features: último mes cerrado antes del mes objetivo
        feat_end = (month_start - pd.offsets.MonthEnd(1)).normalize()

        if feat_end not in m_clean.index:
            # dataset corto / faltan datos
            continue

        try:
            X_final = m_clean.loc[[feat_end], features_list]
        except KeyError as e:
            import streamlit as st

            st.error(f"Error de columnas: {e}. El modelo pide columnas que no se generaron.")
            return None, dfd

        prev_close = float(m_clean.loc[feat_end, "Close"])
        label = month_start.strftime("%Y-%m")

        windows.append(
            {
                "label": label,
                "x_input": X_final,
                "prev_close": prev_close,
                "month_start": month_start,
                "month_end": month_end,
            }
        )

    if not windows:
        return None, dfd

    return windows, dfd
