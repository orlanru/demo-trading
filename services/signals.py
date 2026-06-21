"""
Señales mensuales para asignación dinámica (Smart-DCA).

Todo se calcula a frecuencia MENSUAL y de forma causal: la señal del mes M
usa únicamente información disponible al cierre del mes anterior (M-1). Así el
backtest no tiene look-ahead.

La filosofía es transparente y basada en reglas (no en un .pkl entrenado):
  - Mean-reversion: cuando el activo está caído respecto a su máximo reciente
    y/o sobrevendido (RSI bajo), el retorno forward esperado es mayor → comprar
    MÁS.
  - Cuando está cerca de máximos / sobrecomprado → comprar MENOS y guardar
    "dry powder" (pólvora seca) para los sustos.
  - Filtro de tendencia (SMA): señal binaria de régimen para la estrategia de
    control de drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un DataFrame mensual (índice = fin de mes) con las columnas que
    necesitan las señales. Espera un daily con columnas High/Low/Close.
    """
    close = daily["Close"].resample("ME").last()
    m = pd.DataFrame(index=close.index)
    m["Close"] = close
    m["High"] = daily["High"].resample("ME").max()
    m["Low"] = daily["Low"].resample("ME").min()

    # Retorno mensual
    m["ret_1m"] = m["Close"].pct_change()

    # Volatilidad realizada (std de retornos mensuales, ventana 12m)
    m["vol_12m"] = m["ret_1m"].rolling(12).std()

    # Medias y distancia a la media de 12 meses
    m["SMA_12"] = m["Close"].rolling(12).mean()
    m["dist_SMA12"] = (m["Close"] - m["SMA_12"]) / m["SMA_12"]

    # Drawdown respecto al máximo de cierres de los últimos 12 meses
    roll_max = m["Close"].rolling(12, min_periods=1).max()
    m["drawdown_12m"] = (m["Close"] / roll_max) - 1.0

    # Momentum 12m
    m["mom_12m"] = m["Close"].pct_change(12)

    # RSI(14) sobre cierres mensuales
    delta = m["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    m["RSI_14"] = 100 - (100 / (1 + rs))

    # Régimen de tendencia: cierre por encima de su SMA de 12m
    m["trend_up"] = (m["Close"] > m["SMA_12"]).astype(float)

    return m


def smart_multiplier(
    row: pd.Series,
    base: float = 1.0,
    dd_gain: float = 6.0,
    rsi_gain: float = 0.025,
    lo: float = 0.30,
    hi: float = 3.0,
) -> float:
    """
    Multiplicador de aportación para Smart-DCA a partir de las señales del mes
    anterior (row). Rango acotado a [lo, hi].

    Lógica (mean-reversion, sin parámetros ocultos):
      - Cuanto más profundo el drawdown a 12m, mayor el multiplicador.
        Un drawdown de -20% con dd_gain=6 aporta +1.2 al multiplicador.
      - RSI por debajo de 50 añade; por encima, resta. (50 - RSI) * rsi_gain.
        RSI=30 -> +0.5 ; RSI=70 -> -0.5.
      - Si no hay datos suficientes (NaN), multiplicador neutro = 1.0.
    """
    dd = row.get("drawdown_12m", np.nan)
    rsi = row.get("RSI_14", np.nan)

    mult = base
    if pd.notna(dd):
        mult += dd_gain * abs(min(dd, 0.0))
    if pd.notna(rsi):
        mult += (50.0 - rsi) * rsi_gain

    if not np.isfinite(mult):
        return 1.0
    return float(np.clip(mult, lo, hi))
