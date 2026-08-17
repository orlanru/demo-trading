"""Señales transversales: puntúan cada activo del universo en cada mes.

Convención: **puntuación más alta = más preferido**. Todas las señales se
calculan con información disponible al cierre del mes t-1 y se usan para
decidir la compra en la apertura del mes t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trend_zscore(log_prices: np.ndarray) -> tuple[float, float]:
    """Ajusta una recta a log(precio) y devuelve (z del último residuo, pendiente).

    z < 0  -> el activo cotiza por DEBAJO de su línea de tendencia (barato)
    La pendiente es el crecimiento log mensual ajustado (calidad de la tendencia).
    """
    n = len(log_prices)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, log_prices, 1)
    resid = log_prices - (slope * x + intercept)
    sd = resid.std(ddof=2)
    if sd == 0 or not np.isfinite(sd):
        return 0.0, float(slope)
    return float(resid[-1] / sd), float(slope)


def trend_gap(panel: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """Distancia a la línea de tendencia log-lineal, en desviaciones típicas.

    Es la idea del usuario: comprar el que está más por debajo de su recta.
    Se devuelve **-z** para respetar la convención (más barato = más alto).
    """
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    logp = np.log(panel)
    for i in range(lookback, len(panel) + 1):
        ts = panel.index[i - 1]
        for col in panel.columns:
            w = logp[col].iloc[i - lookback:i].dropna()
            if len(w) < lookback:
                continue
            z, _ = _trend_zscore(w.to_numpy())
            out.loc[ts, col] = -z
    return out


def trend_gap_filtered(panel: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """Igual que trend_gap pero descarta activos con tendencia plana o bajista.

    Comprar barato sólo tiene sentido si la recta a la que revierte sube: si no,
    estás comprando un activo en declive estructural (value trap).
    """
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    logp = np.log(panel)
    for i in range(lookback, len(panel) + 1):
        ts = panel.index[i - 1]
        for col in panel.columns:
            w = logp[col].iloc[i - lookback:i].dropna()
            if len(w) < lookback:
                continue
            z, slope = _trend_zscore(w.to_numpy())
            out.loc[ts, col] = -z if slope > 0 else np.nan
    return out


def momentum_12_1(panel: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Momentum transversal clásico (Jegadeesh-Titman): retorno de t-12 a t-1,
    saltando el último mes. Compra los GANADORES: dirección opuesta a trend_gap.
    """
    return (panel.shift(skip) / panel.shift(lookback) - 1)


def reversal_1m(panel: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal: compra el que peor lo hizo el último mes."""
    return -(panel / panel.shift(1) - 1)


def drawdown_from_high(panel: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """Caída desde el máximo de los últimos `lookback` meses (más caído = mejor)."""
    return -(panel / panel.rolling(lookback).max() - 1)


def equal(panel: pd.DataFrame) -> pd.DataFrame:
    """Señal nula: todos empatados (sirve para la cartera equiponderada)."""
    return pd.DataFrame(0.0, index=panel.index, columns=panel.columns)


REGISTRY = {
    "trend_gap": trend_gap,
    "trend_gap_filtered": trend_gap_filtered,
    "momentum_12_1": momentum_12_1,
    "reversal_1m": reversal_1m,
    "drawdown": drawdown_from_high,
    "equal": equal,
}

# Qué ventana usa cada señal. Un único `--lookback` global sería un error:
# el momentum clásico es 12-1, no 36-1.
TREND_BASED = {"trend_gap", "trend_gap_filtered", "drawdown"}


def make_scores(name: str, panel: pd.DataFrame, trend_lookback: int = 36,
                mom_lookback: int = 12) -> pd.DataFrame:
    """Construye la matriz de puntuaciones de una señal con sus parámetros."""
    fn = REGISTRY[name]
    if name in TREND_BASED:
        return fn(panel, lookback=trend_lookback)
    if name == "momentum_12_1":
        return fn(panel, lookback=mom_lookback, skip=1)
    return fn(panel)
