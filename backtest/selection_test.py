"""¿La señal elige activos mejores que la media del universo?

Las ventanas solapadas de 24 meses se solapan tanto que 98 ventanas equivalen a
~4 observaciones independientes: sirven para ilustrar, no para concluir. Este
test mira directamente la calidad de la selección mes a mes, que es donde hay
grados de libertad reales.

Para cada mes: retorno futuro del activo elegido − retorno medio del universo.
Si la señal no tiene información, esa serie tiene media cero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.crosssec import Panel


def selection_edge(panel: Panel, scores: pd.DataFrame, horizon: int = 1,
                   top_k: int = 1) -> pd.Series:
    """Serie de exceso de retorno del pick sobre la media equiponderada."""
    close = panel.close
    fwd = close.shift(-horizon) / close - 1  # retorno futuro a `horizon` meses
    rows = {}
    for i, ts in enumerate(close.index):
        if i + horizon >= len(close):
            break
        row = scores.loc[ts].dropna() if ts in scores.index else pd.Series(dtype=float)
        if row.empty or row.nunique() == 1:
            continue
        picks = list(row.sort_values(ascending=False).index[:top_k])
        f = fwd.loc[ts].dropna()
        if f.empty:
            continue
        rows[ts] = float(f[picks].mean() - f.mean())
    return pd.Series(rows).sort_index()


def newey_west_t(x: np.ndarray, lags: int) -> float:
    """t de la media con errores estándar Newey-West.

    Con horizonte > 1 mes las observaciones se solapan y autocorrelacionan; el
    t clásico sobreestima la significancia. Newey-West lo corrige.
    """
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    gamma0 = float(e @ e / n)
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        cov = float(e[l:] @ e[:-l] / n)
        var += 2 * (1 - l / (lags + 1)) * cov
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def summarize(edge: pd.Series, name: str, horizon: int) -> dict:
    x = edge.to_numpy(dtype=float)
    return {
        "señal": name,
        "horizonte": f"{horizon}m",
        "n": len(x),
        "edge medio %": 100 * float(np.mean(x)) if len(x) else np.nan,
        "t (Newey-West)": newey_west_t(x, lags=horizon),
        "% aciertos": 100 * float(np.mean(x > 0)) if len(x) else np.nan,
    }
