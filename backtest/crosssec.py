"""DCA con selección transversal: cada mes se aporta lo mismo, pero se elige
**en qué activo** del universo entra el dinero.

Comparaciones:
  - DCA en cada activo por separado (y su mediana)
  - DCA equiponderado en todo el universo  <- el benchmark honesto: mismo
    universo, misma aportación, cero habilidad
  - la estrategia con señal
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.data import load_ohlc

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "SLV", "NVDA", "MSFT"]


@dataclass
class Panel:
    buy: pd.DataFrame    # precio de compra: apertura de la 1ª sesión del mes
    close: pd.DataFrame  # cierre del mes (para señales y valoración)

    @property
    def months(self) -> pd.DatetimeIndex:
        return self.buy.index


def load_panel(tickers: list[str], start_date: str = "2000-01-01", offline: bool = True) -> Panel:
    buys, closes = {}, {}
    for t in tickers:
        d = load_ohlc(t, start_date=start_date, prefer_cache=offline)
        g = d.resample("ME")
        buys[t] = g["Open"].first()
        closes[t] = g["Close"].last()
    buy = pd.DataFrame(buys).dropna()
    close = pd.DataFrame(closes).dropna()
    idx = buy.index.intersection(close.index)
    return Panel(buy=buy.loc[idx], close=close.loc[idx])


def simulate(
    panel: Panel,
    scores: pd.DataFrame,
    top_k: int = 1,
    contribution: float = 1000.0,
    months: pd.DatetimeIndex | None = None,
) -> dict:
    """Cada mes reparte `contribution` entre los `top_k` activos mejor puntuados.

    Las puntuaciones del mes t-1 deciden la compra en la apertura del mes t
    (sin lookahead). Meses sin puntuación válida -> reparto equiponderado.
    """
    idx = panel.months if months is None else months
    shares = pd.Series(0.0, index=panel.buy.columns)
    picks = []

    for ts in idx:
        prev = scores.index[scores.index < ts]
        row = scores.loc[prev[-1]].dropna() if len(prev) else pd.Series(dtype=float)

        if row.empty:
            chosen = list(panel.buy.columns)
        else:
            # empates (p. ej. señal "equal") -> se reparte entre todos
            if row.nunique() == 1:
                chosen = list(row.index)
            else:
                chosen = list(row.sort_values(ascending=False).index[:top_k])

        each = contribution / len(chosen)
        for c in chosen:
            shares[c] += each / panel.buy.loc[ts, c]
        picks.append({"month": ts, "picks": ",".join(chosen)})

    final = float((shares * panel.close.loc[idx[-1]]).sum())
    contributed = contribution * len(idx)
    return {
        "shares": shares,
        "final_value": final,
        "contributed": contributed,
        "total_return": final / contributed - 1,
        "irr": _irr_monthly(contribution, len(idx), final),
        "picks": pd.DataFrame(picks).set_index("month"),
    }


def dca_single(panel: Panel, ticker: str, contribution: float = 1000.0,
               months: pd.DatetimeIndex | None = None) -> dict:
    idx = panel.months if months is None else months
    sh = (contribution / panel.buy.loc[idx, ticker]).sum()
    final = float(sh * panel.close.loc[idx[-1], ticker])
    contributed = contribution * len(idx)
    return {
        "final_value": final,
        "contributed": contributed,
        "total_return": final / contributed - 1,
        "irr": _irr_monthly(contribution, len(idx), final),
    }


def _irr_monthly(contribution: float, n: int, final_value: float) -> float:
    """TIR anualizada de n aportaciones iguales que acaban valiendo final_value."""
    def fv(r: float) -> float:
        return sum(contribution * (1 + r) ** (n - 1 - i) for i in range(n))

    lo, hi = -0.9, 1.0
    while fv(hi) < final_value and hi < 1e3:
        hi *= 2
    if fv(lo) > final_value:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if fv(mid) < final_value:
            lo = mid
        else:
            hi = mid
    return (1 + (lo + hi) / 2) ** 12 - 1


def rolling_windows(panel: Panel, window: int, step: int = 1) -> list[pd.DatetimeIndex]:
    """Todas las ventanas solapadas de `window` meses (p. ej. 24 = 2 años)."""
    idx = panel.months
    return [idx[i:i + window] for i in range(0, len(idx) - window + 1, step)]


def evaluate(
    panel: Panel,
    scores: pd.DataFrame,
    window: int = 24,
    top_k: int = 1,
    contribution: float = 1000.0,
) -> pd.DataFrame:
    """Evalúa la estrategia en TODAS las ventanas solapadas de `window` meses.

    Una sola ventana es anécdota; la distribución sobre ~100 ventanas es
    evidencia. Devuelve una fila por ventana.
    """
    rows = []
    tickers = list(panel.buy.columns)
    ew_scores = pd.DataFrame(0.0, index=panel.months, columns=tickers)
    for w in rolling_windows(panel, window):
        strat = simulate(panel, scores, top_k=top_k, contribution=contribution, months=w)
        singles = {t: dca_single(panel, t, contribution, months=w)["total_return"] for t in tickers}
        ew = simulate(panel, ew_scores, top_k=len(tickers), contribution=contribution, months=w)

        vals = np.array(list(singles.values()))
        rows.append(
            {
                "inicio": w[0],
                "fin": w[-1],
                "estrategia": strat["total_return"],
                "equiponderado": ew["total_return"],
                "mediana_dca": float(np.median(vals)),
                "mejor_dca": float(vals.max()),
                "peor_dca": float(vals.min()),
                # percentil de la estrategia dentro de los DCA individuales
                "percentil": float((vals < strat["total_return"]).mean()),
                "vs_equipond_pp": 100 * (strat["total_return"] - ew["total_return"]),
                "vs_mediana_pp": 100 * (strat["total_return"] - float(np.median(vals))),
                **{f"dca_{t}": singles[t] for t in tickers},
            }
        )
    return pd.DataFrame(rows).set_index("inicio")


def summarize_eval(df: pd.DataFrame, name: str) -> dict:
    return {
        "señal": name,
        "ventanas": len(df),
        "ret. medio %": 100 * df["estrategia"].mean(),
        "equipond. medio %": 100 * df["equiponderado"].mean(),
        "vs equipond. (pp)": df["vs_equipond_pp"].mean(),
        "% gana a equipond.": 100 * (df["vs_equipond_pp"] > 0).mean(),
        "vs mediana (pp)": df["vs_mediana_pp"].mean(),
        "% gana a mediana": 100 * (df["vs_mediana_pp"] > 0).mean(),
        "percentil medio": 100 * df["percentil"].mean(),
    }
