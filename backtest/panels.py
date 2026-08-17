"""Carga de paneles mensuales y comparación vectorizada en ventanas.

El motor de `crosssec.py` recorre mes a mes y es demasiado lento para barrer
cientos de ventanas por señal y universo. Esto hace exactamente lo mismo con
álgebra matricial; `test_panels.py` verifica que ambos coinciden.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.crosssec import Panel

MONTHLY_DIR = Path(__file__).resolve().parent.parent / "data" / "monthly"

SECTORS = ["XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "XLI", "XLY", "XLB"]

UNIVERSES = {
    "cesta_usuario": ["SPY", "QQQ", "IWM", "GLD", "SLV", "NVDA", "MSFT"],
    "cesta_sin_nvda": ["SPY", "QQQ", "IWM", "GLD", "SLV", "MSFT"],
    "sectores": SECTORS + ["SPY"],
    "amplio": SECTORS + ["SPY", "QQQ", "IWM", "GLD", "TLT", "EFA"],
}


def load_panel(tickers: list[str], start: str | None = None) -> Panel:
    """Panel mensual: se compra en la apertura del mes, se valora al cierre."""
    buys, closes = {}, {}
    for t in tickers:
        path = MONTHLY_DIR / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}. Ejecuta: python -m backtest.fetch_data")
        d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        buys[t] = d["Open"]
        closes[t] = d["Close"]
    buy = pd.DataFrame(buys).dropna()
    close = pd.DataFrame(closes).dropna()
    idx = buy.index.intersection(close.index)
    if start:
        idx = idx[idx >= start]
    return Panel(buy=buy.loc[idx], close=close.loc[idx])


def market_drawdown(ticker: str = "SPY") -> pd.Series:
    """Caída del mercado desde su máximo histórico, mensual."""
    d = pd.read_csv(MONTHLY_DIR / f"{ticker}.csv", parse_dates=["Date"]).set_index("Date")
    c = d["Close"].sort_index()
    return c / c.cummax() - 1


def picks_from_scores(panel: Panel, scores: pd.DataFrame) -> np.ndarray:
    """Índice del activo elegido cada mes (-1 = sin señal -> reparto equiponderado).

    Usa SIEMPRE la señal del mes anterior: nunca hay lookahead.
    """
    cols = list(panel.close.columns)
    sc = scores.reindex(index=panel.months, columns=cols)
    out = np.full(len(panel.months), -1, dtype=int)
    for i in range(1, len(panel.months)):
        row = sc.iloc[i - 1].dropna()
        if row.empty or row.nunique() == 1:
            continue
        out[i] = cols.index(row.idxmax())
    return out


def window_table(panel: Panel, scores: pd.DataFrame, window: int = 24,
                 contribution: float = 1000.0) -> pd.DataFrame:
    """Una fila por ventana de `window` aportaciones mensuales.

    Compara la estrategia contra el DCA equiponderado y contra el DCA en cada
    activo por separado (de donde salen mediana, mejor, peor y percentil).
    """
    buy = panel.buy.to_numpy(float)
    close = panel.close.to_numpy(float)
    M, A = buy.shape
    picks = picks_from_scores(panel, scores)

    units = contribution / buy                     # participaciones por aportación
    strat_units = np.zeros((M, A))
    for m in range(M):
        if picks[m] < 0:
            strat_units[m] = units[m] / A
        else:
            strat_units[m, picks[m]] = units[m, picks[m]]
    ew_units = units / A

    rows = []
    for s in range(0, M - window + 1):
        e = s + window - 1
        px = close[e]
        inv = contribution * window
        strat = float(strat_units[s:e + 1].sum(0) @ px) / inv - 1
        ew = float(ew_units[s:e + 1].sum(0) @ px) / inv - 1
        singles = (units[s:e + 1].sum(0) * px) / inv - 1
        rows.append({
            "inicio": panel.months[s], "fin": panel.months[e],
            "estrategia": strat, "equipond": ew,
            "mediana": float(np.median(singles)),
            "mejor": float(singles.max()), "peor": float(singles.min()),
            "percentil": float((singles < strat).mean()),
        })
    return pd.DataFrame(rows)


def window_table_weights(panel: Panel, weights: pd.DataFrame, window: int = 24,
                         contribution: float = 1000.0) -> pd.DataFrame:
    """Como window_table, pero repartiendo cada aportación con pesos arbitrarios.

    Sirve para reglas de REPARTO (volatilidad inversa, paridad de riesgo), que no
    eligen un activo sino que dosifican entre todos. Los pesos del mes t-1 son
    los que se aplican a la compra del mes t.
    """
    cols = list(panel.close.columns)
    w = weights.reindex(index=panel.months, columns=cols)
    buy = panel.buy.to_numpy(float)
    close = panel.close.to_numpy(float)
    M, A = buy.shape

    units = contribution / buy
    strat_units = np.zeros((M, A))
    for m in range(M):
        row = w.iloc[m - 1] if m > 0 else pd.Series(np.nan, index=cols)
        row = row.fillna(0.0)
        s = row.sum()
        pesos = (row / s).to_numpy() if s > 0 else np.full(A, 1.0 / A)
        strat_units[m] = units[m] * pesos
    ew_units = units / A

    rows = []
    for s in range(0, M - window + 1):
        e = s + window - 1
        px = close[e]
        inv = contribution * window
        estrategia = float(strat_units[s:e + 1].sum(0) @ px) / inv - 1
        ew = float(ew_units[s:e + 1].sum(0) @ px) / inv - 1
        singles = (units[s:e + 1].sum(0) * px) / inv - 1
        rows.append({
            "inicio": panel.months[s], "fin": panel.months[e],
            "estrategia": estrategia, "equipond": ew,
            "mediana": float(np.median(singles)),
            "percentil": float((singles < estrategia).mean()),
        })
    return pd.DataFrame(rows)


def random_band(panel: Panel, window: int = 24, n: int = 200, seed: int = 0) -> dict:
    """Distribución de resultados eligiendo al azar. Cualquier señal que caiga
    dentro de esta banda no aporta información."""
    rng = np.random.default_rng(seed)
    cols = panel.close.columns
    diffs, wins, pcts = [], [], []
    for _ in range(n):
        noise = pd.DataFrame(rng.standard_normal((len(panel.months), len(cols))),
                             index=panel.months, columns=cols)
        t = window_table(panel, noise, window=window)
        d = t["estrategia"] - t["equipond"]
        diffs.append(100 * d.mean())
        wins.append(100 * (d > 0).mean())
        pcts.append(100 * t["percentil"].mean())
    return {
        "vs_equipond_medio": float(np.mean(diffs)),
        "p5": float(np.percentile(diffs, 5)),
        "p95": float(np.percentile(diffs, 95)),
        "gana_equipond": float(np.mean(wins)),
        "percentil_medio": float(np.mean(pcts)),
    }
