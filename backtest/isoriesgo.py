"""Elegir la cartera más barata de entre las que tienen el mismo riesgo.

La idea: en vez de elegir UN activo, se generan muchas combinaciones long-only,
se calcula el riesgo ex-ante de cada una con la covarianza reciente, se queda uno
con las que caen en una banda estrecha de riesgo, y de entre ésas se elige la más
barata.

Lo elegante del diseño es que el control viene de serie: **las demás carteras de
la misma banda son exactamente el benchmark aleatorio a igual riesgo**. No hay
que construirlo aparte ni discutir si es comparable.

Sin lookahead: covarianza y baratura se calculan con datos hasta el cierre de
t-1; el retorno evaluado es el de t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import signals as sig
from backtest.crosssec import Panel


def _muestrear_pesos(n_activos: int, n_carteras: int, rng, concentracion: float = 1.0):
    """Carteras long-only al azar. Dirichlet(1) reparte uniforme sobre el símplex."""
    return rng.dirichlet(np.full(n_activos, concentracion), size=n_carteras)


def barrido_mensual(
    panel: Panel,
    cov_lookback: int = 36,
    trend_lookback: int = 36,
    n_carteras: int = 3000,
    banda: float = 0.05,
    percentil_riesgo: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Una fila por mes con el resultado del experimento.

    banda            -> anchura relativa de la banda de riesgo (0,05 = ±5%)
    percentil_riesgo -> dónde se fija el riesgo objetivo dentro de lo alcanzable
    """
    close = panel.close
    rets = close.pct_change()
    barato = sig.trend_gap(close, lookback=trend_lookback)   # alto = más barato
    fwd = close.shift(-1) / close - 1
    cols = list(close.columns)
    A = len(cols)
    rng = np.random.default_rng(seed)

    filas = []
    for i, ts in enumerate(close.index):
        if i < max(cov_lookback, trend_lookback) or i + 1 >= len(close):
            continue
        hist = rets.iloc[i - cov_lookback + 1:i + 1].dropna(how="any")
        if len(hist) < cov_lookback // 2:
            continue
        S = hist.cov().to_numpy() * 12.0                     # anualizada
        c = barato.loc[ts].to_numpy(dtype=float)
        f = fwd.loc[ts].to_numpy(dtype=float)
        if not np.isfinite(c).all() or not np.isfinite(f).all():
            continue

        W = _muestrear_pesos(A, n_carteras, rng)
        riesgo = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))
        objetivo = np.quantile(riesgo, percentil_riesgo)
        dentro = np.abs(riesgo - objetivo) <= banda * objetivo
        if dentro.sum() < 50:
            continue

        Wb = W[dentro]
        baratura = Wb @ c                                    # baratura de cada cartera
        retorno = Wb @ f
        orden = np.argsort(baratura)
        k = max(1, len(orden) // 10)

        filas.append({
            "mes": ts,
            "n_banda": int(dentro.sum()),
            "riesgo_objetivo": float(objetivo),
            # la más barata de la banda, contra la media de la banda (= azar)
            "edge_mas_barata": float(retorno[orden[-1]] - retorno.mean()),
            # decil más barato contra decil más caro: el test limpio de la señal
            "spread_deciles": float(retorno[orden[-k:]].mean() - retorno[orden[:k]].mean()),
            "ret_media_banda": float(retorno.mean()),
            "ret_mas_barata": float(retorno[orden[-1]]),
            "ret_decil_barato": float(retorno[orden[-k:]].mean()),
            # correlación dentro del mes entre baratura y retorno futuro
            "corr_baratura_ret": float(np.corrcoef(baratura, retorno)[0, 1]),
        })
    return pd.DataFrame(filas).set_index("mes")


def pesos_mas_barata(
    panel: Panel,
    cov_lookback: int = 36,
    trend_lookback: int = 36,
    n_carteras: int = 2500,
    banda: float = 0.05,
    percentil_riesgo: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pesos de la cartera más barata de la banda, y de la banda media (el azar).

    Devuelve dos DataFrames de pesos alineados por mes, listos para meter en el
    motor de aportación ponderada.
    """
    close = panel.close
    rets = close.pct_change()
    barato = sig.trend_gap(close, lookback=trend_lookback)
    cols = list(close.columns)
    A = len(cols)
    rng = np.random.default_rng(seed)

    w_barata, w_media = {}, {}
    for i, ts in enumerate(close.index):
        if i < max(cov_lookback, trend_lookback):
            continue
        hist = rets.iloc[i - cov_lookback + 1:i + 1].dropna(how="any")
        if len(hist) < cov_lookback // 2:
            continue
        S = hist.cov().to_numpy() * 12.0
        c = barato.loc[ts].to_numpy(dtype=float)
        if not np.isfinite(c).all():
            continue
        W = _muestrear_pesos(A, n_carteras, rng)
        riesgo = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))
        objetivo = np.quantile(riesgo, percentil_riesgo)
        dentro = np.abs(riesgo - objetivo) <= banda * objetivo
        if dentro.sum() < 50:
            continue
        Wb = W[dentro]
        j = int(np.argmax(Wb @ c))
        w_barata[ts] = pd.Series(Wb[j], index=cols)
        w_media[ts] = pd.Series(Wb.mean(axis=0), index=cols)

    return (pd.DataFrame(w_barata).T.sort_index(),
            pd.DataFrame(w_media).T.sort_index())
