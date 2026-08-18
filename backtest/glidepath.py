"""Trayectoria de riesgo en la fase de acumulación.

Hallazgo con mecanismo claro: en un plan de aportaciones, el dinero está
concentrado en los años FINALES. La exposición de los primeros años se aplica a
una cartera pequeña; la de los últimos, a una grande. Por eso subir el peso de
bolsa con el tiempo bate a bajarlo, aunque el peso MEDIO sea idéntico.

Es lo contrario de lo que hacen los fondos de ciclo de vida, que reducen riesgo
con la edad — y por buenas razones si te acercas a la jubilación y vas a
retirar. En pura acumulación, la aritmética va al revés.

A diferencia del resto del estudio, esto no predice nada y sobrevive a reordenar
los retornos al azar: no es una propiedad de este camino histórico concreto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.crosssec import Panel


def simular_glidepath(r_riesgo: np.ndarray, r_refugio: np.ndarray,
                      w_inicial: float, w_final: float,
                      aportacion: float = 1000.0) -> float:
    """Riqueza final aportando cada mes con el peso de riesgo interpolado."""
    w = np.linspace(w_inicial, w_final, len(r_riesgo))
    v = 0.0
    for i in range(len(r_riesgo)):
        v = (v + aportacion) * (1 + w[i] * r_riesgo[i] + (1 - w[i]) * r_refugio[i])
    return v


def series_de_cartera(panel: Panel, riesgo: list[str], refugio: list[str]):
    r = panel.close.pct_change().dropna()
    return r[riesgo].mean(axis=1).to_numpy(), r[refugio].mean(axis=1).to_numpy()


def robustez_reordenando(r_riesgo, r_refugio, w_ini=0.4, w_fin=1.0,
                  n_permutaciones: int = 2000, seed: int = 0) -> dict:
    """Reordena los retornos al azar para separar el mecanismo del camino.

    Los dos activos se permutan con el MISMO índice, así que se conserva su
    correlación mes a mes; lo único que cambia es el orden temporal.
    """
    rng = np.random.default_rng(seed)
    n = len(r_riesgo)
    vs_decreciente, vs_constante = [], []
    w_medio = (w_ini + w_fin) / 2
    for _ in range(n_permutaciones):
        ix = rng.permutation(n)
        a = simular_glidepath(r_riesgo[ix], r_refugio[ix], w_ini, w_fin)
        b = simular_glidepath(r_riesgo[ix], r_refugio[ix], w_fin, w_ini)
        c = simular_glidepath(r_riesgo[ix], r_refugio[ix], w_medio, w_medio)
        vs_decreciente.append(100 * (a / b - 1))
        vs_constante.append(100 * (a / c - 1))
    d, e = np.array(vs_decreciente), np.array(vs_constante)
    return {
        "vs_decreciente_media": float(d.mean()),
        "vs_decreciente_positivo_pct": 100 * float((d > 0).mean()),
        "vs_decreciente_banda": (float(np.percentile(d, 5)), float(np.percentile(d, 95))),
        "vs_constante_media": float(e.mean()),
        "vs_constante_positivo_pct": 100 * float((e > 0).mean()),
        "vs_constante_banda": (float(np.percentile(e, 5)), float(np.percentile(e, 95))),
    }


def riqueza_por_orden(r: np.ndarray, aportacion: float = 1000.0) -> float:
    """Riqueza final de un plan de aportaciones sobre una serie de retornos."""
    v = 0.0
    for x in r:
        v = (v + aportacion) * (1 + x)
    return v
