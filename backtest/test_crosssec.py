"""Tests del motor transversal. Ejecutar: python -m backtest.test_crosssec"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import signals as sig
from backtest.crosssec import Panel, dca_single, simulate

IDX = pd.date_range("2024-01-31", periods=6, freq="ME")
PANEL = Panel(
    buy=pd.DataFrame({"A": [10.0, 11, 12, 13, 14, 15], "B": [20.0, 19, 18, 17, 16, 15]}, index=IDX),
    close=pd.DataFrame({"A": [11.0, 12, 13, 14, 15, 16], "B": [19.0, 18, 17, 16, 15, 14]}, index=IDX),
)


def test_no_lookahead():
    """Cambiar las puntuaciones del último mes no puede alterar el resultado:
    demuestra que simulate() sólo lee señales ANTERIORES a cada compra."""
    scores = pd.DataFrame({"A": [1.0, 1, 1, 1, 1, 1], "B": [0.0, 0, 0, 0, 0, 0]}, index=IDX)
    base = simulate(PANEL, scores, top_k=1)

    tampered = scores.copy()
    tampered.loc[IDX[-1]] = [-99.0, 99.0]  # señal del último mes: nunca se usa
    after = simulate(PANEL, tampered, top_k=1)

    assert np.isclose(base["final_value"], after["final_value"])


def test_senal_constante_elige_siempre_el_mismo():
    scores = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    r = simulate(PANEL, scores, top_k=1, contribution=100)
    picks = set(r["picks"]["picks"].iloc[1:])  # el 1er mes no tiene señal previa
    assert picks == {"A"}


def test_primer_mes_sin_senal_reparte():
    scores = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    r = simulate(PANEL, scores, top_k=1, contribution=100)
    assert r["picks"]["picks"].iloc[0] == "A,B"


def test_top_k_igual_a_universo_es_equiponderado():
    scores = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    r = simulate(PANEL, scores, top_k=2, contribution=100)
    esperado = sum(50 / PANEL.buy.loc[t, "A"] for t in IDX)
    assert np.isclose(r["shares"]["A"], esperado)


def test_dca_single_coincide_con_simulate_de_un_activo():
    uno = Panel(buy=PANEL.buy[["A"]], close=PANEL.close[["A"]])
    scores = pd.DataFrame({"A": 1.0}, index=IDX)
    a = dca_single(uno, "A", contribution=100)
    b = simulate(uno, scores, top_k=1, contribution=100)
    assert np.isclose(a["final_value"], b["final_value"])


def test_trend_gap_detecta_el_barato():
    """A crece en línea recta y cae al final; B crece limpio.
    A debe puntuar más alto (más por debajo de su tendencia)."""
    n = 40
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    a = np.exp(np.linspace(0, 1, n)); a[-1] *= 0.80   # A se desploma un 20%
    b = np.exp(np.linspace(0, 1, n))
    p = pd.DataFrame({"A": a, "B": b}, index=idx)
    sc = sig.trend_gap(p, lookback=36)
    assert sc.iloc[-1]["A"] > sc.iloc[-1]["B"]


def test_momentum_detecta_el_ganador():
    n = 20
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    p = pd.DataFrame({"A": np.linspace(100, 200, n), "B": np.linspace(100, 105, n)}, index=idx)
    sc = sig.momentum_12_1(p, lookback=12, skip=1)
    assert sc.iloc[-1]["A"] > sc.iloc[-1]["B"]


def test_irr_de_dca_plano_es_cero():
    flat = Panel(
        buy=pd.DataFrame({"A": [100.0] * 6}, index=IDX),
        close=pd.DataFrame({"A": [100.0] * 6}, index=IDX),
    )
    assert abs(dca_single(flat, "A", contribution=100)["irr"]) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests pasados")
