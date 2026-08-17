"""Verifica que el motor vectorizado coincide con el de crosssec.py.

Todas las cifras del estudio largo salen de window_table(); si no coincidiera
con el simulador mes a mes, los resultados no valdrían nada.

Ejecutar: python -m backtest.test_panels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import signals as sig
from backtest.crosssec import Panel, evaluate
from backtest.panels import load_panel, picks_from_scores, window_table

IDX = pd.date_range("2020-01-31", periods=8, freq="ME")
SMALL = Panel(
    buy=pd.DataFrame({"A": [10.0, 11, 12, 13, 14, 15, 16, 17],
                      "B": [20.0, 19, 18, 17, 16, 15, 14, 13]}, index=IDX),
    close=pd.DataFrame({"A": [11.0, 12, 13, 14, 15, 16, 17, 18],
                        "B": [19.0, 18, 17, 16, 15, 14, 13, 12]}, index=IDX),
)


def test_no_lookahead_en_picks():
    """La señal del último mes no puede afectar a ninguna elección."""
    sc = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    base = picks_from_scores(SMALL, sc)
    tampered = sc.copy()
    tampered.loc[IDX[-1]] = [-99.0, 99.0]
    assert np.array_equal(base, picks_from_scores(SMALL, tampered))


def test_primer_mes_sin_senal():
    sc = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    assert picks_from_scores(SMALL, sc)[0] == -1


def test_window_table_equipeso_coincide_con_dca_manual():
    ew = pd.DataFrame(0.0, index=IDX, columns=["A", "B"])
    t = window_table(SMALL, ew, window=4, contribution=100.0)
    fila = t.iloc[0]
    px = SMALL.close.iloc[3]
    esperado = sum(
        50 / SMALL.buy.iloc[m]["A"] * px["A"] + 50 / SMALL.buy.iloc[m]["B"] * px["B"]
        for m in range(4)
    ) / 400 - 1
    assert np.isclose(fila["equipond"], esperado)


def test_percentil_entre_cero_y_uno():
    sc = sig.trend_gap(SMALL.close, lookback=3)
    t = window_table(SMALL, sc, window=4)
    assert t["percentil"].between(0, 1).all()


def test_coincide_con_el_motor_mes_a_mes():
    """La prueba importante: mismo resultado que crosssec.evaluate(), con datos reales."""
    panel = load_panel(["SPY", "QQQ", "GLD", "MSFT"])
    sub = Panel(buy=panel.buy.iloc[-60:], close=panel.close.iloc[-60:])
    for name in ("trend_gap", "momentum_12_1", "reversal_1m"):
        sc = sig.make_scores(name, sub.close)
        lento = evaluate(sub, sc, window=24, top_k=1)
        rapido = window_table(sub, sc, window=24)
        assert np.allclose(lento["estrategia"], rapido["estrategia"]), name
        assert np.allclose(lento["equiponderado"], rapido["equipond"]), name
        assert np.allclose(lento["mediana_dca"], rapido["mediana"]), name


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests pasados")
