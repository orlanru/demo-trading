"""Tests de la señal de noticias. Ejecutar: python -m backtest.test_news"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import news

IDX = pd.date_range("2020-01-31", periods=24, freq="ME")


def _tono(a_base: float, b_base: float, ruido: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"A": a_base + ruido * rng.standard_normal(24),
         "B": b_base + ruido * rng.standard_normal(24)},
        index=IDX,
    )


def test_el_nivel_crudo_confunde_linea_base_con_pesimismo():
    """A tiene un tono base peor que B pero ninguno se mueve.

    La señal de NIVEL elige A siempre (mide el vocabulario del tema);
    la señal normalizada no distingue, que es lo correcto.
    """
    tono = _tono(-1.5, -0.2)          # constantes: no hay pesimismo relativo
    nivel = news.news_contrarian_level(tono)
    assert (nivel["A"] > nivel["B"]).all()          # el nivel siempre prefiere A

    z = news.news_contrarian_z(tono, lookback=12)
    assert z.dropna().empty or np.allclose(z.dropna().to_numpy(), 0, atol=1e-8) or z.isna().all().all()


def test_z_detecta_un_choque_de_pesimismo():
    tono = _tono(-0.5, -0.5)
    tono.iloc[-1, tono.columns.get_loc("A")] = -3.0   # A se hunde el último mes
    z = news.news_contrarian_z(tono, lookback=12)
    assert z.iloc[-1]["A"] > z.iloc[-1]["B"]


def test_contraria_y_momentum_son_opuestas():
    tono = _tono(-0.5, -0.4, ruido=0.3)
    c = news.news_contrarian_z(tono, lookback=12).dropna()
    m = news.news_momentum_z(tono, lookback=12).dropna()
    assert np.allclose(c.to_numpy(), -m.to_numpy())


def test_news_drop_mide_el_cambio_no_el_nivel():
    tono = pd.DataFrame({"A": np.linspace(-2, -2, 24), "B": np.linspace(0, -1.5, 24)},
                        index=IDX)
    d = news.news_drop(tono)
    # A está peor en nivel, pero B es el que empeora -> drop prefiere B
    assert d.iloc[-1]["B"] > d.iloc[-1]["A"]


def test_atencion_no_premia_pesimismo_sin_cobertura():
    tono = _tono(-0.5, -0.5)
    tono.iloc[-1, tono.columns.get_loc("A")] = -3.0
    tono.iloc[-1, tono.columns.get_loc("B")] = -3.0
    vol = pd.DataFrame({"A": [1.0] * 24, "B": [1.0] * 24}, index=IDX)
    vol.iloc[-1, vol.columns.get_loc("B")] = 5.0      # sólo B acapara cobertura
    s = news.news_attention_drop(tono, vol, lookback=12)
    assert s.iloc[-1]["B"] > s.iloc[-1]["A"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests pasados")
