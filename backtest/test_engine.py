"""Tests del simulador. Ejecutar: python -m backtest.test_engine"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import engine

MONTHS = pd.DataFrame(
    {
        "first_open": [100.0, 110.0, 90.0, 120.0],
        "high": [115.0, 115.0, 125.0, 130.0],
        "low": [95.0, 85.0, 88.0, 110.0],
        "close": [110.0, 90.0, 120.0, 125.0],
        "sessions": [21, 20, 22, 21],
    },
    index=pd.date_range("2024-01-31", periods=4, freq="ME"),
)


def test_limit_inalcanzable_equivale_a_dca():
    """Si el límite está muy por encima del precio, la orden ejecuta al open:
    debe dar exactamente el mismo resultado que el DCA."""
    huge = pd.Series(1e9, index=MONTHS.index)
    d = engine.dca(MONTHS, contribution=1000)
    l = engine.limit_strategy(MONTHS, huge, contribution=1000)
    assert np.isclose(d.shares, l.shares), (d.shares, l.shares)
    assert np.isclose(d.final_value, l.final_value)
    assert l.fill_rate == 1.0


def test_limite_nunca_tocado_deja_todo_en_caja():
    tiny = pd.Series(1.0, index=MONTHS.index)
    r = engine.limit_strategy(MONTHS, tiny, contribution=1000, unfilled="carry")
    assert r.shares == 0.0
    assert np.isclose(r.cash, 4000.0)
    assert r.fills == 0


def test_fallback_cierre_invierte_todo():
    tiny = pd.Series(1.0, index=MONTHS.index)
    r = engine.limit_strategy(MONTHS, tiny, contribution=1000, unfilled="close")
    assert np.isclose(r.invested, 4000.0)
    assert np.isclose(r.cash, 0.0)
    esperado = sum(1000 / c for c in MONTHS["close"])
    assert np.isclose(r.shares, esperado)


def test_oracle_compra_en_el_minimo():
    r = engine.limit_strategy(MONTHS, engine.oracle_limits(MONTHS), contribution=1000)
    esperado = sum(1000 / lo for lo in MONTHS["low"])
    assert np.isclose(r.shares, esperado)
    assert r.fill_rate == 1.0


def test_carry_acumula_y_compra_mas_tarde():
    """Mes 1 sin ejecución, mes 2 con límite alcanzable: entran $2000."""
    limits = pd.Series([1.0, 115.0, 1.0, 1.0], index=MONTHS.index)
    r = engine.limit_strategy(MONTHS, limits, contribution=1000, unfilled="carry")
    # Mes 2 abre a 110 <= 115 -> ejecuta al open (mejor precio)
    assert np.isclose(r.shares, 2000 / 110.0)
    assert np.isclose(r.invested, 2000.0)
    assert np.isclose(r.cash, 2000.0)


def test_caja_renta_al_tipo_indicado():
    tiny = pd.Series(1.0, index=MONTHS.index)
    r = engine.limit_strategy(MONTHS, tiny, contribution=1000, unfilled="carry", cash_rate=0.12)
    assert r.cash > 4000.0
    assert np.isclose(r.total_contributed, 4000.0)  # los intereses no son aportación


def test_irr_de_flujo_plano():
    """Sin revalorización, la TIR debe ser ~0."""
    flat = MONTHS.copy()
    for c in ["first_open", "high", "low", "close"]:
        flat[c] = 100.0
    r = engine.dca(flat, contribution=1000)
    assert abs(r.irr_annual()) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests pasados")
