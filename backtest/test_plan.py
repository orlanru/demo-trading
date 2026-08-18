"""Tests del plan mensual. Ejecutar: python -m backtest.test_plan"""
import numpy as np, pandas as pd
from backtest import plan_mensual as pm
from backtest.crosssec import Panel

IDX = pd.date_range("2020-01-31", periods=30, freq="ME")

def _panel(volatil, tranquilo):
    rng = np.random.default_rng(0)
    a = 100*np.cumprod(1+volatil*rng.standard_normal(30))
    b = 100*np.cumprod(1+tranquilo*rng.standard_normal(30))
    df = pd.DataFrame({"VOLATIL": a, "TRANQUILO": b}, index=IDX)
    return Panel(buy=df, close=df)

def test_el_volatil_pesa_menos():
    p = _panel(0.10, 0.02)
    w = pm.pesos_objetivo(p.close, lookback=12, suelo=0.0, techo=1.0).dropna()
    assert (w["TRANQUILO"] > w["VOLATIL"]).all()

def test_los_topes_se_respetan():
    p = _panel(0.20, 0.005)
    w = pm.pesos_objetivo(p.close, lookback=12, suelo=0.10, techo=0.60).dropna()
    assert w.min().min() >= 0.10 - 1e-9 and w.max().max() <= 0.60 + 1e-9
    assert np.allclose(w.sum(axis=1), 1.0)

def test_la_aportacion_va_al_infraponderado():
    actual = pd.Series({"A": 9000.0, "B": 1000.0})
    obj = pd.Series({"A": 0.5, "B": 0.5})
    e = pm.reparto_del_mes(actual, obj, 1000.0)
    assert e["B"] > e["A"] and np.isclose(e.sum(), 1000.0)

def test_nunca_vende():
    actual = pd.Series({"A": 9999.0, "B": 1.0})
    obj = pd.Series({"A": 0.5, "B": 0.5})
    e = pm.reparto_del_mes(actual, obj, 100.0)
    assert (e >= 0).all()

def test_topes_imposibles_dan_error():
    """Con 2 activos y techo del 30% no se puede sumar 100%: debe avisar."""
    p = _panel(0.05, 0.05)
    try:
        pm.pesos_objetivo(p.close, lookback=12, suelo=0.05, techo=0.30)
    except ValueError:
        return
    raise AssertionError("deberia haber avisado de topes imposibles")


def test_cartera_vacia_usa_los_pesos_objetivo():
    actual = pd.Series({"A": 0.0, "B": 0.0})
    obj = pd.Series({"A": 0.7, "B": 0.3})
    e = pm.reparto_del_mes(actual, obj, 1000.0)
    assert np.allclose(e.to_numpy(), [700.0, 300.0])

def test_simular_no_mira_al_futuro():
    """Cambiar el último cierre no puede alterar lo comprado en meses previos."""
    p = _panel(0.08, 0.03)
    r1 = pm.simular(p, aportacion=100.0, suelo=0.1, techo=0.6)
    p2 = Panel(buy=p.buy.copy(), close=p.close.copy())
    p2.close.iloc[-1] *= 3.0
    r2 = pm.simular(p2, aportacion=100.0, suelo=0.1, techo=0.6)
    cols = [c for c in r1["historia"].columns if c.startswith("eur_")]
    assert np.allclose(r1["historia"][cols].to_numpy(), r2["historia"][cols].to_numpy())

if __name__ == "__main__":
    fns=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns: f(); print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests pasados")
