"""Qué comprar este mes. Uso:

    python -m backtest.run_plan --aportacion 500
    python -m backtest.run_plan --aportacion 500 --cartera "GLD=1200,NVDA=3000,SPY=800"
"""
from __future__ import annotations

import argparse

import pandas as pd

from backtest import plan_mensual as pm
from backtest.panels import load_panel

CESTA = ["GLD", "SLV", "NVDA", "QQQ", "SPY", "IWM", "MSFT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cesta", default=",".join(CESTA))
    ap.add_argument("--aportacion", type=float, default=1000.0)
    ap.add_argument("--cartera", default="", help='p.ej. "GLD=1200,NVDA=3000"')
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--suelo", type=float, default=0.05)
    ap.add_argument("--techo", type=float, default=0.30)
    args = ap.parse_args()

    tk = args.cesta.split(",")
    panel = load_panel(tk)
    obj = pm.pesos_objetivo(panel.close, args.lookback, args.suelo, args.techo).iloc[-1]

    actual = pd.Series(0.0, index=tk)
    for parte in filter(None, args.cartera.split(",")):
        k, v = parte.split("=")
        actual[k.strip()] = float(v)

    euros = pm.reparto_del_mes(actual, obj, args.aportacion)
    vol = panel.close.pct_change().rolling(args.lookback).std().iloc[-1] * (12 ** 0.5)
    total = actual.sum() + args.aportacion

    t = pd.DataFrame({
        "vol. anual %": 100 * vol,
        "peso objetivo %": 100 * obj,
        "tienes ahora": actual,
        "peso actual %": 100 * actual / actual.sum() if actual.sum() > 0 else 0.0,
        "COMPRAR": euros,
        "peso tras comprar %": 100 * (actual + euros) / total,
    }).sort_values("COMPRAR", ascending=False)

    print(f"\nDatos hasta {panel.months[-1]:%Y-%m}  |  aportación de {args.aportacion:,.0f}\n")
    print(t.to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"\n  Suma a comprar: {euros.sum():,.2f}")
    if actual.sum() == 0:
        print("  (cartera vacía: el primer mes reparte según los pesos objetivo)")


if __name__ == "__main__":
    main()
