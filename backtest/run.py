"""Compara la estrategia de orden límite del terminal contra DCA.

Uso:
    python -m backtest.run --ticker SPY
    python -m backtest.run --ticker BTC-USD --margin-pct 0.02 --cash-rate 0.04
"""
from __future__ import annotations

import argparse

import pandas as pd

from backtest import engine
from backtest.data import load_ohlc, monthly_ohlc
from backtest.model import walk_forward_predictions
from config import ASSETS


def build_limits(preds: pd.DataFrame, margin_usd: float, margin_pct: float | None) -> pd.Series:
    """limit = pred_low + margen. La app usa margen fijo en USD; margin_pct
    permite probar la variante proporcional al precio."""
    if margin_pct is not None:
        return preds["pred_low"] * (1 + margin_pct)
    return preds["pred_low"] + margin_usd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--contribution", type=float, default=1000.0)
    ap.add_argument("--margin-usd", type=float, default=None, help="por defecto, el de config.py")
    ap.add_argument("--margin-pct", type=float, default=None, help="margen relativo, ej. 0.02 = 2%%")
    ap.add_argument("--cash-rate", type=float, default=0.0, help="rendimiento anual de la caja parada")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=None, help="ej. 0.8 para pérdida pinball")
    ap.add_argument("--offline", action="store_true", help="usar el cache CSV en vez de yfinance")
    ap.add_argument("--sweep-naive", action="store_true", help="barrido de umbrales fijos sin modelo")
    args = ap.parse_args()

    cfg = ASSETS.get(args.ticker, {"margin_usd": 0.0, "start_date": "2000-01-01"})
    margin_usd = args.margin_usd if args.margin_usd is not None else float(cfg["margin_usd"])

    dfd = load_ohlc(args.ticker, start_date=cfg.get("start_date", "2000-01-01"), prefer_cache=args.offline)
    preds = walk_forward_predictions(dfd, warmup=args.warmup, quantile=args.quantile)

    months = monthly_ohlc(dfd)
    months = months.loc[months.index.intersection(preds.index)]
    preds = preds.loc[months.index]

    limits = build_limits(preds, margin_usd, args.margin_pct)
    oracle = engine.oracle_limits(months)

    kw = dict(contribution=args.contribution, cash_rate=args.cash_rate)
    results = [
        engine.dca(months, **kw),
        engine.limit_strategy(months, limits, unfilled="carry", name="LIMIT (acumula caja)", **kw),
        engine.limit_strategy(months, limits, unfilled="close", name="LIMIT (fallback cierre)", **kw),
        engine.limit_strategy(months, limits, unfilled="next", name="LIMIT (fallback open t+1)", **kw),
        engine.limit_strategy(months, oracle, unfilled="carry", name="ORACLE (mínimo real)", **kw),
    ]

    if args.sweep_naive:
        print(f"\n--- Regla ingenua: límite = cierre_mes_anterior * (1 - k), sin modelo ---")
        print(f"{'k':>6} {'% ejec.':>9} {'coste medio':>12} {'valor final':>13} {'vs DCA %':>9}")
        base = results[0]
        for k in [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
            naive = preds["prev_close"] * (1 - k)
            r = engine.limit_strategy(months, naive, unfilled="carry", name=f"k={k}", **kw)
            print(f"{k:>6.2%} {100*r.fill_rate:>8.1f}% {r.avg_cost:>12,.2f} {r.final_value:>13,.2f} "
                  f"{100*(r.final_value/base.final_value-1):>8.2f}")
        print()

    hits = (preds["actual_low"] <= limits).mean()
    mae = (preds["pred_drop"] - preds["actual_drop"]).abs().mean()

    print(f"\n=== {args.ticker} | {months.index[0]:%Y-%m} → {months.index[-1]:%Y-%m} "
          f"({len(months)} meses) | aporte ${args.contribution:,.0f}/mes ===")
    print(f"margen: {'%.2f%%' % (100*args.margin_pct) if args.margin_pct is not None else '$%.2f' % margin_usd}"
          f" | caja al {100*args.cash_rate:.1f}% anual"
          f" | objetivo: {'pinball q=%.2f' % args.quantile if args.quantile else 'error cuadrático (como la app)'}")
    print(f"límite tocado en {100*hits:.1f}% de los meses | MAE de la predicción: {100*mae:.2f} pp\n")

    df = engine.summarize(results)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()


if __name__ == "__main__":
    main()
