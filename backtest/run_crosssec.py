"""Compara señales de selección transversal contra DCA.

Uso:
    python -m backtest.run_crosssec --window 24
    python -m backtest.run_crosssec --window 24 --top-k 2 --signals trend_gap,momentum_12_1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest import signals as sig
from backtest.crosssec import (
    DEFAULT_UNIVERSE, evaluate, load_panel, dca_single, summarize_eval, rolling_windows, simulate
)


def random_baseline(panel, window: int, top_k: int, contribution: float, n_seeds: int = 200) -> dict:
    """Elegir al azar cada mes. Si una señal no bate esto, no aporta nada."""
    tickers = list(panel.buy.columns)
    rng = np.random.default_rng(0)
    wins_ew, diffs = [], []
    ew_scores = pd.DataFrame(0.0, index=panel.months, columns=tickers)
    for w in rolling_windows(panel, window):
        ew = simulate(panel, ew_scores, top_k=len(tickers), contribution=contribution, months=w)
        for _ in range(max(1, n_seeds // 20)):
            noise = pd.DataFrame(
                rng.standard_normal((len(panel.months), len(tickers))),
                index=panel.months, columns=tickers,
            )
            r = simulate(panel, noise, top_k=top_k, contribution=contribution, months=w)
            diffs.append(100 * (r["total_return"] - ew["total_return"]))
            wins_ew.append(r["total_return"] > ew["total_return"])
    return {
        "señal": f"ALEATORIA (media de {len(diffs)} sorteos)",
        "ventanas": len(rolling_windows(panel, window)),
        "vs equipond. (pp)": float(np.mean(diffs)),
        "% gana a equipond.": 100 * float(np.mean(wins_ew)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--window", type=int, default=24, help="meses por ventana (24 = 2 años)")
    ap.add_argument("--top-k", type=int, default=1, help="en cuántos activos se reparte cada aportación")
    ap.add_argument("--contribution", type=float, default=1000.0)
    ap.add_argument("--lookback", type=int, default=36, help="meses de la regresión de tendencia")
    ap.add_argument("--mom-lookback", type=int, default=12, help="meses de formación del momentum")
    ap.add_argument("--show-picks", action="store_true", help="frecuencia de activos elegidos")
    ap.add_argument("--signals", default="trend_gap,trend_gap_filtered,drawdown,reversal_1m,momentum_12_1")
    ap.add_argument("--online", action="store_true", help="descargar de yfinance en vez de usar cache")
    ap.add_argument("--ml", default="", help="modelos ML a incluir, ej. ridge,gbm")
    ap.add_argument("--ml-warmup", type=int, default=24)
    ap.add_argument("--edge-test", action="store_true",
                    help="test estadístico del edge de selección mes a mes")
    args = ap.parse_args()

    tickers = args.universe.split(",")
    panel = load_panel(tickers, offline=not args.online)

    print(f"\nUniverso: {', '.join(tickers)}")
    print(f"Periodo: {panel.months[0]:%Y-%m} → {panel.months[-1]:%Y-%m} ({len(panel.months)} meses)")
    print(f"Ventanas solapadas de {args.window} meses | aporte ${args.contribution:,.0f}/mes "
          f"| top_k={args.top_k}\n")

    # --- referencia: DCA en cada activo sobre todo el periodo ---
    print("--- DCA individual, periodo completo ---")
    ref = []
    for t in tickers:
        r = dca_single(panel, t, args.contribution)
        ref.append({"activo": t, "rent. total %": 100 * r["total_return"], "TIR anual %": 100 * r["irr"]})
    ref_df = pd.DataFrame(ref).sort_values("rent. total %", ascending=False)
    print(ref_df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print(f"\nMEDIANA de los DCA individuales: {ref_df['rent. total %'].median():.1f}%\n")

    # --- señales ---
    all_scores = {}
    for name in args.signals.split(","):
        if name:
            all_scores[name] = sig.make_scores(
                name, panel.close, trend_lookback=args.lookback, mom_lookback=args.mom_lookback
            )
    for m in [x for x in args.ml.split(",") if x]:
        from backtest.ml_selector import walk_forward_scores

        all_scores[f"ML-{m}"] = walk_forward_scores(
            panel, model=m, warmup=args.ml_warmup, trend_lookback=args.lookback
        )

    rows = []
    for name, scores in all_scores.items():
        ev = evaluate(panel, scores, window=args.window, top_k=args.top_k,
                      contribution=args.contribution)
        rows.append(summarize_eval(ev, name))
        if args.show_picks:
            picks = simulate(panel, scores, top_k=args.top_k,
                             contribution=args.contribution)["picks"]["picks"]
            top = picks[~picks.str.contains(",")].value_counts()
            print(f"  {name:22s} elige: {dict(top)}  (+{(picks.str.contains(',')).sum()} meses sin señal)")

    rnd = random_baseline(panel, args.window, args.top_k, args.contribution)
    rows.append(rnd)

    out = pd.DataFrame(rows)
    print(f"--- Resultado sobre todas las ventanas de {args.window} meses ---")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.to_string(index=False, float_format=lambda v: f"{v:,.1f}", na_rep="—"))
    print()

    if args.edge_test:
        from backtest.selection_test import selection_edge, summarize

        print("--- Edge de selección mes a mes (test con potencia estadística real) ---")
        print("    H0: la señal no informa -> edge medio = 0. |t| > 2 sería indicio;")
        print(f"    con {len(all_scores)} señales x 2 horizontes, el umbral honesto está en |t| ~ 2.9.\n")
        et = [
            summarize(selection_edge(panel, s, horizon=h, top_k=args.top_k), n, h)
            for n, s in all_scores.items()
            for h in (1, 3)
        ]
        print(pd.DataFrame(et).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print()


if __name__ == "__main__":
    main()
