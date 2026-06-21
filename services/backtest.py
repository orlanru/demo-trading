"""
Motor de backtest para comparar estrategias de acumulación frente al DCA.

=== Marco de comparación JUSTO ===
Todas las estrategias reciben EXACTAMENTE las mismas entradas de caja: una
aportación fija `budget` al principio de cada mes (como una nómina). Lo único
que cambia es CUÁNTO despliega cada estrategia ese mes y a qué precio. El resto
se queda como caja ("dry powder") y puede opcionalmente rentar el risk-free.

Al final, el valor de la cartera = unidades * precio + caja. Así la comparación
es honesta: mismo dinero aportado, distinta política de despliegue. Esta es la
razón por la que el "DCA con orden límite" original empataba con el DCA: solo
cambiaba el precio de entrada, no la exposición.

Estrategias incluidas:
  - dca            : despliega el 100% de la caja cada mes al cierre.
  - limit_dca      : pone una orden límite al "mínimo proyectado"; si no se
                     llena, la caja se acumula (proxy del enfoque original).
  - smart_dca      : asignación dinámica con multiplicador de señales + dry
                     powder (compra más en las caídas).
  - trend_filtered : DCA mientras la tendencia es alcista; en tendencia bajista
                     no compra y mantiene caja (control de drawdown).

Sin look-ahead: la decisión del mes M usa señales calculadas con datos hasta
el cierre de M-1. El precio de ejecución es del propio mes M.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from services.signals import monthly_frame, smart_multiplier


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                 # valor total de la cartera (mensual)
    invested_value: pd.Series         # valor solo de las unidades (sin caja)
    cash: pd.Series                   # caja a fin de mes
    contributions: pd.Series          # aportación recibida ese mes
    deployed: pd.Series               # cantidad efectivamente invertida ese mes
    metrics: dict = field(default_factory=dict)


def _annualized_irr(cashflows: list[tuple[pd.Timestamp, float]], final_value: float,
                    final_date: pd.Timestamp) -> float:
    """
    TIR anualizada (money-weighted) por bisección. Las aportaciones son flujos
    negativos (sale dinero del bolsillo) y el valor final es positivo.
    """
    flows = [(d, -amt) for d, amt in cashflows]
    flows.append((final_date, final_value))
    t0 = flows[0][0]
    years = [(d - t0).days / 365.25 for d, _ in flows]
    amounts = [a for _, a in flows]

    def npv(rate: float) -> float:
        return sum(a / (1 + rate) ** t for a, t in zip(amounts, years))

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            return mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(equity: pd.Series, rf_annual: float = 0.0) -> float:
    """Sharpe sobre retornos mensuales del valor de la cartera, anualizado."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    excess = rets - rf_m
    return float(np.sqrt(12) * excess.mean() / excess.std())


def _build_result(name, idx, equity, invested_value, cash, contribs, deployed,
                  fills=None, rf_annual=0.0) -> BacktestResult:
    equity = pd.Series(equity, index=idx)
    invested_value = pd.Series(invested_value, index=idx)
    cash = pd.Series(cash, index=idx)
    contribs = pd.Series(contribs, index=idx)
    deployed = pd.Series(deployed, index=idx)

    total_contributed = float(contribs.sum())
    final_value = float(equity.iloc[-1])
    cashflows = [(d, float(c)) for d, c in contribs.items() if c > 0]

    irr = _annualized_irr(cashflows, final_value, idx[-1]) if cashflows else float("nan")
    # Exposición media: fracción del valor que está invertida (no en caja)
    exposure = float((invested_value / equity.replace(0, np.nan)).mean())

    metrics = {
        "final_value": final_value,
        "total_contributed": total_contributed,
        "profit": final_value - total_contributed,
        "total_return_pct": (final_value / total_contributed - 1.0) * 100 if total_contributed else float("nan"),
        "irr_annual_pct": irr * 100 if np.isfinite(irr) else float("nan"),
        "sharpe": _sharpe(equity, rf_annual),
        "max_drawdown_pct": _max_drawdown(equity) * 100,
        "avg_exposure_pct": exposure * 100 if np.isfinite(exposure) else float("nan"),
        "end_cash": float(cash.iloc[-1]),
        "months": len(idx),
    }
    if fills is not None:
        metrics["fill_rate_pct"] = 100.0 * np.mean(fills) if len(fills) else float("nan")

    return BacktestResult(name, equity, invested_value, cash, contribs, deployed, metrics)


def _monthly_inputs(daily: pd.DataFrame):
    """
    Devuelve (m, months) donde m es el frame de señales mensual y months es la
    lista de timestamps de fin de mes con precio de cierre válido.
    """
    m = monthly_frame(daily)
    m = m.dropna(subset=["Close"])
    return m


def run_strategy(
    daily: pd.DataFrame,
    strategy: str = "dca",
    budget: float = 100.0,
    fee_bps: float = 5.0,
    rf_annual: float = 0.0,
    *,
    limit_drop_k: float = 1.0,
    smart_lo: float = 0.30,
    smart_hi: float = 3.0,
) -> BacktestResult:
    """
    Ejecuta una estrategia sobre el histórico diario `daily`.

    Parámetros:
      budget      : aportación fija mensual (misma para todas las estrategias).
      fee_bps     : coste por operación en puntos básicos (5 bps = 0.05%).
      rf_annual   : rentabilidad anual de la caja no invertida (risk-free).
      limit_drop_k: para limit_dca, el límite = prev_close*(1 - k*vol_mensual).
    """
    m = _monthly_inputs(daily)
    if len(m) < 14:
        raise ValueError("Histórico insuficiente para backtest (se necesitan >14 meses).")

    fee = fee_bps / 10_000.0
    rf_m = (1 + rf_annual) ** (1 / 12) - 1

    idx = m.index
    units = 0.0
    cash = 0.0

    equity, invested_value, cash_hist, contribs, deployed, fills = [], [], [], [], [], []

    for i, ts in enumerate(idx):
        row = m.loc[ts]
        price = float(row["Close"])
        month_low = float(row["Low"])

        # 1) Renta de la caja del mes anterior + nueva aportación
        cash = cash * (1 + rf_m) + budget
        contribs.append(budget)

        # 2) Señales causales: usar el mes anterior (i-1)
        prev = m.iloc[i - 1] if i > 0 else None

        invest_amt = 0.0
        exec_price = price
        filled = None

        if strategy == "dca":
            invest_amt = cash
            exec_price = price

        elif strategy == "limit_dca":
            # Mínimo proyectado = cierre previo ajustado por volatilidad reciente.
            if prev is not None and np.isfinite(prev.get("vol_12m", np.nan)):
                prev_close = float(prev["Close"])
                vol = float(prev["vol_12m"])
                limit = prev_close * (1 - limit_drop_k * vol)
                if month_low <= limit:
                    invest_amt = cash         # se llena: despliega toda la caja
                    exec_price = limit
                    filled = True
                else:
                    invest_amt = 0.0          # no se llena: la caja se acumula
                    filled = False
            else:
                invest_amt = cash             # sin señal aún -> DCA
            fills.append(1.0 if filled else 0.0) if filled is not None else None

        elif strategy == "smart_dca":
            mult = smart_multiplier(prev, lo=smart_lo, hi=smart_hi) if prev is not None else 1.0
            target = mult * budget
            invest_amt = min(cash, target)    # nunca más que la caja disponible
            exec_price = price

        elif strategy == "trend_filtered":
            trend_up = bool(prev["trend_up"]) if (prev is not None and np.isfinite(prev.get("trend_up", np.nan))) else True
            invest_amt = cash if trend_up else 0.0
            exec_price = price

        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

        # 3) Ejecutar la compra (con coste)
        invest_amt = max(0.0, min(invest_amt, cash))
        if invest_amt > 0 and exec_price > 0:
            bought = (invest_amt * (1 - fee)) / exec_price
            units += bought
            cash -= invest_amt
        deployed.append(invest_amt)

        # 4) Marcar a mercado a cierre del mes
        inv_val = units * price
        invested_value.append(inv_val)
        cash_hist.append(cash)
        equity.append(inv_val + cash)

    name = {
        "dca": "DCA (baseline)",
        "limit_dca": "Limit-order DCA (original)",
        "smart_dca": "Smart-DCA (dynamic)",
        "trend_filtered": "Trend-filtered DCA",
    }.get(strategy, strategy)

    fills_arg = [f for f in fills if f is not None] if strategy == "limit_dca" else None
    return _build_result(name, idx, equity, invested_value, cash_hist, contribs,
                         deployed, fills=fills_arg, rf_annual=rf_annual)


def run_all(
    daily: pd.DataFrame,
    strategies: list[str] | None = None,
    budget: float = 100.0,
    fee_bps: float = 5.0,
    rf_annual: float = 0.0,
    **kwargs,
) -> dict[str, BacktestResult]:
    if strategies is None:
        strategies = ["dca", "limit_dca", "smart_dca", "trend_filtered"]
    out: dict[str, BacktestResult] = {}
    for s in strategies:
        out[s] = run_strategy(daily, s, budget=budget, fee_bps=fee_bps,
                              rf_annual=rf_annual, **kwargs)
    return out


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Tabla comparativa lista para mostrar."""
    rows = []
    for r in results.values():
        mm = r.metrics
        rows.append({
            "Strategy": r.name,
            "Final $": round(mm["final_value"], 2),
            "Contributed $": round(mm["total_contributed"], 2),
            "Profit $": round(mm["profit"], 2),
            "Total Return %": round(mm["total_return_pct"], 2),
            "IRR %/yr": round(mm["irr_annual_pct"], 2),
            "Sharpe": round(mm["sharpe"], 2),
            "Max DD %": round(mm["max_drawdown_pct"], 2),
            "Avg Exposure %": round(mm["avg_exposure_pct"], 1),
            "Fill Rate %": round(mm.get("fill_rate_pct", float("nan")), 1),
        })
    return pd.DataFrame(rows)
