"""Simulador de acumulación mensual: DCA vs órdenes límite.

Todas las estrategias aportan el MISMO capital (`contribution` cada mes), lo que
hace la comparación justa: lo único que cambia es a qué precio entra ese dinero
y si llega a entrar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Result:
    name: str
    trades: pd.DataFrame = field(repr=False)
    shares: float = 0.0
    invested: float = 0.0
    cash: float = 0.0
    final_price: float = 0.0
    months: int = 0
    fills: int = 0
    contributed: float = 0.0  # dinero realmente aportado (sin intereses de la caja)

    @property
    def fill_rate(self) -> float:
        return self.fills / self.months if self.months else float("nan")

    @property
    def avg_cost(self) -> float:
        return self.invested / self.shares if self.shares else float("nan")

    @property
    def final_value(self) -> float:
        """Valor final: posición + caja sin invertir (la caja también es tuya)."""
        return self.shares * self.final_price + self.cash

    @property
    def total_contributed(self) -> float:
        return self.contributed

    @property
    def total_return(self) -> float:
        return self.final_value / self.total_contributed - 1

    def irr_annual(self) -> float:
        """TIR anualizada sobre el flujo real de aportaciones mensuales."""
        flows = [-c for c in self.trades["contribution"].tolist()]
        flows[-1] += self.final_value
        return _irr(flows, periods_per_year=12)


def _irr(flows: list[float], periods_per_year: int = 12) -> float:
    """TIR por bisección sobre la tasa periódica; devuelve tasa anualizada."""
    def npv(r: float) -> float:
        total, disc = 0.0, 1.0
        for f in flows:
            total += f / disc
            disc *= 1 + r
            if disc == 0.0 or not np.isfinite(disc):
                break
        return total

    lo, hi = -0.9, 1.0
    while npv(hi) > 0 and hi < 1e3:  # activo muy rentable: ampliamos el techo
        hi *= 2
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (1 + (lo + hi) / 2) ** periods_per_year - 1


def _run(
    name: str,
    months: pd.DataFrame,
    contribution: float,
    limits: pd.Series | None,
    unfilled: str,
    cash_rate: float,
) -> Result:
    """Motor común.

    limits=None  -> DCA puro (compra al primer open del mes).
    unfilled     -> qué hacer si la orden límite no se ejecuta:
                    "carry"  : el dinero se acumula para el mes siguiente
                    "close"  : se compra al cierre del mes
                    "next"   : se compra al primer open del mes siguiente
    cash_rate    -> rendimiento anual de la caja no invertida (ej. 0.04)
    """
    monthly_cash_rate = (1 + cash_rate) ** (1 / 12) - 1
    shares = invested = cash = 0.0
    fills = 0
    pending_next = 0.0  # dinero comprometido a comprar en el open del mes siguiente
    rows = []

    idx = months.index
    for i, ts in enumerate(idx):
        row = months.loc[ts]
        cash *= 1 + monthly_cash_rate
        budget = contribution + cash
        cash = 0.0

        # Ejecución diferida del modo "next"
        if pending_next > 0:
            px = float(row["first_open"])
            shares += pending_next / px
            invested += pending_next
            pending_next = 0.0

        limit = float(limits.loc[ts]) if limits is not None and ts in limits.index else np.nan
        fill_price = np.nan

        if limits is None:
            fill_price = float(row["first_open"])
        elif not np.isnan(limit):
            # Si el mes ya abre por debajo del límite, la orden ejecuta al open
            # (mejor precio); si no, ejecuta en el límite si el mínimo lo toca.
            if float(row["first_open"]) <= limit:
                fill_price = float(row["first_open"])
            elif float(row["low"]) <= limit:
                fill_price = limit

        if not np.isnan(fill_price):
            shares += budget / fill_price
            invested += budget
            fills += 1
        else:
            if unfilled == "close":
                fill_price = float(row["close"])
                shares += budget / fill_price
                invested += budget
            elif unfilled == "next" and i + 1 < len(idx):
                pending_next = budget
            else:  # "carry" (o "next" en el último mes: queda en caja)
                cash = budget

        rows.append(
            {
                "month": ts,
                "contribution": contribution,
                "limit": limit,
                "low": float(row["low"]),
                "first_open": float(row["first_open"]),
                "close": float(row["close"]),
                "fill_price": fill_price,
                "filled": not np.isnan(fill_price),
                "cash_after": cash,
            }
        )

    if pending_next > 0:  # no debería ocurrir, pero no perdemos dinero
        cash += pending_next

    return Result(
        name=name,
        trades=pd.DataFrame(rows).set_index("month"),
        shares=shares,
        invested=invested,
        cash=cash,
        final_price=float(months["close"].iloc[-1]),
        months=len(idx),
        fills=fills,
        contributed=contribution * len(idx),
    )


def dca(months: pd.DataFrame, contribution: float = 1000.0, cash_rate: float = 0.0) -> Result:
    """Baseline: comprar al primer open de cada mes, siempre."""
    return _run("DCA", months, contribution, None, "carry", cash_rate)


def limit_strategy(
    months: pd.DataFrame,
    limits: pd.Series,
    contribution: float = 1000.0,
    unfilled: str = "carry",
    cash_rate: float = 0.0,
    name: str = "LIMIT",
) -> Result:
    return _run(name, months, contribution, limits, unfilled, cash_rate)


def oracle_limits(months: pd.DataFrame) -> pd.Series:
    """Previsión perfecta: el límite es exactamente el mínimo real del mes.

    Es la cota superior teórica de toda la arquitectura del proyecto: ningún
    modelo puede hacerlo mejor que esto.
    """
    return months["low"].copy()


def summarize(results: list[Result]) -> pd.DataFrame:
    base = results[0]
    out = []
    for r in results:
        out.append(
            {
                "estrategia": r.name,
                "meses": r.months,
                "% ejecutado": 100 * r.fill_rate,
                "aportado": r.total_contributed,
                "invertido": r.invested,
                "caja final": r.cash,
                "coste medio": r.avg_cost,
                "acciones": r.shares,
                "valor final": r.final_value,
                "rent. total %": 100 * r.total_return,
                "TIR anual %": 100 * r.irr_annual(),
                "vs DCA %": 100 * (r.final_value / base.final_value - 1),
            }
        )
    return pd.DataFrame(out)
