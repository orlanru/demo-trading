"""Reglas de APORTACIÓN y REBALANCEO, no de selección.

Todo lo anterior intentaba predecir qué activo subiría. Esto es distinto: no
predice nada. Sólo decide cuánto dinero va a cada sitio y cuándo restaurar los
pesos. El premio por rebalanceo es un efecto mecánico —la cartera rebalanceada
crece más que la media de sus componentes cuando éstos son volátiles y no están
perfectamente correlacionados— y no requiere acertar nada.

Simulador con estado: hace falta porque estas reglas son dependientes del
camino. Cuánto dinero mandar a un activo depende de cuánto tienes ya en él, y
eso depende de todo el historial.

Regla de juego para que la comparación sea justa: todas las estrategias reciben
exactamente la misma renta (`income` al mes). Lo que no se invierte queda en
caja, la caja cuenta en la riqueza final, y nunca se puede invertir dinero que
no se ha ingresado todavía.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.crosssec import Panel


@dataclass
class Resultado:
    nombre: str
    shares: np.ndarray
    cash: float
    aportado: float
    valor_final: float
    historia: pd.DataFrame = field(repr=False, default=None)

    @property
    def retorno_total(self) -> float:
        return self.valor_final / self.aportado - 1

    def tir_anual(self, n_meses: int, income: float) -> float:
        """TIR anualizada de n aportaciones iguales que acaban valiendo valor_final."""
        def fv(r):
            return sum(income * (1 + r) ** (n_meses - 1 - i) for i in range(n_meses))
        lo, hi = -0.9, 1.0
        while fv(hi) < self.valor_final and hi < 1e3:
            hi *= 2
        if fv(lo) > self.valor_final:
            return float("nan")
        for _ in range(200):
            mid = (lo + hi) / 2
            if fv(mid) < self.valor_final:
                lo = mid
            else:
                hi = mid
        return (1 + (lo + hi) / 2) ** 12 - 1


def _simular(panel: Panel, regla, nombre: str, income: float = 1000.0,
             cash_rate: float = 0.0, meses: pd.DatetimeIndex | None = None) -> Resultado:
    """Motor común. `regla(estado) -> (pesos_compra, vender_para_rebalancear)`.

    pesos_compra: array que suma 1, cómo repartir el dinero a desplegar
    Devuelve además cuánto desplegar; ver las reglas concretas más abajo.
    """
    idx = panel.months if meses is None else meses
    buy = panel.buy.loc[idx].to_numpy(float)
    close = panel.close.loc[idx].to_numpy(float)
    A = buy.shape[1]
    tasa_mensual = (1 + cash_rate) ** (1 / 12) - 1

    shares = np.zeros(A)
    cash = 0.0
    filas = []

    for m in range(len(idx)):
        cash = cash * (1 + tasa_mensual) + income
        precios = buy[m]
        valor_pos = shares * precios
        estado = {"m": m, "n": len(idx), "shares": shares, "precios": precios,
                  "valor_pos": valor_pos, "cash": cash, "income": income, "A": A,
                  "panel": panel, "idx": idx}

        desplegar, pesos, ventas = regla(estado)
        desplegar = float(min(max(desplegar, 0.0), cash))

        # ventas (sólo las reglas que rebalancean vendiendo)
        if ventas is not None:
            deltas = ventas  # en participaciones, negativas = vender
            shares = shares + deltas
            cash -= float(deltas @ precios)

        if desplegar > 0 and pesos is not None:
            s = pesos.sum()
            if s > 0:
                shares = shares + desplegar * (pesos / s) / precios
                cash -= desplegar

        filas.append({"mes": idx[m], "cash": cash,
                      "valor": float(shares @ close[m]) + cash})

    valor_final = float(shares @ close[-1]) + cash
    return Resultado(nombre=nombre, shares=shares, cash=cash,
                     aportado=income * len(idx), valor_final=valor_final,
                     historia=pd.DataFrame(filas).set_index("mes"))


# ---------------------------------------------------------------- reglas

def dca_equiponderado(estado):
    """Referencia: todo el dinero, repartido a partes iguales. La cartera deriva
    hacia los ganadores porque nunca se corrige."""
    A = estado["A"]
    return estado["cash"], np.ones(A), None


def aportacion_al_infraponderado(estado):
    """Rebalanceo por aportación: el dinero nuevo va a los activos que están por
    debajo de su peso objetivo (equiponderado). Nunca vende, así que no genera
    peaje fiscal ni comisiones de venta.
    """
    A = estado["A"]
    valor = estado["valor_pos"]
    total = valor.sum() + estado["cash"]
    objetivo = total / A
    falta = np.clip(objetivo - valor, 0, None)
    if falta.sum() <= 0:
        return estado["cash"], np.ones(A), None
    return estado["cash"], falta, None


def rebalanceo_anual(estado):
    """DCA equiponderado + rebalanceo completo cada 12 meses (con ventas)."""
    A = estado["A"]
    ventas = None
    if estado["m"] > 0 and estado["m"] % 12 == 0:
        valor = estado["valor_pos"]
        total = valor.sum()
        if total > 0:
            objetivo = total / A
            ventas = (objetivo - valor) / estado["precios"]   # participaciones
    return estado["cash"], np.ones(A), ventas


def value_averaging(estado, crecimiento_mensual: float = 0.008):
    """Value averaging (Edleson): la cartera debe seguir una senda de valor
    objetivo que crece con las aportaciones. Si va por debajo, se invierte más;
    si va por encima, menos (aquí no se vende: sólo se deja de aportar).

    El crecimiento objetivo es un parámetro libre — y ahí está la trampa del
    método: elegirlo mirando el resultado es sobreajuste.
    """
    A = estado["A"]
    m, income = estado["m"], estado["income"]
    # senda objetivo: aportaciones acumuladas capitalizadas al ritmo objetivo
    objetivo = sum(income * (1 + crecimiento_mensual) ** (m - i) for i in range(m + 1))
    actual = estado["valor_pos"].sum()
    desplegar = max(0.0, objetivo - actual)
    return desplegar, np.ones(A), None


def aportacion_segun_caida(estado, dd: pd.Series, umbral: float = -0.10,
                           multiplicador: float = 2.0):
    """Aportar más cuando el mercado está caído, guardando caja cuando está en
    máximos. Requiere haber acumulado caja antes: no se puede invertir dinero
    que no se ha ingresado.
    """
    A = estado["A"]
    ts = estado["idx"][estado["m"]]
    caida = float(dd.get(ts, 0.0))
    if caida <= umbral:
        return estado["cash"], np.ones(A), None          # despliega todo lo guardado
    return estado["income"] * 0.6, np.ones(A), None      # guarda el 40%


def volatilidad_inversa(estado, pesos: pd.DataFrame):
    A = estado["A"]
    ts = estado["idx"][estado["m"] - 1] if estado["m"] > 0 else None
    if ts is None or ts not in pesos.index:
        return estado["cash"], np.ones(A), None
    w = pesos.loc[ts].to_numpy(float)
    w = np.nan_to_num(w, nan=0.0)
    if w.sum() <= 0:
        w = np.ones(A)
    return estado["cash"], w, None


def simular(panel: Panel, nombre: str, regla, **kw) -> Resultado:
    return _simular(panel, regla, nombre, **kw)
