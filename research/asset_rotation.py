"""
Análisis de rotación de activos: ¿elegir cada mes QUÉ activo seguro y de riesgo
comprar mejora al DCA fijo? Incluye un modelo ML evaluado WALK-FORWARD (fuera de
muestra), que es la única prueba honesta.

Marco justo: todas las estrategias reciben $budget/mes, lo reparten risk/safe en
la misma proporción, y solo ACUMULAN (nunca venden). Lo único que cambia es QUÉ
activo de cada cubo se compra ese mes. Sin look-ahead: la decisión del mes t usa
features con datos hasta el cierre de t-1; el retorno objetivo es el de t.
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

RISK = ["SPY", "QQQ", "EEM", "EFA", "IWM", "BTC-USD"]
SAFE = ["TLT", "IEF", "GLD", "LQD"]
ALL = RISK + SAFE
RISK_W, SAFE_W = 0.80, 0.20
BUDGET = 100.0
START_BT = "2010-01-01"   # inicio del backtest (tras warmup de features + ML)
FEATURES = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "dd_12m", "dist_sma12", "rsi14", "vol_12m"]


def fetch_monthly(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, start="2004-01-01", progress=False, auto_adjust=True)
    if hasattr(d.columns, "levels"):
        d.columns = [c[0] for c in d.columns]
    d = d.dropna()
    m = pd.DataFrame(index=d["Close"].resample("ME").last().index)
    m["Close"] = d["Close"].resample("ME").last()
    return m


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    c = m["Close"]
    f = pd.DataFrame(index=m.index)
    f["Close"] = c
    f["fwd_ret"] = c.pct_change().shift(-1)          # objetivo: retorno del mes siguiente
    f["mom_1m"] = c.pct_change(1)
    f["mom_3m"] = c.pct_change(3)
    f["mom_6m"] = c.pct_change(6)
    f["mom_12m"] = c.pct_change(12)
    roll_max = c.rolling(12, min_periods=1).max()
    f["dd_12m"] = c / roll_max - 1.0
    sma12 = c.rolling(12).mean()
    f["dist_sma12"] = (c - sma12) / sma12
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean().replace(0, np.nan)
    f["rsi14"] = 100 - 100 / (1 + gain / loss)
    f["vol_12m"] = c.pct_change().rolling(12).std()
    return f


def load_panel() -> dict[str, pd.DataFrame]:
    return {t: build_features(fetch_monthly(t)) for t in ALL}


# ---------- motor de acumulación ----------
def run_accumulation(panel, pick_fn, months, *, fee_bps=5.0) -> pd.Series:
    """
    pick_fn(t, panel) -> (risk_ticker | None, safe_ticker | None)
    Devuelve la serie mensual de valor de cartera (equity).
    """
    fee = fee_bps / 1e4
    units = {t: 0.0 for t in ALL}
    equity = []
    for t in months:
        risk_t, safe_t = pick_fn(t, panel)
        for tick, w in [(risk_t, RISK_W), (safe_t, SAFE_W)]:
            if tick is None:
                continue
            price = panel[tick]["Close"].get(t, np.nan)
            if np.isfinite(price) and price > 0:
                units[tick] += (w * BUDGET * (1 - fee)) / price
        # marca a mercado
        val = 0.0
        for tick in ALL:
            price = panel[tick]["Close"].get(t, np.nan)
            if np.isfinite(price):
                val += units[tick] * price
        equity.append(val)
    return pd.Series(equity, index=months)


def available(panel, bucket, t):
    """Activos del cubo con features válidas en t (causal: usa t para decidir compra en t)."""
    out = []
    for tick in bucket:
        row = panel[tick].loc[t] if t in panel[tick].index else None
        if row is not None and np.isfinite(row["Close"]) and np.isfinite(row.get("mom_12m", np.nan)):
            out.append(tick)
    return out


# ---------- estrategias de selección ----------
def pick_static(t, panel):
    return ("SPY", "TLT")

def pick_equalweight(t, panel):
    # "comprar todo" no encaja en pick (1 por cubo); aproximamos con el de mayor cap implícita:
    # en su lugar lo tratamos aparte abajo (run_equalweight).
    return ("SPY", "TLT")

def pick_value(t, panel):
    # comprar el MÁS barato = el de mayor drawdown (más caído) en cada cubo
    rs = available(panel, RISK, t)
    sf = available(panel, SAFE, t)
    rpick = min(rs, key=lambda x: panel[x].loc[t, "dd_12m"]) if rs else None
    spick = min(sf, key=lambda x: panel[x].loc[t, "dd_12m"]) if sf else None
    return (rpick, spick)

def pick_momentum(t, panel):
    rs = available(panel, RISK, t)
    sf = available(panel, SAFE, t)
    rpick = max(rs, key=lambda x: panel[x].loc[t, "mom_12m"]) if rs else None
    spick = max(sf, key=lambda x: panel[x].loc[t, "mom_12m"]) if sf else None
    return (rpick, spick)


def run_equalweight(panel, months, *, fee_bps=5.0) -> pd.Series:
    """DCA diversificado naive: reparte el peso del cubo entre TODOS sus activos disponibles."""
    fee = fee_bps / 1e4
    units = {t: 0.0 for t in ALL}
    equity = []
    for t in months:
        for bucket, w in [(RISK, RISK_W), (SAFE, SAFE_W)]:
            av = available(panel, bucket, t)
            if not av:
                continue
            each = w * BUDGET / len(av)
            for tick in av:
                price = panel[tick]["Close"].loc[t]
                if price > 0:
                    units[tick] += each * (1 - fee) / price
        val = sum(units[tick] * panel[tick]["Close"].get(t, 0.0) for tick in ALL if np.isfinite(panel[tick]["Close"].get(t, np.nan)))
        equity.append(val)
    return pd.Series(equity, index=months)


# ---------- ML walk-forward ----------
def build_ml_picker(panel, model_kind="hgb", min_train=48):
    """
    Devuelve pick_fn que entrena un modelo POOLED (todos los activos) con datos
    estrictamente anteriores a t y predice el retorno forward de cada activo en t.
    Expanding window, re-entreno cada mes. Cache de predicciones por mes.
    """
    # tabla larga: (date, ticker, features..., fwd_ret)
    rows = []
    for tick in ALL:
        f = panel[tick].dropna(subset=FEATURES)
        for dt, r in f.iterrows():
            rows.append((dt, tick, *[r[c] for c in FEATURES], r["fwd_ret"]))
    long = pd.DataFrame(rows, columns=["date", "ticker", *FEATURES, "fwd_ret"]).sort_values("date")

    cache = {}

    def predict_month(t):
        if t in cache:
            return cache[t]
        train = long[(long["date"] < t) & long["fwd_ret"].notna()]
        cur = long[long["date"] == t]
        if len(train) < min_train or cur.empty:
            cache[t] = {}
            return cache[t]
        X, y = train[FEATURES].values, train["fwd_ret"].values
        model = (HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=20, l2_regularization=1.0)
                 if model_kind == "hgb" else Ridge(alpha=10.0))
        model.fit(X, y)
        pred = model.predict(cur[FEATURES].values)
        cache[t] = dict(zip(cur["ticker"], pred))
        return cache[t]

    def pick_ml(t, panel):
        preds = predict_month(t)
        rs = [x for x in available(panel, RISK, t) if x in preds]
        sf = [x for x in available(panel, SAFE, t) if x in preds]
        rpick = max(rs, key=lambda x: preds[x]) if rs else None
        spick = max(sf, key=lambda x: preds[x]) if sf else None
        return (rpick, spick)

    return pick_ml


# ---------- métricas ----------
def irr_annual(contribs_per_month, final_value, months):
    flows = [(d, -BUDGET) for d in months]
    flows.append((months[-1], final_value))
    t0 = flows[0][0]
    yrs = [(d - t0).days / 365.25 for d, _ in flows]
    amt = [a for _, a in flows]
    def npv(r): return sum(a / (1 + r) ** t for a, t in zip(amt, yrs))
    lo, hi = -0.99, 5.0
    if np.sign(npv(lo)) == np.sign(npv(hi)): return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sign(npv(mid)) == np.sign(npv(lo)): lo = mid
        else: hi = mid
    return (lo + hi) / 2

def metrics(equity, months):
    final = float(equity.iloc[-1]); contributed = BUDGET * len(months)
    rets = equity.pct_change().dropna()
    sharpe = float(np.sqrt(12) * rets.mean() / rets.std()) if rets.std() > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1).min()) * 100
    irr = irr_annual(None, final, list(months)) * 100
    return dict(final=final, contributed=contributed, profit=final - contributed,
                ret_pct=(final / contributed - 1) * 100, irr=irr, sharpe=sharpe, maxdd=dd)


def main():
    print("Descargando panel de activos...")
    panel = load_panel()
    # meses comunes desde START_BT donde al menos SPY tiene features
    months = panel["SPY"].dropna(subset=FEATURES).index
    months = months[months >= pd.Timestamp(START_BT)]

    strategies = {
        "Static SPY+TLT": lambda: run_accumulation(panel, pick_static, months),
        "Diversified (equal-weight all)": lambda: run_equalweight(panel, months),
        "Value rotation (buy cheapest)": lambda: run_accumulation(panel, pick_value, months),
        "Momentum rotation (buy strongest)": lambda: run_accumulation(panel, pick_momentum, months),
        "ML walk-forward (HGB)": lambda: run_accumulation(panel, build_ml_picker(panel, "hgb"), months),
        "ML walk-forward (Ridge)": lambda: run_accumulation(panel, build_ml_picker(panel, "ridge"), months),
    }

    print(f"\nPeriodo backtest: {months[0].date()} -> {months[-1].date()}  ({len(months)} meses)")
    print(f"Universo RISK: {RISK}\nUniverso SAFE: {SAFE}")
    print(f"Aportación: ${BUDGET}/mes  ({int(RISK_W*100)}% risk / {int(SAFE_W*100)}% safe)\n")

    rows = []
    for name, fn in strategies.items():
        eq = fn()
        mm = metrics(eq, months)
        rows.append((name, mm))
    print(f"{'Strategy':36s} {'Profit $':>12s} {'Ret %':>8s} {'IRR %/yr':>9s} {'Sharpe':>7s} {'MaxDD %':>8s}")
    print("-" * 86)
    for name, mm in rows:
        print(f"{name:36s} {mm['profit']:>12,.0f} {mm['ret_pct']:>8.1f} {mm['irr']:>9.2f} {mm['sharpe']:>7.2f} {mm['maxdd']:>8.1f}")


if __name__ == "__main__":
    main()
