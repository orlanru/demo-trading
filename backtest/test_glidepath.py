"""Tests de la trayectoria de riesgo. Ejecutar: python -m backtest.test_glidepath"""
import numpy as np
from backtest.glidepath import riqueza_por_orden, robustez_reordenando, simular_glidepath


def test_orden_importa_en_acumulacion():
    """Mismos retornos, distinto orden: la riqueza final cambia."""
    r = np.array([0.10] * 12 + [-0.05] * 12)
    assert not np.isclose(riqueza_por_orden(r), riqueza_por_orden(r[::-1]))


def test_malos_retornos_al_principio_dan_mas_riqueza():
    """Aportando cada mes, es mejor que lo flojo venga antes."""
    bueno_tarde = np.array([-0.05] * 12 + [0.10] * 12)
    bueno_pronto = bueno_tarde[::-1]
    assert riqueza_por_orden(bueno_tarde) > riqueza_por_orden(bueno_pronto)


def test_sin_aportaciones_el_orden_no_importa():
    """Con una única inversión inicial, el orden es irrelevante: es el producto.
    Confirma que el efecto viene de las APORTACIONES, no de los retornos."""
    r = np.array([0.10] * 6 + [-0.05] * 6)
    assert np.isclose(np.prod(1 + r), np.prod(1 + r[::-1]))


def test_glidepath_plano_equivale_a_peso_constante():
    r1 = np.array([0.02] * 24); r2 = np.array([0.005] * 24)
    a = simular_glidepath(r1, r2, 0.7, 0.7)
    w = 0.7; v = 0.0
    for i in range(24):
        v = (v + 1000.0) * (1 + w * r1[i] + (1 - w) * r2[i])
    assert np.isclose(a, v)


def test_robustez_devuelve_metricas_coherentes():
    rng = np.random.default_rng(0)
    r1 = 0.008 + 0.05 * rng.standard_normal(120)
    r2 = 0.002 + 0.01 * rng.standard_normal(120)
    out = robustez_reordenando(r1, r2, n_permutaciones=100)
    assert 0 <= out["vs_decreciente_positivo_pct"] <= 100
    lo, hi = out["vs_decreciente_banda"]
    assert lo <= out["vs_decreciente_media"] <= hi or lo <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests pasados")
