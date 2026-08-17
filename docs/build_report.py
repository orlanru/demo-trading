# -*- coding: utf-8 -*-
"""Genera el informe visual de los backtests como HTML autocontenido."""
import html, json, os

OUT = "/home/user/demo-trading/docs/informe.html"

# ---------------------------------------------------------------- datos
UNIVERSOS = ["sectores", "cesta_usuario", "cesta_sin_nvda", "amplio"]
UNI_LABEL = {
    "sectores": "10 sectores + SPY",
    "cesta_usuario": "tu cesta (con NVDA)",
    "cesta_sin_nvda": "tu cesta sin NVDA",
    "amplio": "amplio, 15 activos",
}
UNI_META = {
    "sectores": "1999–2026 · 333 meses",
    "cesta_usuario": "2006–2026 · 245 meses",
    "cesta_sin_nvda": "2006–2026 · 245 meses",
    "amplio": "2004–2026 · 262 meses",
}

# vs equiponderado en pp, por señal y universo
VS_EW = {
    "trend_gap":        {"sectores": 0.5, "cesta_usuario": -1.6, "cesta_sin_nvda": -2.9, "amplio": -1.6},
    "trend_gap_filtrado": {"sectores": 0.1, "cesta_usuario": 0.8, "cesta_sin_nvda": -1.8, "amplio": -1.8},
    "el más caído":     {"sectores": -1.2, "cesta_usuario": 6.9, "cesta_sin_nvda": -8.0, "amplio": -5.8},
    "reversión 1 mes":  {"sectores": -0.6, "cesta_usuario": 1.5, "cesta_sin_nvda": -1.7, "amplio": -1.1},
    "momentum 12-1":    {"sectores": 0.4, "cesta_usuario": 24.5, "cesta_sin_nvda": 3.9, "amplio": 0.1},
    "value 5 años":     {"sectores": -0.3, "cesta_usuario": None, "cesta_sin_nvda": -2.3, "amplio": -1.8},
    "ML Ridge":         {"sectores": -0.1, "cesta_usuario": None, "cesta_sin_nvda": 2.0, "amplio": -1.8},
    "ML GBM":           {"sectores": 0.8, "cesta_usuario": None, "cesta_sin_nvda": 1.3, "amplio": -0.3},
}
BANDA = {"sectores": (-1.1, 1.0), "cesta_usuario": (-3.8, 4.5),
         "cesta_sin_nvda": (-1.7, 1.5), "amplio": (-1.3, 1.0)}

PERCENTIL = [  # universo, estrategia (trend_gap), azar
    ("10 sectores + SPY", 51.0, 50.4),
    ("tu cesta (con NVDA)", 55.9, 59.0),
    ("tu cesta sin NVDA", 50.5, 51.4),
    ("amplio, 15 activos", 48.5, 49.0),
]

TSTATS = [  # señal, t en cesta_sin_nvda, t en sectores
    ("trend_gap", -1.17, -0.61),
    ("trend_gap en alcista", -1.70, -0.70),
    ("trend_gap filtrado", -1.69, -0.79),
    ("el más caído", -1.70, -1.13),
    ("el más caído en alcista", -2.10, -0.71),
    ("value 5 años", 0.17, -0.60),
    ("momentum 12-1", 0.72, 0.33),
    ("combo mom+value", 1.19, 0.52),
    ("ML Ridge", -1.28, -0.51),
    ("ML GBM", -0.05, -0.94),
]

ML = [  # universo, obs, modelo, R2 dentro, edge fuera, t
    ("sectores", 2960, "Ridge", 0.004, -0.12, -0.51),
    ("sectores", 2960, "GBM", 0.108, -0.22, -0.94),
    ("amplio", 3375, "Ridge", 0.004, -0.41, -1.34),
    ("amplio", 3375, "GBM", 0.108, -0.68, -2.46),
    ("cesta sin NVDA", 1248, "Ridge", 0.009, -0.36, -1.28),
    ("cesta sin NVDA", 1248, "GBM", 0.174, -0.02, -0.05),
]

BARRIDO = [(0, -0.33), (1, -0.65), (2, -1.02), (3, -1.81), (5, -1.79), (7, -3.86), (10, -5.94)]

ESTUDIO1 = {
    "SPY": [("DCA", 0.0), ("límite, acumula caja", -3.77), ("límite, compra al cierre", -0.82),
            ("límite, compra mes siguiente", -0.95), ("ORÁCULO: mínimo exacto", 3.56)],
    "BTC-USD": [("DCA", 0.0), ("límite, acumula caja", -9.83), ("límite, compra al cierre", -3.68),
                ("límite, compra mes siguiente", -3.67), ("ORÁCULO: mínimo exacto", 16.54)],
}

REPARTO = [  # método, universo -> (retorno medio, desviación típica)
    ("equiponderado", {"sectores": (10.4, 12.2), "cesta_sin_nvda": (14.8, 13.8), "amplio": (11.5, 10.4)}),
    ("volatilidad inversa", {"sectores": (9.9, 11.7), "cesta_sin_nvda": (15.0, 13.2), "amplio": (11.2, 10.0)}),
]

NOTICIAS = json.load(open("/tmp/claude-0/-home-user-demo-trading/fa96fec3-fcc9-5936-8976-5cd43ecf1070/scratchpad/noticias.json")) \
    if os.path.exists("/tmp/claude-0/-home-user-demo-trading/fa96fec3-fcc9-5936-8976-5cd43ecf1070/scratchpad/noticias.json") else None

# ---------------------------------------------------------------- helpers SVG
def esc(s):
    return html.escape(str(s), quote=True)


def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(label)}" '
            f'class="chart">')


def eje_x(x0, x1, y, vmin, vmax, ticks, sufijo="", fmt="{:+.0f}"):
    """Eje horizontal con marcas."""
    out = [f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="axis"/>']
    for v in ticks:
        px = x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
        out.append(f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" y2="{y+5}" class="axis"/>')
        out.append(f'<text x="{px:.1f}" y="{y+18}" class="tick" text-anchor="middle">'
                   f'{fmt.format(v)}{sufijo}</text>')
    return "".join(out)


# ---------------------------------------------------------------- gráfico 1: banda del azar
def chart_banda():
    señales = list(VS_EW.keys())
    W, PADL, PADR = 1040, 210, 30
    fila_h, cab_h, gap = 26, 46, 26
    alto_uni = cab_h + fila_h * len(señales) + gap
    H = 40 + alto_uni * len(UNIVERSOS)
    vmin, vmax = -9.5, 9.5
    x0, x1 = PADL, W - PADR

    def px(v):
        v = max(vmin, min(vmax, v))
        return x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)

    p = [svg_open(W, H, "Diferencia frente al DCA equiponderado, por señal y universo")]
    y = 30
    for uni in UNIVERSOS:
        lo, hi = BANDA[uni]
        p.append(f'<text x="0" y="{y}" class="facet">{esc(UNI_LABEL[uni])}</text>')
        p.append(f'<text x="{x1}" y="{y}" class="facet-meta" text-anchor="end">{esc(UNI_META[uni])}</text>')
        ytop, ybot = y + 12, y + 12 + fila_h * len(señales)
        # banda del azar
        p.append(f'<rect x="{px(lo):.1f}" y="{ytop}" width="{px(hi)-px(lo):.1f}" '
                 f'height="{ybot-ytop}" class="band"/>')
        p.append(f'<line x1="{px(0):.1f}" y1="{ytop}" x2="{px(0):.1f}" y2="{ybot}" class="zero"/>')
        for i, s in enumerate(señales):
            v = VS_EW[s][uni]
            yy = ytop + fila_h * i + fila_h / 2
            p.append(f'<text x="{PADL-14}" y="{yy+4}" class="rowlab" text-anchor="end">{esc(s)}</text>')
            if v is None:
                p.append(f'<text x="{px(0)+8:.1f}" y="{yy+4}" class="tick">no probado</text>')
                continue
            dentro = lo <= v <= hi
            cls = "dot-null" if dentro else ("dot-pos" if v > 0 else "dot-neg")
            fuera = v > vmax
            cx = px(v)
            p.append(f'<line x1="{px(0):.1f}" y1="{yy}" x2="{cx:.1f}" y2="{yy}" class="stem"/>')
            p.append(f'<circle cx="{cx:.1f}" cy="{yy}" r="5.5" class="{cls}" '
                     f'data-tip="{esc(s)} · {esc(UNI_LABEL[uni])}: {v:+.1f} pp frente al equiponderado'
                     f'{" — dentro de la banda del azar" if dentro else ""}"/>')
            etiqueta = f"{v:+.1f}"
            if fuera:
                p.append(f'<polygon points="{x1-4},{yy} {x1-14},{yy-5} {x1-14},{yy+5}" class="dot-pos-fill"/>')
                etiqueta = f"{v:+.1f} → es comprar NVDA"
                p.append(f'<text x="{x1-22:.1f}" y="{yy+4}" class="val" text-anchor="end">{esc(etiqueta)}</text>')
            else:
                anchor = "start" if v >= 0 else "end"
                dx = 10 if v >= 0 else -10
                p.append(f'<text x="{cx+dx:.1f}" y="{yy+4}" class="val" text-anchor="{anchor}">{etiqueta}</text>')
        p.append(eje_x(x0, x1, ybot + 6, vmin, vmax, [-8, -4, 0, 4, 8], " pp"))
        y += alto_uni
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- gráfico 2: percentil
def chart_percentil():
    W, H = 1040, 250
    PADL, PADR = 210, 40
    x0, x1 = PADL, W - PADR
    fila = 46

    def px(v):
        return x0 + v / 100 * (x1 - x0)

    p = [svg_open(W, H, "Percentil en que acaba la estrategia entre los DCA individuales")]
    p.append(f'<rect x="{px(40):.1f}" y="26" width="{px(60)-px(40):.1f}" height="{fila*4}" class="band"/>')
    p.append(f'<line x1="{px(50):.1f}" y1="26" x2="{px(50):.1f}" y2="{26+fila*4}" class="zero"/>')
    p.append(f'<text x="{px(50):.1f}" y="18" class="tick" text-anchor="middle">mediana</text>')
    for i, (lab, est, azar) in enumerate(PERCENTIL):
        yy = 26 + fila * i + fila / 2
        p.append(f'<text x="{PADL-14}" y="{yy+4}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
        p.append(f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" class="track"/>')
        p.append(f'<circle cx="{px(azar):.1f}" cy="{yy}" r="7" class="dot-null" '
                 f'data-tip="elección aleatoria: percentil {azar:.1f}"/>')
        p.append(f'<circle cx="{px(est):.1f}" cy="{yy}" r="5" class="dot-neg" '
                 f'data-tip="comprar el más barato: percentil {est:.1f}"/>')
    p.append(eje_x(x0, x1, 26 + fila * 4 + 8, 0, 100, [0, 25, 50, 75, 100], "", "{:.0f}"))
    p.append(f'<text x="{x0}" y="{H-8}" class="tick">0 = el peor activo · 100 = el mejor</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- gráfico 3: t-stats
def chart_tstats():
    W = 1040
    PADL, PADR = 230, 30
    fila = 30
    H = 70 + fila * len(TSTATS)
    x0, x1 = PADL, W - PADR
    vmin, vmax = -3.2, 3.2

    def px(v):
        return x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)

    p = [svg_open(W, H, "Estadístico t del edge de selección")]
    ytop, ybot = 40, 40 + fila * len(TSTATS)
    # zona no significativa
    p.append(f'<rect x="{px(-2.9):.1f}" y="{ytop}" width="{px(2.9)-px(-2.9):.1f}" '
             f'height="{ybot-ytop}" class="band"/>')
    for v, lab in [(-2.9, "−2,9"), (2.9, "+2,9")]:
        p.append(f'<line x1="{px(v):.1f}" y1="{ytop}" x2="{px(v):.1f}" y2="{ybot}" class="thresh"/>')
        p.append(f'<text x="{px(v):.1f}" y="{ytop-8}" class="tick" text-anchor="middle">{lab}</text>')
    p.append(f'<text x="{px(0):.1f}" y="{ytop-24}" class="tick" text-anchor="middle">'
             f'zona sin significancia (corregida por probar 10 señales)</text>')
    p.append(f'<line x1="{px(0):.1f}" y1="{ytop}" x2="{px(0):.1f}" y2="{ybot}" class="zero"/>')
    for i, (lab, t1, t2) in enumerate(TSTATS):
        yy = ytop + fila * i + fila / 2
        p.append(f'<text x="{PADL-14}" y="{yy+4}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
        for t, r, cls, quien in [(t2, 4.5, "dot-null", "10 sectores + SPY"),
                                 (t1, 6.0, "dot-neg" if t1 < 0 else "dot-pos", "tu cesta sin NVDA")]:
            p.append(f'<circle cx="{px(t):.1f}" cy="{yy}" r="{r}" class="{cls}" '
                     f'data-tip="{esc(lab)} · {esc(quien)}: t = {t:+.2f}"/>')
    p.append(eje_x(x0, x1, ybot + 6, vmin, vmax, [-3, -2, -1, 0, 1, 2, 3], "", "{:+.0f}"))
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- gráfico 4: estudio 1
def chart_estudio1():
    W, H = 1040, 330
    PADL, PADR = 230, 30
    x0, x1 = PADL, W - PADR
    vmin, vmax = -11, 18

    def px(v):
        return x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)

    p = [svg_open(W, H, "Estrategia de orden límite frente al DCA")]
    y = 26
    fila = 26
    for activo, filas in ESTUDIO1.items():
        p.append(f'<text x="0" y="{y}" class="facet">{esc(activo)}</text>')
        ytop = y + 10
        for i, (lab, v) in enumerate(filas):
            yy = ytop + fila * i + fila / 2
            p.append(f'<text x="{PADL-14}" y="{yy+4}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
            if v == 0:
                p.append(f'<circle cx="{px(0):.1f}" cy="{yy}" r="5" class="dot-null" '
                         f'data-tip="DCA: referencia"/>')
                continue
            cls = "bar-ceil" if v > 0 else "bar-neg"
            xa, xb = (px(0), px(v)) if v > 0 else (px(v), px(0))
            p.append(f'<rect x="{xa:.1f}" y="{yy-8}" width="{max(1,xb-xa):.1f}" height="16" rx="4" '
                     f'class="{cls}" data-tip="{esc(lab)}: {v:+.2f}% frente al DCA"/>')
            anchor = "start" if v > 0 else "end"
            dx = 8 if v > 0 else -8
            p.append(f'<text x="{px(v)+dx:.1f}" y="{yy+4}" class="val" text-anchor="{anchor}">{v:+.2f}%</text>')
        ybot = ytop + fila * len(filas)
        p.append(f'<line x1="{px(0):.1f}" y1="{ytop}" x2="{px(0):.1f}" y2="{ybot}" class="zero"/>')
        y = ybot + 44
    p.append(eje_x(x0, x1, y - 30, vmin, vmax, [-10, -5, 0, 5, 10, 15], "%"))
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- gráfico 5: barrido
def chart_barrido():
    W, H = 1040, 300
    PADL, PADR, PADT, PADB = 70, 40, 30, 50
    x0, x1 = PADL, W - PADR
    y0, y1 = PADT, H - PADB
    vmin, vmax = -6.5, 0.5

    def px(k):
        return x0 + k / 10 * (x1 - x0)

    def py(v):
        return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    p = [svg_open(W, H, "Cuanto más esperas a que baje, peor")]
    for v in [0, -2, -4, -6]:
        p.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="grid"/>')
        p.append(f'<text x="{x0-10}" y="{py(v)+4:.1f}" class="tick" text-anchor="end">{v}%</text>')
    p.append(f'<line x1="{x0}" y1="{py(0):.1f}" x2="{x1}" y2="{py(0):.1f}" class="zero"/>')
    p.append(f'<text x="{x1}" y="{py(0)-8:.1f}" class="tick" text-anchor="end">DCA</text>')
    pts = " ".join(f"{px(k):.1f},{py(v):.1f}" for k, v in BARRIDO)
    p.append(f'<polyline points="{pts}" class="line"/>')
    for k, v in BARRIDO:
        p.append(f'<circle cx="{px(k):.1f}" cy="{py(v):.1f}" r="5" class="dot-neg" '
                 f'data-tip="esperar una caída del {k}%: {v:+.2f}% frente al DCA"/>')
    p.append(f'<text x="{px(0):.1f}" y="{py(BARRIDO[0][1])-14:.1f}" class="val" text-anchor="middle">−0,33%</text>')
    p.append(f'<text x="{px(10):.1f}" y="{py(BARRIDO[-1][1])+22:.1f}" class="val" text-anchor="end">−5,94%</text>')
    for k in [0, 2, 5, 7, 10]:
        p.append(f'<text x="{px(k):.1f}" y="{y1+22:.1f}" class="tick" text-anchor="middle">{k}%</text>')
    p.append(f'<text x="{(x0+x1)/2:.1f}" y="{H-8}" class="tick" text-anchor="middle">'
             f'descuento exigido antes de comprar</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- gráfico 6: ML
def chart_ml():
    W = 1040
    PADL, PADR = 230, 30
    fila = 34
    H = 60 + fila * len(ML)
    mid = PADL + (W - PADR - PADL) * 0.46
    p = [svg_open(W, H, "R² dentro de muestra frente a edge fuera de muestra")]
    p.append(f'<text x="{PADL+10}" y="26" class="tick">R² DENTRO de muestra</text>')
    p.append(f'<text x="{mid+30}" y="26" class="tick">edge FUERA de muestra (%/mes)</text>')
    esc_in = (mid - PADL - 30) / 0.20
    esc_out = (W - PADR - mid - 40) / 1.0
    for i, (uni, obs, modelo, r2, edge, t) in enumerate(ML):
        yy = 40 + fila * i + fila / 2
        p.append(f'<text x="{PADL-14}" y="{yy+4}" class="rowlab" text-anchor="end">'
                 f'{esc(modelo)} · {esc(uni)}</text>')
        w = max(1.5, r2 * esc_in)
        cls = "bar-ceil" if r2 > 0.05 else "bar-null"
        p.append(f'<rect x="{PADL}" y="{yy-8}" width="{w:.1f}" height="16" rx="4" class="{cls}" '
                 f'data-tip="{esc(modelo)} · {esc(uni)}: R² dentro de muestra {r2:.3f} con {obs:,} observaciones"/>')
        p.append(f'<text x="{PADL+w+8:.1f}" y="{yy+4}" class="val">{r2:.3f}</text>')
        # edge fuera de muestra
        zero = mid + 40 + esc_out * 0.7
        xe = zero + edge * esc_out
        p.append(f'<line x1="{zero:.1f}" y1="{yy-10}" x2="{zero:.1f}" y2="{yy+10}" class="zero"/>')
        p.append(f'<rect x="{min(xe,zero):.1f}" y="{yy-8}" width="{abs(xe-zero):.1f}" height="16" rx="4" '
                 f'class="bar-neg" data-tip="edge fuera de muestra {edge:+.2f}%/mes (t = {t:+.2f})"/>')
        p.append(f'<text x="{min(xe,zero)-8:.1f}" y="{yy+4}" class="val" text-anchor="end">'
                 f'{edge:+.2f} (t={t:+.2f})</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- tablas
def tabla(cabeceras, filas, clases=None):
    th = "".join(f"<th>{esc(c)}</th>" for c in cabeceras)
    tr = []
    for f in filas:
        tds = []
        for j, c in enumerate(f):
            cls = (clases or {}).get(j, "")
            tds.append(f'<td class="{cls}">{c}</td>')
        tr.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(tr)}</tbody></table></div>')


CSS = """
:root{
  --ground:#EFF2F1; --surface:#FBFCFB; --ink:#141C1A; --ink-2:#3D4B48; --muted:#5F6D6A;
  --hair:#D6DEDB; --band:rgba(120,134,130,.16); --band-line:#93A19E;
  --neg:#C0442A; --pos:#07836F; --ceil:#B07A12; --null:#8A9895;
  --stem:#C3CDCA;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#111614; --surface:#19201F; --ink:#E4EBE9; --ink-2:#BCC7C4; --muted:#93A19E;
    --hair:#2B3634; --band:rgba(150,166,162,.14); --band-line:#6B7B78;
    --neg:#D06A50; --pos:#2CA08B; --ceil:#B98D25; --null:#7F8E8B;
    --stem:#38443F;
  }
}
:root[data-theme="dark"]{
  --ground:#111614; --surface:#19201F; --ink:#E4EBE9; --ink-2:#BCC7C4; --muted:#93A19E;
  --hair:#2B3634; --band:rgba(150,166,162,.14); --band-line:#6B7B78;
  --neg:#D06A50; --pos:#2CA08B; --ceil:#B98D25; --null:#7F8E8B;
  --stem:#38443F;
}

*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:17px; line-height:1.65; margin:0;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 100px}
.prose{max-width:68ch}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;font-weight:600;
  text-wrap:balance;line-height:1.18;margin:0}
h1{font-size:clamp(2.3rem,5.2vw,3.6rem);letter-spacing:-.015em}
h2{font-size:clamp(1.5rem,3vw,2rem);margin:76px 0 6px}
h3{font-size:1.14rem;margin:38px 0 4px}
p{margin:14px 0}
a{color:var(--pos)}
strong{font-weight:640}
.lede{font-size:1.2rem;color:var(--ink-2);max-width:60ch;margin-top:18px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;
  letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.dek{color:var(--muted);font-size:.96rem;margin:6px 0 22px;max-width:66ch}

header.top{padding:76px 0 0;border-bottom:1px solid var(--hair);margin-bottom:8px}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;margin:30px 0 34px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;color:var(--muted)}
.meta b{color:var(--ink-2);font-weight:500}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:34px 0 8px}
.tile{background:var(--surface);border:1px solid var(--hair);border-radius:10px;padding:20px 20px 18px}
.tile .n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:2.15rem;
  font-variant-numeric:tabular-nums;line-height:1.05;letter-spacing:-.02em}
.tile .k{font-size:.83rem;color:var(--muted);margin-top:9px}
.tile.bad .n{color:var(--neg)} .tile.flat .n{color:var(--null)} .tile.good .n{color:var(--pos)}

figure{margin:26px 0 8px;background:var(--surface);border:1px solid var(--hair);
  border-radius:10px;padding:20px 22px 14px;overflow-x:auto}
figcaption{font-size:.86rem;color:var(--muted);margin-top:12px;max-width:74ch}
svg.chart{display:block;min-width:660px;overflow:visible}
.facet{font-family:Georgia,serif;font-size:.98rem;fill:var(--ink)}
.facet-meta{font-family:ui-monospace,Menlo,monospace;font-size:.7rem;fill:var(--muted)}
.rowlab{font-size:.8rem;fill:var(--ink-2)}
.tick{font-family:ui-monospace,Menlo,monospace;font-size:.68rem;fill:var(--muted)}
.val{font-family:ui-monospace,Menlo,monospace;font-size:.73rem;fill:var(--ink-2);
  font-variant-numeric:tabular-nums}
.axis{stroke:var(--hair);stroke-width:1}
.grid{stroke:var(--hair);stroke-width:1;stroke-dasharray:2 4}
.zero{stroke:var(--band-line);stroke-width:1.5}
.thresh{stroke:var(--band-line);stroke-width:1;stroke-dasharray:3 3}
.track{stroke:var(--hair);stroke-width:1}
.stem{stroke:var(--stem);stroke-width:1.5}
.band{fill:var(--band)}
.line{fill:none;stroke:var(--neg);stroke-width:2;stroke-linejoin:round}
circle,rect[class^="bar"]{stroke:var(--surface);stroke-width:2}
.dot-neg{fill:var(--neg)} .dot-pos{fill:var(--pos)} .dot-null{fill:var(--null)}
.dot-pos-fill{fill:var(--pos);stroke:none}
.bar-neg{fill:var(--neg)} .bar-ceil{fill:var(--ceil)} .bar-null{fill:var(--null)}
[data-tip]{cursor:crosshair}

.legend{display:flex;flex-wrap:wrap;gap:7px 20px;margin:2px 0 0;font-size:.8rem;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw.band{width:22px;height:11px;border-radius:3px;background:var(--band);
  border:1px solid var(--band-line)}

.tw{overflow-x:auto;margin:24px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:520px}
th,td{padding:9px 13px;text-align:right;border-bottom:1px solid var(--hair);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-family:ui-monospace,Menlo,monospace;font-size:.7rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);font-weight:500;border-bottom:1px solid var(--band-line)}
tbody tr:hover{background:var(--band)}
td.neg{color:var(--neg)} td.pos{color:var(--pos)} td.k{font-family:ui-monospace,Menlo,monospace}

blockquote{margin:24px 0;padding:2px 0 2px 20px;border-left:3px solid var(--band-line);
  color:var(--ink-2);font-size:1.02rem}
.callout{background:var(--surface);border:1px solid var(--hair);border-left:3px solid var(--neg);
  border-radius:8px;padding:18px 22px;margin:26px 0}
.callout.ok{border-left-color:var(--pos)}
.callout h3{margin:0 0 6px;font-size:1.04rem}
.callout p{margin:6px 0 0;font-size:.95rem;color:var(--ink-2)}
ul{max-width:66ch;padding-left:20px} li{margin:9px 0}
hr{border:0;border-top:1px solid var(--hair);margin:64px 0}
footer{color:var(--muted);font-size:.85rem;margin-top:70px;padding-top:24px;
  border-top:1px solid var(--hair);max-width:74ch}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .12s;
  background:var(--ink);color:var(--ground);padding:7px 11px;border-radius:6px;
  font-size:.79rem;max-width:280px;line-height:1.4;z-index:99}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]'); if(!t)return;
    tip.textContent=t.getAttribute('data-tip'); tip.style.opacity='1';
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.opacity!=='1')return;
    var x=e.clientX+14,y=e.clientY+16;
    if(x+tip.offsetWidth>innerWidth-10)x=e.clientX-tip.offsetWidth-14;
    if(y+tip.offsetHeight>innerHeight-10)y=e.clientY-tip.offsetHeight-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]'))tip.style.opacity='0';
  });
})();
"""


def leyenda(items):
    sp = "".join(f'<span><i class="sw {c}" style="background:{b}"></i>{esc(t)}</span>'
                 for c, b, t in items)
    return f'<div class="legend">{sp}</div>'


def build():
    banda_leg = ('<div class="legend">'
                 '<span><i class="sw band"></i>banda del azar (percentiles 5–95 de 200 sorteos)</span>'
                 '<span><i class="sw" style="background:var(--null)"></i>dentro del azar</span>'
                 '<span><i class="sw" style="background:var(--neg)"></i>peor que el equiponderado</span>'
                 '<span><i class="sw" style="background:var(--pos)"></i>mejor que el equiponderado</span>'
                 '</div>')

    filas_ml = []
    for uni, obs, modelo, r2, edge, t in ML:
        filas_ml.append([f"{modelo} · {uni}", f'<span class="k">{obs:,}</span>'.replace(",", "."),
                         f"{r2:.3f}".replace(".", ","),
                         f'<span style="color:var(--neg)">{edge:+.2f} %</span>'.replace(".", ","),
                         f"{t:+.2f}".replace(".", ",")])
    tabla_ml = tabla(["modelo · universo", "observaciones", "R² dentro de muestra",
                      "edge fuera de muestra", "t"], filas_ml)

    filas_rep = []
    for metodo, d in REPARTO:
        filas_rep.append([metodo] + [f"{d[u][0]:.1f} %".replace(".", ",") for u in
                                     ["sectores", "cesta_sin_nvda", "amplio"]]
                         + [f"{d[u][1]:.1f}".replace(".", ",") for u in
                            ["sectores", "cesta_sin_nvda", "amplio"]])
    tabla_rep = tabla(["reparto", "ret. sectores", "ret. sin NVDA", "ret. amplio",
                       "disp. sectores", "disp. sin NVDA", "disp. amplio"], filas_rep)

    largo = tabla(
        ["señal", "1 mes", "12 meses", "36 meses", "60 meses"],
        [["comprar el más caído",
          '<span style="color:var(--neg)">−0,64 %</span>',
          '<span style="color:var(--neg)">−6,1 %</span>',
          '<span style="color:var(--neg)">−27,5 % (t=−2,80)</span>',
          '<span style="color:var(--neg)">−53,2 % (t=−2,65)</span>'],
         ["value 5 años", "+0,06 %", "−2,8 %", "−13,4 %", "−28,4 %"],
         ["momentum 12-1", "+0,27 %", "+3,8 %", "+19,6 %", "+34,0 %"],
         ["comprar el más caído · <b>sectores</b>", "−0,33 %", "−1,8 %", "−4,7 %", "−13,1 %"],
         ["momentum 12-1 · <b>sectores</b>", "+0,08 %", "+0,7 %", "+3,1 %", "−0,7 %"]])

    noticias_html = ""
    if NOTICIAS:
        noticias_html = NOTICIAS.get("html", "")

    return f"""<title>La banda del azar</title>
<style>{CSS}</style>
<div id="tip"></div>
<div class="wrap">

<header class="top">
  <div class="eyebrow">Backtest · DCA · 1999–2026</div>
  <h1>La banda del azar</h1>
  <p class="lede">Dos ideas para mejorar la aportación mensual, medidas contra
  lo que pasa si no decides nada. Ninguna sale de la banda que marca el puro azar.</p>
  <div class="meta">
    <span><b>4</b> universos · hasta <b>27</b> años</span>
    <span><b>10</b> señales + <b>2</b> modelos de ML</span>
    <span>precios ajustados por dividendos</span>
    <span>aportación de <b>1.000 $</b>/mes</span>
  </div>
</header>

<section class="prose">
<div class="tiles">
  <div class="tile flat"><div class="n">50</div><div class="k">percentil en que acaba
    “comprar el más barato” entre los activos disponibles. El azar acaba en el 50</div></div>
  <div class="tile bad"><div class="n">−2,9<span style="font-size:1.1rem"> pp</span></div>
    <div class="k">frente al DCA equiponderado en tu cesta sin NVDA, sobre 222 ventanas
    de 24 aportaciones</div></div>
  <div class="tile flat"><div class="n">1,19</div><div class="k">el mayor |t| de las 10 señales.
    El umbral honesto, corregido por probar tantas, está en 2,9</div></div>
  <div class="tile bad"><div class="n">−27<span style="font-size:1.1rem"> %</span></div>
    <div class="k">de retraso a 3 años del activo más caído frente a la media del
    universo (t = −2,80)</div></div>
</div>
</section>

<h2>1 · Elegir el momento del mes</h2>
<p class="dek">La primera idea: en vez de comprar el día 1, poner una orden límite en el
mínimo que predice el modelo. Se prueba con el modelo reentrenado walk-forward, y
además con un oráculo que acierta el mínimo exacto — la cota que ningún modelo puede superar.</p>

<figure>{chart_estudio1()}
<figcaption>Diferencia de riqueza final frente a comprar el día 1 de cada mes.
El oráculo (ocre) es el techo teórico de toda la arquitectura: <b>ni siquiera
adivinando el mínimo exacto 70 veces seguidas se gana más de un 3,6 % en SPY</b>,
porque el descuento se aplica una vez y no compone. Fallar un mes, en cambio, sí
compone: ese dinero se pierde toda la deriva del activo.</figcaption></figure>

<div class="callout">
<h3>El error del modelo es del tamaño del premio entero</h3>
<p>El descuento medio capturable dentro de un mes en SPY es del <b>3,25 %</b>.
El error medio de la predicción es de <b>3,26 puntos</b>. No hay margen del que vivir.
En BTC el premio es mayor (12,8 %) pero el error también (9,2 pp).</p>
</div>

<figure>{chart_barrido()}
<figcaption>Sin modelo ninguno: exigir un descuento fijo antes de comprar. La caída
es monótona — <b>cuanto más esperas, peor</b>. Ni siquiera exigir un 0 % (comprar cuando
el precio vuelve al cierre del mes anterior) bate al DCA. El problema no es el
modelo: es el mecanismo.</figcaption></figure>

<hr>

<h2>2 · Elegir el activo</h2>
<p class="dek">La segunda idea: aportar lo mismo cada mes, pero al activo que cotiza
más barato respecto a su línea de tendencia. El benchmark honesto no es la mediana de
los activos —esa la bate cualquiera que diversifique— sino el <b>DCA equiponderado</b>:
mismo dinero, mismos activos, cero decisiones.</p>

<figure>{chart_banda()}
{banda_leg}
<figcaption>Cada punto es una señal en un universo. La banda gris es lo que consigue
elegir al azar. <b>Casi todo cae dentro.</b> Lo que sobresale por arriba es el momentum
en tu cesta original, y es un espejismo: esa señal elige NVDA en 70 de los 84 meses con
señal, y NVDA está en la cesta porque hoy sabemos que multiplicó por 143. Quítala y el
+24,5 se queda en +3,9.</figcaption></figure>

<figure>{chart_percentil()}
{leyenda([("", "var(--neg)", "comprar el más barato"), ("", "var(--null)", "elegir al azar")])}
<figcaption>Dónde acaba colocada la inversión entre los activos disponibles. Los dos
puntos se solapan en los cuatro universos: <b>elegir el más barato deja el dinero
exactamente donde lo deja un dado</b>.</figcaption></figure>

<h3>El test que sí tiene potencia</h3>
<p>Las ventanas solapadas ilustran magnitudes, pero 300 ventanas que se pisan mes a
mes no son 300 datos independientes: son unos 12. El test correcto mide la selección
mes a mes — retorno del activo elegido menos la media del universo.</p>

<figure>{chart_tstats()}
{leyenda([("", "var(--neg)", "tu cesta sin NVDA"), ("", "var(--null)", "10 sectores + SPY")])}
<figcaption>Estadístico t con errores Newey-West. <b>Ninguna señal sale de la zona
sin significancia.</b> Lo informativo no es eso, sino el signo: todas las variantes de
“comprar barato” son negativas en los dos universos. La señal informa — informa al revés.</figcaption></figure>

<hr>

<h2>3 · El horizonte que de verdad cobras</h2>
<p class="dek">Un DCA no vende: el dinero que entra en el mes <i>t</i> se queda años.
Medir la señal contra el retorno del mes siguiente evalúa algo que la estrategia no
cobra. Repetido a 12, 36 y 60 meses, el cuadro se agrava.</p>

{largo}

<p>En tu cesta, comprar el activo más caído y mantenerlo tres años deja
<b>27 puntos por detrás</b> de la media del universo, y cinco años, <b>53</b>. Es el único
resultado del estudio que roza la significancia, y va justo en contra de la idea.
En el universo de sectores —el menos contaminado por haber elegido ganadores— todo
vuelve a cero: ahí la señal ni ayuda ni perjudica.</p>

<hr>

<h2>4 · Lo que dice la literatura</h2>
<p class="dek">Antes de seguir probando conviene saber en qué dirección apunta lo publicado.
Resumen de la búsqueda, con las cifras de los papers originales.</p>

<blockquote>El efecto contrario documentado por Tetlock (2007) es de <b>−8,1 puntos básicos</b>
al día siguiente, revierte <b>+6,8</b> durante el resto de la semana, y la suma de los cinco
días <b>no es significativa</b>. Es a nivel de mercado y a horizonte de días.</blockquote>

<ul>
<li><b>El sentimiento de noticias, en transversal, es momentum, no reversión.</b>
Tetlock, Saar-Tsechansky y Macskassy (2008) encuentran infrarreacción: quedan 7,5 pb de
deriva sin incorporar. La estrategia rinde un 21 % anual bruto y <b>deja de ser rentable con
10 pb de costes</b>.</li>
<li><b>Las malas noticias tienen reacción retardada larga</b> (Heston y Sinha, 2017): el activo
peor tratado sigue cayendo durante meses. Evidencia directa contra la idea contraria.</li>
<li><b>Momentum en ETFs sectoriales: no se replica</b> fuera de muestra
(<i>Journal of Asset Management</i>, 2014). Nueve de tus quince activos son sectoriales.</li>
<li><b>Momentum temporal: replicación fallida.</b> Huang, Li, Wang y Zhou (2020) concluyen que
el t-stat agregado no supera los valores críticos de bootstrap.</li>
<li><b>1/N es durísimo de batir.</b> DeMiguel, Garlappi y Uppal (2009): de 14 modelos de
optimización, ninguno bate consistentemente al equiponderado; harían falta <b>~3.000 meses</b>
de historia con 25 activos.</li>
</ul>

<div class="callout">
<h3>Tu cesta tiene ~3 apuestas independientes, no 15</h3>
<p>SPY, QQQ, IWM, XLK, XLY, XLI, XLF, NVDA y MSFT son todos beta de bolsa
estadounidense, con correlaciones de 0,8 a 0,95. Las apuestas realmente independientes
son tres: bolsa USA, metales preciosos y defensivas. La ley fundamental de la gestión
activa dice que el ratio de información escala con la raíz del número de apuestas
<i>independientes</i>.</p>
</div>

<hr>

<h2>5 · El aprendizaje automático</h2>
<p class="dek">Ridge y gradient boosting sobre 8 features en rango transversal, objetivo
desmediado, walk-forward con embargo de un mes y reentreno anual. Su mejor escenario:
universo ancho y 22 años.</p>

{tabla_ml}

<figure>{chart_ml()}
<figcaption>El GBM explica hasta el <b>17 %</b> de la varianza dentro de muestra y transporta
cero —o menos— fuera. Es memorización de ruido. Y hay un techo estructural: quince activos
dan quince opciones al mes; los trabajos de ML transversal que funcionan operan sobre
miles.</figcaption></figure>

<hr>

<h2>6 · Lo único que sí aparece</h2>
<p class="dek">Repartir la aportación inversamente a la volatilidad no es una señal de
selección: no predice nada, sólo dosifica. No sube el retorno, pero baja la dispersión
de resultados de forma consistente en los tres universos.</p>

{tabla_rep}

<div class="callout ok">
<h3>Mismo retorno, menos varianza</h3>
<p>Es exactamente lo que predice la literatura: la ponderación por volatilidad inversa
mejora la relación rentabilidad/riesgo con muy poco error de estimación, porque sólo
estima segundos momentos. No es alfa, pero es replicable — y es lo contrario de lo que
pasa con las señales de selección.</p>
</div>

{noticias_html}

<hr>

<h2>Cómo leer todo esto</h2>
<p>“No hay señal” y “no se puede detectar señal” no son lo mismo, y aquí aplica lo segundo.
Con ~250 meses y un tracking error mensual del 3 %, para dar por bueno un edge real de
0,30 % al mes con t = 2 harían falta unos <b>400 meses</b>, 33 años. El momentum con
t = 0,72 es perfectamente compatible con un edge real de +0,27 % mensual: no está
refutado, está <b>sin resolver</b>, y no se resolverá con estos datos.</p>

<p>Lo que sí está establecido, porque el signo se repite en todos los universos y
horizontes, es que <b>la dirección de “comprar lo más caído” no ayuda</b>, y en la cesta
concreta perjudica. Parte de ese signo negativo no mide el fracaso del value: mide que la
cesta se eligió mirando hacia atrás, con NVDA y MSFT dentro porque ya sabemos que ganaron.</p>

<h3>Limitaciones</h3>
<ul>
<li>Cesta con sesgo de supervivencia. Cuantificado, no eliminado.</li>
<li>Sin comisiones, spreads ni fiscalidad. La estrategia de selección rota y el
equiponderado no: en real quedaría aún peor.</li>
<li>Una sola realización histórica, aunque incluya puntocom, 2008, covid y 2022.</li>
<li>Las ventanas solapadas no son observaciones independientes; las conclusiones se
apoyan en los tests mes a mes.</li>
</ul>

<footer>
Datos: precios diarios ajustados por dividendos (Yahoo Finance), agregados a mensual ·
20 activos · 1993–2026. Sentimiento de noticias: GDELT DOC 2.0, tono medio de la
cobertura, desde 2017. Todo el código y los tests están en el repositorio; cada cifra
de esta página se reproduce con <code>python -m backtest.run_robustness --all</code>.
</footer>

</div>
<script>{JS}</script>
"""


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(build())
    print(f"escrito {OUT} ({len(build()):,} bytes)")
