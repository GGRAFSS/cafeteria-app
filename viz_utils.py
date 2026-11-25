"""Gráficas del temario: histogramas, boxplots, barras, ajustes, violín,
mapa de correlación y árbol bayesiano."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    return fig, ax


def plot_hist(series, title, xlabel, bins=20):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.3)
    return fig


def plot_box(series, title, ylabel):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    ax.boxplot(s, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#DD8452"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels([ylabel])
    ax.grid(alpha=0.3)
    return fig


def plot_box_by_group(df, value_col, group_col, title, ylabel):
    """Boxplot comparativo por grupo (compatible con Matplotlib 3.7–3.9+)."""
    fig, ax = _fig()
    grouped = df.groupby(group_col)
    groups = [g[value_col].dropna().values for _, g in grouped]
    labels = [str(k) for k, _ in grouped]

    ax.boxplot(groups, patch_artist=True,
               boxprops=dict(facecolor="#55A868"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig


def plot_bars(df, col, title):
    fig, ax = _fig()
    vc = df[col].fillna("(Nulo)").value_counts()
    ax.bar(vc.index.astype(str), vc.values, color="#C44E52", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_exponential_fit(series, title="Ajuste Exponencial"):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    ax.hist(s, bins=25, density=True, alpha=0.6,
            color="#4C72B0", edgecolor="white", label="Datos")
    lam = 1.0 / s.mean()
    x = np.linspace(0, s.max(), 300)
    ax.plot(x, stats.expon.pdf(x, scale=1 / lam),
            "r-", lw=2, label=f"Exp(λ={lam:.3f})")

    ks_stat, p = stats.kstest(s, stats.expon(scale=1 / lam).cdf)

    ax.set_title(f"{title}\nKS={ks_stat:.3f}, p={p:.3f}")
    ax.set_xlabel("Tiempo_Espera_Min")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_normal_fit(series, title="Ajuste Normal"):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    ax.hist(s, bins=25, density=True, alpha=0.6,
            color="#55A868", edgecolor="white", label="Datos")
    mu, sigma = s.mean(), s.std(ddof=1)
    x = np.linspace(s.min(), s.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma),
            "r-", lw=2, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")

    ks_stat, p = stats.kstest(s, stats.norm(loc=mu, scale=sigma).cdf)

    ax.set_title(f"{title}\nKS={ks_stat:.3f}, p={p:.3f}")
    ax.set_xlabel("Tiempo_Servicio_Min")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_box_by_binary(df, value_col, binary_col, title, ylabel):
    """Boxplot comparativo para una variable binaria (Sí/No)."""
    fig, ax = _fig()

    order = [v for v in ["No", "Sí"] if v in df[binary_col].unique()]
    grouped = df.groupby(binary_col)
    groups = [grouped.get_group(k)[value_col].dropna().values for k in order]
    labels = order

    bp = ax.boxplot(groups, patch_artist=True, widths=0.5)
    colors = ["#4C72B0", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(binary_col)
    ax.grid(alpha=0.3, axis="y")

    for i, g in enumerate(groups):
        if len(g) > 0:
            ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                    f"μ={g.mean():.1f}", ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="gray", alpha=0.8))

    return fig


def plot_grouped_bars(df, value_col, group_col, title, ylabel="Conteo"):
    """
    Barras agrupadas: distribución de value_col (categórica) por group_col.
    Compatible con columnas Int64 que contienen pd.NA.
    """
    fig, ax = _fig()

    # Convertir a object primero, luego reemplazar nulos, luego a str
    valores = df[value_col].astype(object)
    valores = valores.where(valores.notna(), "(Nulo)")
    valores = valores.astype(str)

    ct = pd.crosstab(df[group_col], valores).sort_index()

    cmap = plt.get_cmap("viridis")
    n_vals = ct.shape[1]
    colors = [cmap(i / max(n_vals - 1, 1)) for i in range(n_vals)]

    ct.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_col)
    ax.legend(title=value_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


def plot_violin(df, value_col, group_col, title, ylabel):
    """Violín de `value_col` por grupo, con caja interna y puntos observados."""
    datos = df[[value_col, group_col]].copy()
    datos[value_col] = pd.to_numeric(datos[value_col], errors="coerce")
    datos = datos.dropna(subset=[value_col, group_col])

    if datos.empty:
        raise ValueError("No hay datos válidos para el gráfico de violín.")

    grupos = sorted(datos[group_col].astype(str).unique())
    series = [datos.loc[datos[group_col].astype(str) == g, value_col].values
              for g in grupos]
    series = [s for s in series if len(s) > 0]
    if not series:
        raise ValueError("Ningún grupo tiene valores para el violín.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    posiciones = np.arange(1, len(series) + 1)
    cmap = plt.get_cmap("viridis")
    colores = [cmap(i / max(len(series) - 1, 1)) for i in range(len(series))]

    partes = ax.violinplot(series, positions=posiciones, widths=0.7,
                           showmeans=True, showextrema=True,
                           showmedians=False)
    for cuerpo, color in zip(partes["bodies"], colores):
        cuerpo.set_facecolor(color)
        cuerpo.set_edgecolor("white")
        cuerpo.set_alpha(0.65)
    for clave in ("cmeans", "cmins", "cmaxes", "cbars"):
        if clave in partes:
            partes[clave].set_color("#333333")
            partes[clave].set_linewidth(1.2)

    ax.boxplot(series, positions=posiciones, widths=0.12, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor="white", alpha=0.85),
               medianprops=dict(color="#C44E52", linewidth=1.4))

    rng = np.random.default_rng(0)
    for pos, valores, color in zip(posiciones, series, colores):
        x = pos + rng.normal(0, 0.045, size=len(valores))
        ax.scatter(x, valores, s=9, color=color, alpha=0.55,
                   edgecolors="none", zorder=3)

    etiquetas = [f"{g}\n(n={len(s)})" for g, s in zip(grupos, series)]
    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_col)
    ax.grid(alpha=0.3, axis="y")

    return fig


def plot_matriz_correlacion(df, cols=None,
                            title="Matriz de correlación de Pearson"):
    """Mapa de calor de la matriz de correlación (usa stats_utils)."""
    from stats_utils import matriz_correlacion

    corr = matriz_correlacion(df, cols)
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.55 * n), max(4.8, 1.35 * n)))

    imagen = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            valor = corr.values[i, j]
            if pd.isna(valor):
                continue
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if abs(valor) > 0.6 else "#222222")

    ax.set_title(title)
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04,
                 label="Coeficiente r")
    fig.tight_layout()

    return fig


def plot_arbol_bayes(resultado: dict, etiqueta_a: str = "A",
                     etiqueta_b: str = "B"):
    """Árbol de decisión bayesiano: prior → verosimilitud → posterior.

    La rama superior representa el evento A y la inferior su complemento; las
    hojas muestran la probabilidad conjunta P(A∩B) = P(B|A)·P(A). La hoja A∩B
    se resalta porque es el numerador del posterior P(A|B).
    """
    p_a = float(resultado.get("p_a", 0.0))
    p_no_a = float(resultado.get("p_no_a", 1.0 - p_a))
    p_b_a = float(resultado.get("p_b_dado_a", 0.0))
    p_b_no_a = float(resultado.get("p_b_dado_no_a", 0.0))
    posterior = float(resultado.get("p_a_dado_b", 0.0))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    azul, naranja = "#4C72B0", "#DD8452"
    rojo = "#C44E52"

    def nodo(x, y, texto, color, tam=8.5, borde="white", grosor=1.1):
        ax.text(x, y, texto, ha="center", va="center", fontsize=tam,
                fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                          edgecolor=borde, linewidth=grosor))

    def rama(x0, y0, x1, y1, color="#7F7F7F"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                    shrinkA=16, shrinkB=18))

    # Nivel 1: población
    nodo(0.5, 0.93, "Clientes de la cafetería", "#333333")
    # Nivel 2: A y ¬A
    rama(0.5, 0.90, 0.25, 0.68, azul)
    rama(0.5, 0.90, 0.75, 0.68, naranja)
    nodo(0.25, 0.62, f"A: es {etiqueta_a}\nP(A) = {p_a:.3f}", azul)
    nodo(0.75, 0.62, f"¬A: no es {etiqueta_a}\nP(¬A) = {p_no_a:.3f}",
         naranja)

    def etiqueta_rama(x0, y0, x1, y1, texto, color):
        """Escribe la verosimilitud sobre la rama, alineada a su pendiente."""
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x1, y1))
        angulo = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if angulo > 90:
            angulo -= 180
        elif angulo < -90:
            angulo += 180
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, texto, rotation=angulo,
                rotation_mode="anchor", ha="center", va="bottom",
                fontsize=7.2, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # Nivel 3: hojas B / ¬B
    hojas = [
        (0.12, "B", p_a * p_b_a, azul, True),
        (0.38, "¬B", p_a * (1 - p_b_a), "#8FA8C8", False),
        (0.62, "B", p_no_a * p_b_no_a, "#E5B98F", False),
        (0.88, "¬B", p_no_a * (1 - p_b_no_a), naranja, False),
    ]
    origenes = [0.25, 0.25, 0.75, 0.75]
    for (x, simbolo, prob, color, resaltado), x0 in zip(hojas, origenes):
        rama(x0, 0.57, x, 0.35, color if resaltado else "#9A9A9A")
        nodo(x, 0.28, f"{simbolo}\nP = {prob:.3f}", color, tam=8.0,
             borde=rojo if resaltado else "white",
             grosor=2.2 if resaltado else 1.1)

    # Verosimilitudes sobre cada rama
    etiqueta_rama(0.25, 0.57, 0.12, 0.35, f"P(B|A) = {p_b_a:.3f}", azul)
    etiqueta_rama(0.25, 0.57, 0.38, 0.35, f"P(¬B|A) = {1 - p_b_a:.3f}", azul)
    etiqueta_rama(0.75, 0.57, 0.62, 0.35, f"P(B|¬A) = {p_b_no_a:.3f}", naranja)
    etiqueta_rama(0.75, 0.57, 0.88, 0.35, f"P(¬B|¬A) = {1 - p_b_no_a:.3f}",
                  naranja)

    ax.text(0.5, 0.12,
            f"P(A|B) = P(B|A)·P(A) / P(B) = {posterior:.4f}",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F2F2F2",
                      edgecolor=rojo, linewidth=1.2))
    fig.text(0.5, 0.015, f"B = «{etiqueta_b}»", ha="center", fontsize=8.5,
             color="#444444")
    ax.set_title(f"Árbol bayesiano: {etiqueta_a} dado «{etiqueta_b}»",
                 fontsize=10.5)
    fig.tight_layout()

    return fig


# [FASE 2 - REPORTE] Agregado: plot_violin, plot_matriz_correlacion,
# plot_arbol_bayes