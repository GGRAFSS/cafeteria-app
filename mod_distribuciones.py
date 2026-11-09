"""
Módulo 3: Distribuciones de Probabilidad (Unidad 2)
- Simulación de variables aleatorias (Poisson para llegadas, Exponencial para espera)
- Ajuste de curvas de densidad (Normal para servicio, Exponencial para espera)
- Validación de supuestos: test KS + Q-Q plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ============================================================
# 1. SIMULACIÓN POISSON (llegadas por minuto)
# ============================================================
def simular_poisson_llegadas(df: pd.DataFrame, minutos: int = 240, n_sim: int = 1):
    """
    Simula el número de llegadas por minuto usando una distribución Poisson.

    - Calcula λ empírico: promedio de llegadas por minuto en el horario observado.
    - Genera n_sim simulaciones de `minutos` minutos cada una.
    - Compara la distribución simulada con la empírica (conteo por minuto real).

    Parámetros:
      df: DataFrame con Hora_Llegada (float, horas decimales)
      minutos: duración del periodo (7:00–11:00 = 240 min)
      n_sim: número de simulaciones (para promedio)

    Devuelve:
      - fig_hist: histograma comparativo (empírico vs Poisson simulada)
      - fig_qq: Q-Q plot Poisson
      - resumen: dict con λ, media empírica, media simulada, KS p-value
    """
    if df.empty or "Hora_Llegada" not in df.columns:
        return None, None, {}

    # Convertir hora decimal a minuto del periodo (0 = 7:00, 240 = 11:00)
    minutos_llegada = ((df["Hora_Llegada"] - 7.0) * 60).dropna().astype(int)
    minutos_llegada = minutos_llegada[(minutos_llegada >= 0) & (minutos_llegada < minutos)]

    # Conteo empírico: cuántas llegadas hubo en cada minuto
    conteo_empirico = np.zeros(minutos, dtype=int)
    for m in minutos_llegada:
        conteo_empirico[m] += 1

    lam = conteo_empirico.mean()  # λ = promedio de llegadas por minuto

    # Simulación
    simulaciones = np.random.poisson(lam=lam, size=(n_sim, minutos))
    sim_prom = simulaciones.mean(axis=0)

    # KS test: comparar empírico vs simulada
    ks_stat, p_val = stats.kstest(conteo_empirico, stats.poisson(mu=lam).cdf)

    # Histograma comparativo
    fig_hist, ax = _fig()
    max_val = max(conteo_empirico.max(), sim_prom.max()) + 1
    bins = np.arange(0, max_val + 1) - 0.5
    ax.hist(conteo_empirico, bins=bins, alpha=0.6, label="Empírico (real)",
            color="#4C72B0", edgecolor="white", density=True)
    ax.hist(sim_prom, bins=bins, alpha=0.6, label=f"Poisson(λ={lam:.2f})",
            color="#C44E52", edgecolor="white", density=True)
    ax.set_title("Llegadas por minuto: empírico vs Poisson simulada")
    ax.set_xlabel("Número de llegadas en un minuto")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.3)

    # Q-Q plot
    fig_qq, ax2 = _fig()
    stats.probplot(conteo_empirico, dist="poisson", sparams=(lam,), plot=ax2)
    ax2.set_title(f"Q-Q plot — Poisson(λ={lam:.2f})\nKS p-value = {p_val:.4f}")
    ax2.grid(alpha=0.3)

    resumen = {
        "lambda": lam,
        "media_empirica": conteo_empirico.mean(),
        "media_simulada": sim_prom.mean(),
        "ks_stat": ks_stat,
        "p_value": p_val,
        "n_minutos": minutos,
        "total_llegadas": int(conteo_empirico.sum()),
    }

    return fig_hist, fig_qq, resumen


# ============================================================
# 2. Q-Q PLOTS (Normal y Exponencial)
# ============================================================
def qq_plot_normal(series, title="Q-Q plot — Normal"):
    """Q-Q plot para verificar normalidad de una serie."""
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return fig

    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(f"{title}\nR² = {r**2:.4f}")
    ax.set_xlabel("Cuantiles teóricos (Normal)")
    ax.set_ylabel("Cuantiles observados")
    ax.grid(alpha=0.3)
    return fig


def qq_plot_exponencial(series, title="Q-Q plot — Exponencial"):
    """Q-Q plot para verificar ajuste exponencial de una serie."""
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return fig

    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="expon", plot=ax)
    ax.set_title(f"{title}\nR² = {r**2:.4f}")
    ax.set_xlabel("Cuantiles teóricos (Exponencial)")
    ax.set_ylabel("Cuantiles observados")
    ax.grid(alpha=0.3)
    return fig


# ============================================================
# 3. TABLA RESUMEN DE BONDAD DE AJUSTE (KS)
# ============================================================
def tabla_bondad_ajuste(df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula KS para las distribuciones ajustadas:
      - Tiempo_Espera_Min ~ Exponencial(λ = 1/media)
      - Tiempo_Servicio_Min ~ Normal(μ, σ)

    Devuelve una tabla con: Variable, Distribución, Parámetros, KS, p-value, Conclusión.
    """
    rows = []

    # Espera ~ Exponencial
    if "Tiempo_Espera_Min" in df.columns:
        s = pd.to_numeric(df["Tiempo_Espera_Min"], errors="coerce").dropna()
        s = s[s > 0]
        if not s.empty:
            lam = 1.0 / s.mean()
            ks, p = stats.kstest(s, stats.expon(scale=1/lam).cdf)
            rows.append({
                "Variable": "Tiempo_Espera_Min",
                "Distribución": "Exponencial",
                "Parámetros": f"λ = {lam:.4f}",
                "KS": round(ks, 4),
                "p-value": round(p, 4),
                "¿Se acepta? (α=0.05)": "Sí" if p > 0.05 else "No",
            })

    # Servicio ~ Normal
    if "Tiempo_Servicio_Min" in clean_df.columns:
        s = pd.to_numeric(clean_df["Tiempo_Servicio_Min"], errors="coerce").dropna()
        s = s[s > 0]
        if not s.empty:
            mu, sigma = s.mean(), s.std(ddof=1)
            ks, p = stats.kstest(s, stats.norm(loc=mu, scale=sigma).cdf)
            rows.append({
                "Variable": "Tiempo_Servicio_Min",
                "Distribución": "Normal",
                "Parámetros": f"μ = {mu:.4f}, σ = {sigma:.4f}",
                "KS": round(ks, 4),
                "p-value": round(p, 4),
                "¿Se acepta? (α=0.05)": "Sí" if p > 0.05 else "No",
            })

    return pd.DataFrame(rows)


# ============================================================
# 4. GRÁFICA COMPUESTA: histograma + curva + Q-Q
# ============================================================
def plot_ajuste_con_qq(series, dist="expon", title_prefix="Tiempo de Espera"):
    """
    Genera una figura con dos subplots:
      - Izquierda: histograma + curva teórica ajustada
      - Derecha: Q-Q plot

    dist: "expon" o "norm"
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        fig, ax = _fig()
        ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Izquierda: histograma + curva ---
    ax1.hist(s, bins=25, density=True, alpha=0.6, color="#4C72B0",
             edgecolor="white", label="Datos")
    x = np.linspace(s.min(), s.max(), 300)

    if dist == "expon":
        lam = 1.0 / s.mean()
        ax1.plot(x, stats.expon.pdf(x, scale=1/lam), "r-", lw=2,
                 label=f"Exp(λ={lam:.3f})")
        ks, p = stats.kstest(s, stats.expon(scale=1/lam).cdf)
        dist_name = "Exponencial"
    else:
        mu, sigma = s.mean(), s.std(ddof=1)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2,
                 label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
        ks, p = stats.kstest(s, stats.norm(loc=mu, scale=sigma).cdf)
        dist_name = "Normal"

    ax1.set_title(f"{title_prefix} — Ajuste {dist_name}\nKS={ks:.4f}, p={p:.4f}")
    ax1.set_xlabel(series.name if hasattr(series, "name") else "Valor")
    ax1.set_ylabel("Densidad")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Derecha: Q-Q plot ---
    (osm, osr), (slope, intercept, r) = stats.probplot(
        s, dist=dist, plot=ax2
    )
    ax2.set_title(f"Q-Q plot — {dist_name}\nR² = {r**2:.4f}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig