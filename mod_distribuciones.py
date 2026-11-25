"""
Módulo 3: Distribuciones de Probabilidad (Unidad 2)
- Simulación de variables aleatorias (Poisson para llegadas, Exponencial para espera)
- Ajuste de curvas de densidad (Normal para servicio, Exponencial para espera)
- Validación de supuestos: test KS + Q-Q plots
- Simulaciones de apoyo: Bernoulli, Binomial, Geométrica, Hipergeométrica, Gamma
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


# ============================================================
# 6. SIMULACIONES DE APOYO (Bernoulli, Binomial, Geométrica,
#    Hipergeométrica y Gamma)
# ============================================================
def _resumen_sim(nombre: str, parametros: dict, muestra, media_teorica: float,
                 varianza_teorica: float) -> dict:
    """Resumen numérico de una simulación listo para tabla del reporte."""
    valores = np.asarray(muestra, dtype=float)
    var_teorica = max(float(varianza_teorica), 0.0)
    resumen = dict(parametros)
    resumen.update({
        "distribucion": nombre,
        "n_simulaciones": int(valores.size),
        "media_teorica": float(media_teorica),
        "varianza_teorica": var_teorica,
        "desv_std_teorica": float(np.sqrt(var_teorica)),
        "media_empirica": float(valores.mean()),
        "desv_std_empirica": (float(valores.std(ddof=1))
                              if valores.size > 1 else 0.0),
        "diferencia_media": float(valores.mean() - media_teorica),
        "minimo": float(valores.min()),
        "maximo": float(valores.max()),
    })
    return resumen


def _grafico_sim(muestra, titulo: str, xlabel: str, teorica=None,
                 discreta: bool = True, bins: int = 30):
    """Barras (discretas) o histograma (continuas) frente al valor teórico."""
    valores = np.asarray(muestra, dtype=float)
    fig, ax = _fig()

    if discreta:
        base = int(np.floor(valores.min()))
        conteos = np.bincount(valores.astype(int) - base)
        ejes = np.arange(base, base + conteos.size)
        ax.bar(ejes, conteos / valores.size, color="#4C72B0",
               edgecolor="white", alpha=0.85, label="Simulación")
        ax.set_ylabel("Probabilidad")
    else:
        ax.hist(valores, bins=bins, density=True, color="#4C72B0",
                edgecolor="white", alpha=0.85, label="Simulación")
        ax.set_ylabel("Densidad")

    if teorica is not None:
        x_teorica, y_teorica = teorica
        ax.plot(x_teorica, y_teorica, "-o" if discreta else "-",
                color="#C44E52", lw=2, ms=5, label="Valor teórico")
        if discreta:
            ax.set_xlim(float(x_teorica.min()) - 0.6,
                        float(x_teorica.max()) + 0.6)

    ax.axvline(valores.mean(), color="#55A868", ls="--", lw=1.5,
               label=f"Media simulada = {valores.mean():.3f}")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def simular_bernoulli(p: float = 0.5, n: int = 1000, semilla=None):
    """
    Simula `n` ensayos Bernoulli(p): 1 = éxito (probabilidad p), 0 = fracaso.

    Devuelve {"fig": Figure, "resumen": dict}. La media teórica es p y la
    varianza p·(1 − p).
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p debe estar estrictamente entre 0 y 1.")
    n = int(n)
    if n < 2:
        raise ValueError("n debe ser al menos 2 simulaciones.")

    generador = np.random.default_rng(semilla)
    muestra = generador.binomial(1, p, size=n)

    x = np.array([0.0, 1.0])
    fig = _grafico_sim(
        muestra, f"Simulación Bernoulli (p = {p:.2f})",
        "Resultado (0 = fracaso, 1 = éxito)",
        teorica=(x, stats.bernoulli(p).pmf(x)))

    resumen = _resumen_sim("Bernoulli", {"p": p}, muestra, p, p * (1 - p))
    return {"fig": fig, "resumen": resumen}


def simular_binomial(n_ensayos: int = 10, p: float = 0.5, n: int = 1000,
                     semilla=None):
    """
    Simula `n` réplicas de X ~ Binomial(n_ensayos, p).

    Devuelve {"fig": Figure, "resumen": dict}. Media n·p y varianza
    n·p·(1 − p).
    """
    n_ensayos = int(n_ensayos)
    p = float(p)
    n = int(n)
    if n_ensayos < 1:
        raise ValueError("n_ensayos debe ser al menos 1.")
    if not 0.0 < p < 1.0:
        raise ValueError("p debe estar estrictamente entre 0 y 1.")
    if n < 2:
        raise ValueError("n debe ser al menos 2 simulaciones.")

    generador = np.random.default_rng(semilla)
    muestra = generador.binomial(n_ensayos, p, size=n)

    x = np.arange(0, n_ensayos + 1)
    fig = _grafico_sim(
        muestra, f"Simulación Binomial (n = {n_ensayos}, p = {p:.2f})",
        "Número de éxitos",
        teorica=(x, stats.binom.pmf(x, n_ensayos, p)))

    media = n_ensayos * p
    resumen = _resumen_sim(
        "Binomial", {"n_ensayos": n_ensayos, "p": p}, muestra,
        media, media * (1 - p))
    return {"fig": fig, "resumen": resumen}


def simular_geometrica(p: float = 0.3, n: int = 1000, semilla=None):
    """
    Simula `n` tiempos de espera X ~ Geométrica(p) (ensayos hasta el 1er éxito).

    Devuelve {"fig": Figure, "resumen": dict}. Soporte 1, 2, 3, …; media 1/p y
    varianza (1 − p)/p².
    """
    p = float(p)
    if not 0.0 < p <= 1.0:
        raise ValueError("p debe estar en (0, 1].")
    n = int(n)
    if n < 2:
        raise ValueError("n debe ser al menos 2 simulaciones.")

    generador = np.random.default_rng(semilla)
    muestra = generador.geometric(p, size=n)

    alto = int(min(max(5, int(muestra.max())), 60))
    x = np.arange(1, alto + 1)
    fig = _grafico_sim(
        muestra, f"Simulación Geométrica (p = {p:.2f})",
        "Ensayos hasta el primer éxito",
        teorica=(x, stats.geom.pmf(x, p)))

    resumen = _resumen_sim(
        "Geométrica", {"p": p}, muestra, 1.0 / p, (1.0 - p) / (p ** 2))
    return {"fig": fig, "resumen": resumen}


def simular_hipergeometrica(N: int = 200, K: int = 60, n: int = 10,
                            n_sim: int = 1000, semilla=None):
    """
    Simula `n_sim` extracciones SIN reemplazo de `n` elementos tomados de una
    población de tamaño `N` con `K` éxitos (distribución hipergeométrica).

    Devuelve {"fig": Figure, "resumen": dict}. Media n·K/N y varianza
    n·(K/N)·(1 − K/N)·(N − n)/(N − 1).
    """
    N, K, n, n_sim = int(N), int(K), int(n), int(n_sim)
    if N < 2:
        raise ValueError("N debe ser al menos 2.")
    if not 0 <= K <= N:
        raise ValueError("K debe cumplir 0 <= K <= N.")
    if not 1 <= n <= N:
        raise ValueError("n debe cumplir 1 <= n <= N.")
    if n_sim < 2:
        raise ValueError("n_sim debe ser al menos 2 simulaciones.")

    generador = np.random.default_rng(semilla)
    muestra = generador.hypergeometric(K, N - K, n, size=n_sim)

    bajo = max(0, n - (N - K))
    alto = min(n, K)
    x = np.arange(bajo, alto + 1)
    fig = _grafico_sim(
        muestra, f"Simulación Hipergeométrica (N = {N}, K = {K}, n = {n})",
        "Éxitos en la muestra",
        teorica=(x, stats.hypergeom.pmf(x, N, K, n)))

    proporcion = K / N
    varianza = n * proporcion * (1 - proporcion)
    if N > 1:
        varianza *= (N - n) / (N - 1)
    resumen = _resumen_sim(
        "Hipergeométrica", {"N": N, "K": K, "n": n, "n_simulaciones": n_sim},
        muestra, n * proporcion, varianza)
    return {"fig": fig, "resumen": resumen}


def simular_gamma(shape: float = 2.0, scale: float = 1.5, n: int = 1000,
                  semilla=None):
    """
    Simula `n` valores X ~ Gamma(shape, scale).

    Devuelve {"fig": Figure, "resumen": dict}. Media shape·scale y varianza
    shape·scale². Con shape = 1 la Gamma se reduce a la Exponencial.
    """
    shape = float(shape)
    scale = float(scale)
    n = int(n)
    if shape <= 0 or scale <= 0:
        raise ValueError("shape y scale deben ser positivos.")
    if n < 2:
        raise ValueError("n debe ser al menos 2 simulaciones.")

    generador = np.random.default_rng(semilla)
    muestra = generador.gamma(shape, scale, size=n)

    limite = float(stats.gamma.ppf(0.999, shape, scale=scale))
    x = np.linspace(0.0, max(limite, 1e-6), 300)
    fig = _grafico_sim(
        muestra, f"Simulación Gamma (shape = {shape:g}, scale = {scale:g})",
        "Valor", teorica=(x, stats.gamma.pdf(x, shape, scale=scale)),
        discreta=False)

    resumen = _resumen_sim(
        "Gamma", {"shape": shape, "scale": scale}, muestra,
        shape * scale, shape * scale ** 2)
    return {"fig": fig, "resumen": resumen}


# [FASE 2 - REPORTE] Agregado: simular_bernoulli, simular_binomial,
# simular_geometrica, simular_hipergeometrica, simular_gamma