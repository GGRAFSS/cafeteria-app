"""
Módulo 4: Pruebas de Hipótesis y Validación de Modelos (Unidad 3)
- Pruebas paramétricas: t de Student para muestras independientes
- Pruebas no paramétricas: Mann-Whitney U
- Validación de supuestos: Shapiro-Wilk (normalidad), Levene (homocedasticidad)
- Modelo de clasificación: regresión logística + matriz de confusión + ROC
- Comparación de más de dos grupos: ANOVA de un factor + chi-cuadrado
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report,
)


def _fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ============================================================
# 1. PRUEBAS DE HIPÓTESIS PARA DOS GRUPOS
# ============================================================
def comparar_dos_grupos(df: pd.DataFrame, valor_col: str, grupo_col: str,
                        grupo_a: str, grupo_b: str) -> dict:
    """
    Compara una variable numérica entre dos grupos con:
      - Prueba de normalidad (Shapiro-Wilk) por grupo
      - Prueba de homocedasticidad (Levene)
      - Prueba t de Student (paramétrica)
      - Prueba de Mann-Whitney U (no paramétrica)

    Devuelve un dict con todos los resultados listos para mostrar en Streamlit.
    """
    if valor_col not in df.columns or grupo_col not in df.columns:
        return {"error": f"Columnas {valor_col} o {grupo_col} no encontradas."}

    a = pd.to_numeric(df[df[grupo_col] == grupo_a][valor_col], errors="coerce").dropna()
    b = pd.to_numeric(df[df[grupo_col] == grupo_b][valor_col], errors="coerce").dropna()

    if len(a) < 3 or len(b) < 3:
        return {"error": f"Se necesitan al menos 3 datos por grupo (n_a={len(a)}, n_b={len(b)})."}

    # Normalidad por grupo
    sh_a_stat, sh_a_p = stats.shapiro(a) if len(a) >= 3 else (np.nan, np.nan)
    sh_b_stat, sh_b_p = stats.shapiro(b) if len(b) >= 3 else (np.nan, np.nan)

    # Homocedasticidad
    lev_stat, lev_p = stats.levene(a, b)

    # Prueba t (Welch si varianzas desiguales)
    equal_var = lev_p > 0.05
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=equal_var)

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Tamaño del efecto (Cohen's d)
    pooled_std = np.sqrt(((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2) /
                         (len(a) + len(b) - 2))
    cohen_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan

    return {
        "valor_col": valor_col,
        "grupo_col": grupo_col,
        "grupo_a": grupo_a,
        "grupo_b": grupo_b,
        "n_a": len(a),
        "n_b": len(b),
        "media_a": a.mean(),
        "media_b": b.mean(),
        "std_a": a.std(),
        "std_b": b.std(),
        "shapiro_a_p": sh_a_p,
        "shapiro_b_p": sh_b_p,
        "levene_p": lev_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
        "ttest_equal_var": equal_var,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_p,
        "cohen_d": cohen_d,
        "interpretacion_d": _interpretar_cohen_d(cohen_d),
    }


def _interpretar_cohen_d(d):
    if d is None or np.isnan(d):
        return "N/A"
    d = abs(d)
    if d < 0.2:
        return "Muy pequeño"
    elif d < 0.5:
        return "Pequeño"
    elif d < 0.8:
        return "Mediano"
    else:
        return "Grande"


def plot_comparacion_grupos(df: pd.DataFrame, valor_col: str, grupo_col: str,
                            grupo_a: str, grupo_b: str, titulo: str):
    """
    Genera un boxplot + jitter para visualizar los dos grupos comparados.
    """
    a = pd.to_numeric(df[df[grupo_col] == grupo_a][valor_col], errors="coerce").dropna()
    b = pd.to_numeric(df[df[grupo_col] == grupo_b][valor_col], errors="coerce").dropna()

    fig, ax = _fig()
    bp = ax.boxplot([a, b], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jitter
    rng = np.random.default_rng(42)
    x_a = rng.normal(1, 0.04, len(a))
    x_b = rng.normal(2, 0.04, len(b))
    ax.scatter(x_a, a, alpha=0.5, s=15, color="#1f4e8c")
    ax.scatter(x_b, b, alpha=0.5, s=15, color="#8c2c3a")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([grupo_a, grupo_b])
    ax.set_ylabel(valor_col)
    ax.set_title(titulo)
    ax.grid(alpha=0.3, axis="y")

    return fig


# ============================================================
# 2. REGRESIÓN LOGÍSTICA + MÉTRICAS DE CLASIFICACIÓN
# ============================================================
def entrenar_regresion_logistica(df: pd.DataFrame, target_col: str = "Es_Hora_Pico"):
    """
    Entrena una regresión logística para predecir Es_Hora_Pico (Sí/No)
    a partir de variables numéricas y categóricas.

    Devuelve un dict con:
      - fig_cm: matriz de confusión
      - fig_roc: curva ROC
      - metricas: accuracy, precision, recall, f1, auc
      - reporte: classification_report como DataFrame
      - y_test, y_pred, y_proba: para análisis adicional
      - error: si no se pudo entrenar
    """
    if target_col not in df.columns:
        return {"error": f"Columna objetivo {target_col} no encontrada."}

    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(str).map({"Sí": 1, "No": 0})
    if y.isna().any():
        return {"error": "La columna objetivo debe tener valores 'Sí' / 'No'."}

    # Features: numéricas + categóricas codificadas
    feature_cols_num = ["Tiempo_Espera_Min", "Tiempo_Servicio_Min", "Consumo_Total_MXN"]
    feature_cols_cat = ["Tipo_Cliente", "Metodo_Pago"]

    X_parts = []
    for col in feature_cols_num:
        if col in df.columns:
            X_parts.append(pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()).values.reshape(-1, 1) if len(df) else np.zeros((0, 1)))
    for col in feature_cols_cat:
        if col in df.columns:
            le = LabelEncoder()
            values = df[col].astype(str).fillna("Nulo")
            X_parts.append(le.fit_transform(values).reshape(-1, 1))

    if not X_parts:
        return {"error": "No hay variables predictoras válidas en el dataset."}

    X = np.hstack(X_parts)

    # Validación: al menos 20 registros y ambas clases presentes
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"error": f"Se necesitan al menos 20 registros con ambas clases (hay {len(X)})."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Matriz de confusión
    fig_cm, ax = _fig(5, 4)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No (0)", "Sí (1)"])
    ax.set_yticklabels(["No (0)", "Sí (1)"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Curva ROC
    fig_roc, ax2 = _fig(6, 5)
    ax2.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "r--", lw=1, label="Aleatorio")
    ax2.set_xlabel("Falsos positivos (FPR)")
    ax2.set_ylabel("Verdaderos positivos (TPR)")
    ax2.set_title("Curva ROC — Regresión Logística")
    ax2.legend()
    ax2.grid(alpha=0.3)

    reporte = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T

    return {
        "fig_cm": fig_cm,
        "fig_roc": fig_roc,
        "metricas": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": roc_auc,
        },
        "reporte": reporte,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
        "error": None,
    }


# ============================================================
# 3. ANOVA DE UN FACTOR Y CHI-CUADRADO (apoyo del reporte)
# ============================================================
def _interpretar_eta2(valor: float) -> str:
    """Tamaño del efecto según los cortes de Cohen para eta² / V de Cramér."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "N/A"
    if valor < 0.01:
        return "Muy pequeño"
    if valor < 0.06:
        return "Pequeño"
    if valor < 0.14:
        return "Mediano"
    return "Grande"


def _interpretar_v_cramer(valor: float, gl: int) -> str:
    """Tamaño del efecto de V de Cramér ajustado por los grados de libertad."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)) or gl <= 0:
        return "N/A"
    if gl == 1:
        cortes = (0.10, 0.30, 0.50)
    elif gl == 2:
        cortes = (0.07, 0.21, 0.35)
    else:
        cortes = (0.06, 0.17, 0.29)
    if valor < cortes[0]:
        return "Muy pequeño"
    if valor < cortes[1]:
        return "Pequeño"
    if valor < cortes[2]:
        return "Mediano"
    return "Grande"


def anova(df: pd.DataFrame, valor_col: str, grupo_col: str) -> dict:
    """
    ANOVA de un factor: contrasta si la media de `valor_col` difiere entre los
    niveles de `grupo_col`.

    Acompaña el contraste con la prueba de Levene (homocedasticidad), el
    tamaño del efecto eta² y la alternativa no paramétrica de Kruskal-Wallis.
    Devuelve un dict de escalares (listo para tabla) o {"error": ...}.
    """
    if valor_col not in df.columns or grupo_col not in df.columns:
        return {"error": f"Columnas {valor_col} o {grupo_col} no encontradas."}

    datos = pd.DataFrame({
        "valor": pd.to_numeric(df[valor_col], errors="coerce"),
        "grupo": df[grupo_col].astype("string"),
    }).dropna()
    if datos.empty:
        return {"error": f"No hay datos válidos de {valor_col} por {grupo_col}."}

    series = [g["valor"].to_numpy()
              for _, g in datos.groupby("grupo", observed=True)
              if len(g) >= 2]
    k = len(series)
    n_total = int(sum(len(s) for s in series))
    if k < 2 or n_total - k < 1:
        return {"error": "Se necesitan al menos dos grupos con 2+ datos cada uno."}

    f_stat, p_val = stats.f_oneway(*series)
    _, lev_p = stats.levene(*series)

    gran_media = float(datos["valor"].mean())
    ss_entre = float(sum(len(s) * (s.mean() - gran_media) ** 2 for s in series))
    ss_total = float(((datos["valor"] - gran_media) ** 2).sum())
    eta2 = ss_entre / ss_total if ss_total > 0 else 0.0

    if p_val < 0.05:
        conclusion = (
            f"Se rechaza H0 (F = {f_stat:.3f}, p = {p_val:.4f} < 0.05): al "
            f"menos un grupo presenta una media distinta de {valor_col}."
        )
    else:
        conclusion = (
            f"No se rechaza H0 (F = {f_stat:.3f}, p = {p_val:.4f} >= 0.05): no "
            f"hay evidencia de medias distintas de {valor_col} entre grupos."
        )
    if lev_p <= 0.05:
        conclusion += (
            " Levene indica varianzas desiguales; conviene revisar la versión "
            "no paramétrica (Kruskal-Wallis)."
        )

    resultado = {
        "prueba": "ANOVA de un factor (F de Fisher)",
        "valor_col": valor_col,
        "grupo_col": grupo_col,
        "grupos": k,
        "n_total": n_total,
        "F": float(f_stat),
        "p_value": float(p_val),
        "gl_entre": k - 1,
        "gl_dentro": n_total - k,
        "eta_cuadrado": float(eta2),
        "tamano_efecto": _interpretar_eta2(eta2),
        "levene_p": float(lev_p),
        "homocedasticidad": ("Sí (p > 0.05)" if lev_p > 0.05
                             else "No (p <= 0.05)"),
        "conclusion": conclusion,
    }

    try:
        h_stat, h_p = stats.kruskal(*series)
    except Exception:
        h_stat, h_p = None, None
    if h_p is not None and not np.isnan(h_p):
        resultado["kruskal_h"] = float(h_stat)
        resultado["kruskal_p"] = float(h_p)

    return resultado


def chi_cuadrado(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Prueba chi-cuadrado de independencia entre dos variables categóricas.

    Reporta el estadístico, el p-valor, la V de Cramér como tamaño del efecto
    y el porcentaje de celdas con frecuencia esperada menor que 5 (supuesto de
    aproximación). Devuelve un dict de escalares o {"error": ...}.
    """
    if var1 not in df.columns or var2 not in df.columns:
        return {"error": f"Columnas {var1} o {var2} no encontradas."}

    tabla = pd.crosstab(
        df[var1].fillna("(Nulo)").astype(str),
        df[var2].fillna("(Nulo)").astype(str),
    )
    tabla = tabla.loc[tabla.sum(axis=1) > 0, tabla.sum(axis=0) > 0]
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return {"error": "Se requieren al menos 2 categorías en cada variable."}

    chi2, p_val, gl, esperadas = stats.chi2_contingency(tabla)
    n_total = int(tabla.to_numpy().sum())
    min_dim = min(tabla.shape) - 1
    v_cramer = (float(np.sqrt(chi2 / (n_total * min_dim)))
                if n_total > 0 and min_dim > 0 else 0.0)
    bajo5 = float((esperadas < 5).mean() * 100)

    if p_val < 0.05:
        conclusion = (
            f"Se rechaza la independencia (chi² = {chi2:.3f}, "
            f"p = {p_val:.4f} < 0.05): {var1} y {var2} están asociadas."
        )
    else:
        conclusion = (
            f"No se rechaza la independencia (chi² = {chi2:.3f}, "
            f"p = {p_val:.4f} >= 0.05): no hay evidencia de asociación entre "
            f"{var1} y {var2}."
        )
    if bajo5 > 20:
        conclusion += (
            " Más del 20 % de las celdas tiene frecuencia esperada menor que 5, "
            "por lo que la aproximación chi² debe interpretarse con cautela."
        )

    return {
        "prueba": "Chi-cuadrado de independencia",
        "variable_1": var1,
        "variable_2": var2,
        "categorias_1": int(tabla.shape[0]),
        "categorias_2": int(tabla.shape[1]),
        "n_total": n_total,
        "chi2": float(chi2),
        "p_value": float(p_val),
        "grados_libertad": int(gl),
        "v_cramer": v_cramer,
        "tamano_efecto": _interpretar_v_cramer(v_cramer, int(gl)),
        "celdas_esperadas_bajo_5_pct": bajo5,
        "supuesto_frecuencias": ("Cumplido" if bajo5 <= 20
                                 else "Revisar (esperadas < 5)"),
        "conclusion": conclusion,
    }


# [FASE 2 - REPORTE] Agregado: anova, chi_cuadrado (y auxiliares de tamaño
# del efecto _interpretar_eta2, _interpretar_v_cramer)