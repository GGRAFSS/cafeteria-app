"""
Módulo 4: Pruebas de Hipótesis y Validación de Modelos (Unidad 3)
- Pruebas paramétricas: t de Student para muestras independientes
- Pruebas no paramétricas: Mann-Whitney U
- Validación de supuestos: Shapiro-Wilk (normalidad), Levene (homocedasticidad)
- Modelo de clasificación: regresión logística + matriz de confusión + ROC
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