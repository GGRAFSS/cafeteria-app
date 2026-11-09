"""
Módulo 5: Estadística Bayesiana (Unidad 4)
- Calculadora del Teorema de Bayes
- Clasificador Naive Bayes (GaussianNB) para Satisfaccion_Alta
- Visualización de probabilidades posteriores
- Métricas de calibración: Log-loss y Brier Score
- Comparación de modelos con validación cruzada + prueba de Friedman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score,
    confusion_matrix, classification_report,
)


def _fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ============================================================
# 1. CALCULADORA DEL TEOREMA DE BAYES
# ============================================================
def teorema_bayes(p_a: float, p_b_dado_a: float, p_b_dado_no_a: float) -> dict:
    """
    Calcula P(A|B) usando el Teorema de Bayes.

    P(A|B) = P(B|A) · P(A) / [P(B|A)·P(A) + P(B|¬A)·P(¬A)]

    Parámetros:
      p_a:          P(A)         — probabilidad a priori de A
      p_b_dado_a:   P(B|A)       — verosimilitud de B dado A
      p_b_dado_no_a:P(B|¬A)      — verosimilitud de B dado ¬A

    Devuelve un dict con:
      - p_a, p_no_a
      - p_b_dado_a, p_b_dado_no_a
      - p_b: probabilidad total de B (evidencia)
      - p_a_dado_b: probabilidad posterior P(A|B)
      - p_no_a_dado_b: probabilidad posterior P(¬A|B)
    """
    # Validaciones
    p_a = max(0.0, min(1.0, p_a))
    p_no_a = 1.0 - p_a
    p_b_dado_a = max(0.0, min(1.0, p_b_dado_a))
    p_b_dado_no_a = max(0.0, min(1.0, p_b_dado_no_a))

    p_b = p_b_dado_a * p_a + p_b_dado_no_a * p_no_a

    if p_b == 0:
        return {
            "p_a": p_a, "p_no_a": p_no_a,
            "p_b_dado_a": p_b_dado_a, "p_b_dado_no_a": p_b_dado_no_a,
            "p_b": 0.0, "p_a_dado_b": 0.0, "p_no_a_dado_b": 0.0,
        }

    p_a_dado_b = (p_b_dado_a * p_a) / p_b
    p_no_a_dado_b = (p_b_dado_no_a * p_no_a) / p_b

    return {
        "p_a": p_a, "p_no_a": p_no_a,
        "p_b_dado_a": p_b_dado_a, "p_b_dado_no_a": p_b_dado_no_a,
        "p_b": p_b, "p_a_dado_b": p_a_dado_b, "p_no_a_dado_b": p_no_a_dado_b,
    }


def plot_bayes_barras(resultado: dict, etiqueta_a: str = "A", etiqueta_b: str = "B"):
    """Barra comparativa: prior vs posterior de A."""
    fig, ax = _fig(7, 4)
    categorias = [f"P({etiqueta_a}) — Prior", f"P({etiqueta_a}|{etiqueta_b}) — Posterior"]
    valores = [resultado["p_a"], resultado["p_a_dado_b"]]
    colores = ["#4C72B0", "#C44E52"]
    bars = ax.bar(categorias, valores, color=colores, edgecolor="white", width=0.5)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probabilidad")
    ax.set_title(f"Actualización bayesiana: {etiqueta_a} dado {etiqueta_b}")
    ax.grid(alpha=0.3, axis="y")
    return fig


# ============================================================
# 2. NAIVE BAYES PARA Satisfaccion_Alta
# ============================================================
def preparar_dataset_bayes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """
    Prepara X, y para Naive Bayes.
    Binariza Satisfaccion_1_al_5: Alta (4-5) = 1, Baja (1-3) = 0.
    Elimina filas con Satisfaccion nula.
    Codifica variables categóricas con LabelEncoder.
    """
    df = df.copy()
    df = df.dropna(subset=["Satisfaccion_1_al_5"])
    df["Satisfaccion_Alta"] = (df["Satisfaccion_1_al_5"] >= 4).astype(int)
    y = df["Satisfaccion_Alta"].values

    feature_cols_num = ["Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min",
                        "Consumo_Total_MXN"]
    feature_cols_cat = ["Tipo_Cliente", "Metodo_Pago", "Es_Hora_Pico"]

    X_parts = []
    feature_names = []

    for col in feature_cols_num:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()).values
            X_parts.append(vals.reshape(-1, 1))
            feature_names.append(col)

    for col in feature_cols_cat:
        if col in df.columns:
            le = LabelEncoder()
            vals = df[col].astype(str).fillna("Nulo").values
            X_parts.append(le.fit_transform(vals).reshape(-1, 1))
            feature_names.append(col)

    if not X_parts:
        return np.zeros((0, 0)), np.zeros(0), [], ""

    X = np.hstack(X_parts)

    # Verificar que haya al menos 20 registros y 2 clases
    if len(X) < 20 or len(np.unique(y)) < 2:
        return X, y, feature_names, ""

    return X, y, feature_names, "Satisfaccion_Alta"


def entrenar_naive_bayes(df: pd.DataFrame) -> dict:
    """
    Entrena GaussianNB para predecir Satisfaccion_Alta.
    Devuelve dict con figuras, métricas y reporte.
    """
    if "Satisfaccion_1_al_5" not in df.columns:
        return {"error": "Columna Satisfaccion_1_al_5 no encontrada."}

    X, y, feature_names, target_name = preparar_dataset_bayes(df)
    if X.size == 0 or len(y) < 20:
        return {"error": f"Datos insuficientes (n={len(y)}). Se necesitan al menos 20 registros con satisfacción no nula."}

    if len(np.unique(y)) < 2:
        return {"error": "Solo hay una clase de satisfacción en los datos (todas Altas o todas Bajas)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    # Brier Score (binario): usar probabilidad de la clase positiva
    brier = brier_score_loss(y_test, y_proba[:, 1])

    cm = confusion_matrix(y_test, y_pred)

    # Figura matriz de confusión
    fig_cm, ax = _fig(5, 4)
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Baja (0)", "Alta (1)"])
    ax.set_yticklabels(["Baja (0)", "Alta (1)"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión — Naive Bayes")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Figura probabilidades posteriores (promedio por clase real)
    fig_proba, ax2 = _fig(8, 4.5)
    df_pred = pd.DataFrame({
        "Real": y_test,
        "P(Baja)": y_proba[:, 0],
        "P(Alta)": y_proba[:, 1],
    })
    promedios = df_pred.groupby("Real")[["P(Baja)", "P(Alta)"]].mean()
    promedios.index = ["Baja (0)", "Alta (1)"]

    promedios.plot(kind="bar", ax=ax2, color=["#7f7f7f", "#9467bd"],
                   edgecolor="white", width=0.7)
    ax2.set_title("Probabilidades posteriores promedio por clase real")
    ax2.set_ylabel("Probabilidad posterior promedio")
    ax2.set_xlabel("Clase real")
    ax2.set_ylim(0, 1)
    ax2.legend(title="Predicción")
    ax2.grid(alpha=0.3, axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=0)

    reporte = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T

    return {
        "fig_cm": fig_cm,
        "fig_proba": fig_proba,
        "metricas": {
            "accuracy": acc,
            "log_loss": ll,
            "brier_score": brier,
        },
        "reporte": reporte,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "error": None,
    }


# ============================================================
# 3. COMPARACIÓN DE MODELOS + FRIEDMAN
# ============================================================
def comparar_modelos(df: pd.DataFrame) -> dict:
    """
    Compara 3 modelos (Naive Bayes, Regresión Logística, Árbol de Decisión)
    sobre el target Satisfaccion_Alta usando validación cruzada estratificada (5-fold).

    Devuelve:
      - tabla: DataFrame con accuracy media y std por modelo
      - fig: gráfica de barras comparativa
      - friedman: dict con estadístico y p-value
      - error: si no se pudo ejecutar
    """
    if "Satisfaccion_1_al_5" not in df.columns:
        return {"error": "Columna Satisfaccion_1_al_5 no encontrada."}

    X, y, feature_names, _ = preparar_dataset_bayes(df)
    if X.size == 0 or len(y) < 30:
        return {"error": f"Datos insuficientes (n={len(y)}). Se necesitan al menos 30 registros."}
    if len(np.unique(y)) < 2:
        return {"error": "Solo hay una clase en los datos."}

    X_scaled = StandardScaler().fit_transform(X)

    modelos = {
        "Naive Bayes": GaussianNB(),
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for nombre, modelo in modelos.items():
        s = cross_val_score(modelo, X_scaled, y, cv=cv, scoring="accuracy")
        scores[nombre] = s

    tabla = pd.DataFrame({
        "Modelo": list(scores.keys()),
        "Accuracy media": [round(s.mean(), 4) for s in scores.values()],
        "Desv_Std": [round(s.std(), 4) for s in scores.values()],
    }).sort_values("Accuracy media", ascending=False).reset_index(drop=True)

    # Gráfica
    fig, ax = _fig(8, 4.5)
    nombres = list(scores.keys())
    medias = [s.mean() for s in scores.values()]
    stds = [s.std() for s in scores.values()]
    colores = ["#4C72B0", "#C44E52", "#55A868"]
    bars = ax.bar(nombres, medias, yerr=stds, capsize=5,
                  color=colores, edgecolor="white", alpha=0.85)
    for bar, m in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy (validación cruzada 5-fold)")
    ax.set_title("Comparación de modelos de clasificación")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15)

    # Friedman (si al menos 3 modelos)
    friedman_res = {"stat": None, "p": None, "interpretacion": ""}
    if len(scores) >= 3:
        stat, p = friedmanchisquare(*[scores[k] for k in scores])
        friedman_res = {
            "stat": float(stat),
            "p": float(p),
            "interpretacion": (
                "Hay diferencias significativas entre modelos (p < 0.05)"
                if p < 0.05
                else "No hay diferencias significativas entre modelos (p ≥ 0.05)"
            ),
        }

    return {
        "tabla": tabla,
        "fig": fig,
        "friedman": friedman_res,
        "error": None,
    }