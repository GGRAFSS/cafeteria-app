"""
Reporte Didáctico — App 2: Modelos Estadísticos (FASE D)
========================================================
Ficha de siete bloques por cada figura o tabla clave del reporte
estadístico, pensada para que cualquier lector reconozca el concepto del
temario que hay detrás de cada resultado.

Diferencia con `mod_reporte_est.py`
-----------------------------------
Este módulo NO modifica el reporte institucional: importa la clase
`ReporteCafeteriaEst`, la especializa (`ReporteDidacticoEst`) y reutiliza su
contexto de datos (`_contexto`) para que las cifras y las figuras sean
exactamente las mismas del reporte completo.

Estructura del documento
------------------------
    Portada institucional (variante didáctica)
    Índice de Fichas Didácticas
    Introducción: cómo leer una ficha + mapa de fichas
    D1 .. D10   una ficha por página (7 bloques cada una)

Contenido
---------
Cada ficha envía a `mod_ia_analista.analizar_seccion_ia(tipo="didactico")`
sus métricas reales, el PNG de la figura (si aplica) y un respaldo curado
bloque por bloque: si Gemini no está disponible, el PDF se genera igual.

API pública
-----------
    generar_reporte_didactico_est_pdf(df=None, min_registros=20)
        -> (bytes | None, str | None)

Uso
---
    from mod_reporte_didactico_est import generar_reporte_didactico_est_pdf
    pdf_bytes, error = generar_reporte_didactico_est_pdf(df)
"""
import io
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

import data_manager as dm
from mod_reporte_est import (
    AUTORES,
    AZUL,
    DOCENTE,
    GRIS,
    INSTITUCION,
    PROGRAMA,
    ReporteCafeteriaEst,
    _contexto,
    _get_asset_path,
)

# ============================================================
# CAMPOS DE LA FICHA (mismo orden que mod_ia_analista.CAMPOS_DIDACTICOS)
# ============================================================
_CAMPOS_FICHA = (
    ("que_se_ve", "Qué se ve"),
    ("tecnica", "Interpretación técnica"),
    ("simple", "En palabras simples"),
    ("temario", "Relación con el temario"),
    ("pedagogia", "Por qué importa pedagógicamente"),
    ("errores", "Errores comunes al interpretar"),
    ("no_aplica", "¿Qué técnicas NO se aplican aquí y por qué?"),
)

try:
    from mod_ia_analista import CAMPOS_DIDACTICOS, analizar_seccion_ia
    HAS_IA = True
except ImportError:  # pragma: no cover - la IA es opcional
    CAMPOS_DIDACTICOS = _CAMPOS_FICHA
    HAS_IA = False

    def analizar_seccion_ia(seccion, datos_metricas, fallback_texto,
                            imagen_png_bytes=None, instruccion_extra=None,
                            doble_capa=True, tipo="analisis",
                            fallback_campos=None):
        """Stub con la misma firma: devuelve el respaldo curado."""
        campos = fallback_campos or {}
        return {clave: str(campos.get(clave) or fallback_texto)
                for clave, _etiqueta in _CAMPOS_FICHA}


# Color de la etiqueta de cada bloque de la ficha didáctica.
_COLOR_CAMPO_DIDACTICO = {
    "que_se_ve": (0, 51, 102),      # azul institucional
    "tecnica": (0, 51, 102),
    "simple": (20, 110, 60),        # verde: lenguaje llano
    "temario": (90, 60, 130),       # violeta: temario
    "pedagogia": (150, 60, 20),     # naranja: pedagogía
    "errores": (170, 30, 30),       # rojo: errores frecuentes
    "no_aplica": (120, 70, 0),      # ámbar oscuro: bloque crítico
}

# Texto de seguridad: si un bloque llegara vacío se dibuja este aviso.
_BLOQUE_SIN_TEXTO = "Contenido no disponible para este bloque."

# Qué responde cada bloque (se usa en la página introductoria).
_DESCRIPCION_CAMPO = {
    "que_se_ve": "Descripción objetiva del objeto: tipo de gráfico o tabla, "
                 "ejes, categorías, escalas y rangos observados.",
    "tecnica": "Lectura formal con las métricas reales del análisis "
               "(promedios, dispersión, proporciones, p-values, tamaños n).",
    "simple": "La misma conclusión sin jerga, con una analogía de la "
              "operación diaria de la cafetería.",
    "temario": "Unidad y concepto del curso al que pertenece la figura o "
               "tabla (U1 descriptiva, U2 distribuciones, U3 hipótesis, "
               "U4 bayesiana).",
    "pedagogia": "Qué aprende el estudiante y qué error conceptual previene "
                 "esta ficha.",
    "errores": "Equivocaciones frecuentes al interpretar el objeto y cómo "
               "evitarlas.",
    "no_aplica": "Técnicas del temario que NO sirven para este objeto y la "
                 "razón técnica concreta.",
}

# Instrucción base: fija el rol y evita que el modelo cambie el formato.
_INSTRUCCION_FICHA = (
    "Actúas como docente de Estadística que explica a estudiantes de "
    "posgrado un objeto concreto del reporte de la cafetería. Describe solo "
    "ese objeto, con los valores reales del payload, sin inventar cifras y "
    "sin repetir el contenido de otras fichas. El bloque 'no_aplica' debe "
    "nombrar técnicas concretas del temario que no sirven para este objeto y "
    "dar la razón técnica (tipo de variable, supuestos incumplidos, objetivo "
    "del análisis o costo computacional)."
)

# 4 workers, mismo balance que _ejecutar_ia_paralelo (velocidad vs cuota).
_FICHA_WORKERS = 4

# Reintento único ante 429 (límite por minuto del plan gratuito): si la
# primera llamada devuelve el respaldo, se espera y se intenta otra vez.
_FICHA_INTENTOS = 2
_FICHA_ESPERA = 8.0


# ============================================================
# HELPERS DE FORMATO Y EXTRACCIÓN DE MÉTRICAS
# ============================================================
def _d(valor, formato="{:.2f}", defecto="N/D") -> str:
    """Formatea un número del contexto; usa `defecto` si falta."""
    try:
        if valor is None:
            return defecto
        return formato.format(float(valor))
    except (TypeError, ValueError):
        return defecto


def _num(valor, decimales=4):
    """Número redondeado para el payload de la IA (`None` si no es válido)."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numero):
        return None
    return round(numero, decimales)


def _es_numero(valor) -> bool:
    return (isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, (bool, np.bool_)))


def _fila_descriptiva(desc, variable: str) -> dict:
    """Fila de `stats_utils.descriptive_stats` convertida a dict simple."""
    if desc is None or getattr(desc, "empty", True) or "Variable" not in desc.columns:
        return {}
    fila = desc[desc["Variable"] == variable]
    if fila.empty:
        return {}
    return {
        str(clave): (_num(valor, 3) if _es_numero(valor) else str(valor))
        for clave, valor in fila.iloc[0].items()
    }


def _fila_por_variable(tabla, variable: str) -> dict:
    """Fila de una tabla indexada por la columna 'Variable' (bondad, outliers)."""
    if tabla is None or getattr(tabla, "empty", True) or "Variable" not in tabla.columns:
        return {}
    fila = tabla[tabla["Variable"] == variable]
    if fila.empty:
        return {}
    return {
        str(clave): (_num(valor, 4) if _es_numero(valor) else str(valor))
        for clave, valor in fila.iloc[0].items()
    }


def _pares_correlacion(corr, limite: int = 3) -> list:
    """Pares con mayor |r| de la matriz de correlación (para la ficha D5)."""
    if corr is None or getattr(corr, "empty", True):
        return []
    columnas = list(corr.columns)
    pares = []
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            try:
                r = float(corr.loc[a, b])
            except (TypeError, ValueError, KeyError):
                continue
            if not math.isfinite(r):
                continue
            pares.append({"par": f"{a} vs {b}", "r": round(r, 4)})
    pares.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pares[:limite]


def _descriptivos_para(ctx: dict, usar_limpios: bool = False):
    """Tabla descriptiva disponible (limpia o cruda) sin comparar DataFrames."""
    orden = (ctx.get("desc_clean"), ctx.get("desc")) if usar_limpios else (
        ctx.get("desc"), ctx.get("desc_clean"))
    for tabla in orden:
        if tabla is not None and not getattr(tabla, "empty", True):
            return tabla
    return None


def _consumo_por_tipo(df) -> dict:
    """Consumo total por tipo de cliente: n, media, mediana y desviación."""
    if df is None or getattr(df, "empty", True) or "Tipo_Cliente" not in df.columns:
        return {}
    base = pd.DataFrame({
        "tipo": df["Tipo_Cliente"].fillna("(Nulo)").astype(str),
        "consumo": pd.to_numeric(df["Consumo_Total_MXN"], errors="coerce"),
    }).dropna()
    if base.empty:
        return {}
    tabla = base.groupby("tipo")["consumo"].agg(
        n="count", media="mean", mediana="median", desv="std")
    return {
        str(tipo): {clave: _num(valor, 2) for clave, valor in fila.items()}
        for tipo, fila in tabla.iterrows()
    }


def _ficha(titulo: str, referencia: str, datos: dict, fallback: dict,
           foco: str, imagen: bytes | None = None,
           pie_imagen: str | None = None) -> dict:
    """Empaqueta una ficha: render, payload real, respaldo curado y enfoque."""
    return {
        "titulo": titulo,
        "referencia": referencia,
        "datos": datos,
        "fallback": fallback,
        "instruccion": f"{_INSTRUCCION_FICHA}\nFoco de esta ficha: {foco}",
        "imagen": imagen,
        "pie_imagen": pie_imagen,
    }


# ============================================================
# ESPECIFICACIONES DE LAS DIEZ FICHAS
# ============================================================
# Cada especificación trae: `titulo` y `referencia` (lo que dibuja el
# render), `datos` (métricas reales que recibe la IA), `fallback` (respaldo
# curado bloque por bloque), `instruccion` (enfoque) e `imagen` (PNG).
def _ficha_d1(ctx: dict) -> dict:
    """D1 — Histograma del tiempo de espera."""
    k = ctx.get("kpis") or {}
    png = ctx.get("png") or {}
    f = _fila_descriptiva(ctx.get("desc"), "Tiempo_Espera_Min")
    media, mediana = f.get("Media"), f.get("Mediana")
    desv, p75, maximo = f.get("Desv_Std"), f.get("P75"), f.get("Max")
    datos = {
        "variable": "Tiempo_Espera_Min",
        "grafico": "histograma de frecuencias",
        "n": f.get("n", k.get("n_total")),
        "estadisticos": f,
        "espera_promedio_min": _num(k.get("espera_prom"), 2),
        "espera_maxima_min": _num(k.get("espera_max"), 2),
        "porcentaje_hora_pico": _num(k.get("pct_pico"), 2),
        "registros_atipicos_iqr": k.get("outliers"),
    }
    fallback = {
        "que_se_ve": (
            f"Histograma de barras contiguas del tiempo de espera en minutos "
            f"para {f.get('n', k.get('n_total', 'N/D'))} registros. El eje "
            f"horizontal mide minutos y el vertical la frecuencia de clientes; "
            f"la masa principal se concentra en los valores bajos y queda una "
            f"cola hacia la derecha que alcanza los {_d(maximo, '{:.1f}')} "
            f"minutos observados."
        ),
        "tecnica": (
            f"La media es {_d(media)} min y la mediana {_d(mediana)} min; al "
            f"ser la media mayor que la mediana, la distribución es asimétrica "
            f"positiva, coherente con un proceso de arribos aleatorio donde "
            f"muchos clientes esperan poco y unos pocos acumulan esperas "
            f"largas. La desviación estándar es {_d(desv)} min y el percentil "
            f"75 llega a {_d(p75)} min, de modo que una de cada cuatro "
            f"personas espera más que ese valor."
        ),
        "simple": (
            f"Casi todos los clientes esperan poco, pero unos cuantos esperan "
            f"muchísimo y por eso el promedio sube a {_d(media)} minutos. Es "
            f"como una fila del café: avanza rápido hasta que alguien pide "
            f"varios pedidos y entonces todos detrás se quedan parados."
        ),
        "temario": (
            "Unidad 1 — Medidas de tendencia central y dispersión: el "
            "histograma es la representación gráfica de la distribución de "
            "una variable cuantitativa y permite comparar media, mediana y "
            "forma de la distribución."
        ),
        "pedagogia": (
            "El estudiante aprende a leer la forma de una distribución antes "
            "de calcular cualquier medida y a reconocer la asimetría positiva. "
            "Previene el error de reportar solo el promedio cuando la cola "
            "derecha describe una parte importante de la experiencia real."
        ),
        "errores": (
            f"Leer la altura de una barra como minutos de espera cuando es "
            f"frecuencia de clientes • Concluir que la espera típica es "
            f"{_d(media)} min sin mirar que la mediana es {_d(mediana)} min • "
            f"Interpretar la cola derecha como error de captura cuando "
            f"corresponde a episodios reales de saturación."
        ),
        "no_aplica": (
            "La prueba t de Student y la comparación de dos grupos no "
            "aplican porque aquí se describe una sola variable sin grupos que "
            "contrastar. El teorema de Bayes tampoco corresponde: no se "
            "actualiza una probabilidad con evidencia. Ajustar una normal es "
            "discutible porque la asimetría positiva viola el supuesto de "
            "simetría."
        ),
    }
    return _ficha(
        "Histograma del tiempo de espera",
        "Figura del capítulo 5 — Estadística Descriptiva (U1)",
        datos, fallback,
        "Descriptiva univariada del tiempo de espera: forma, centro, "
        "dispersión y asimetría.",
        imagen=png.get("hist_espera"),
        pie_imagen="Tiempo de espera observado (minutos)",
    )


def _ficha_d2(ctx: dict) -> dict:
    """D2 — Histograma del consumo total."""
    k = ctx.get("kpis") or {}
    png = ctx.get("png") or {}
    f = _fila_descriptiva(ctx.get("desc"), "Consumo_Total_MXN")
    freq = ctx.get("freq_tipo")
    tipos = ""
    if freq is not None and not getattr(freq, "empty", True):
        tipos = ", ".join(str(t) for t in freq.iloc[:, 0].tolist()[:3])
    datos = {
        "variable": "Consumo_Total_MXN",
        "grafico": "histograma de frecuencias",
        "n": f.get("n", k.get("n_total")),
        "estadisticos": f,
        "consumo_promedio_mxn": _num(k.get("consumo_prom"), 2),
        "consumo_total_mxn": _num(k.get("consumo_total"), 2),
        "tipos_de_cliente": tipos,
        "satisfaccion_promedio": _num(k.get("sat_prom"), 2),
    }
    fallback = {
        "que_se_ve": (
            f"Histograma del consumo total por cliente en pesos mexicanos "
            f"({f.get('n', k.get('n_total', 'N/D'))} registros). El eje "
            f"horizontal muestra el monto gastado y el vertical cuántos "
            f"clientes caen en cada intervalo; el máximo observado es "
            f"{_d(f.get('Max'), '{:.2f}')} MXN."
        ),
        "tecnica": (
            f"El consumo medio es {_d(k.get('consumo_prom'))} MXN con mediana "
            f"{_d(f.get('Mediana'))} MXN y desviación {_d(f.get('Desv_Std'))} "
            f"MXN; la diferencia entre media y mediana indica una cola de "
            f"tickets altos. El total facturado en la muestra es "
            f"{_d(k.get('consumo_total'), '{:,.2f}')} MXN y el ingreso se "
            f"concentra en los intervalos bajos, donde se ubica la mayoría de "
            f"los clientes."
        ),
        "simple": (
            f"Lo normal es que cada persona gaste alrededor de "
            f"{_d(f.get('Mediana'))} pesos, pero unos cuantos compran mucho y "
            f"elevan el promedio a {_d(k.get('consumo_prom'))} pesos. Es la "
            f"diferencia entre el ticket de un café solo y el de alguien que "
            f"pide para todo su equipo."
        ),
        "temario": (
            "Unidad 1 — Distribución de frecuencias y medidas de posición: "
            "el histograma del consumo permite describir la variable "
            "económica del estudio y detectar asimetría en el gasto."
        ),
        "pedagogia": (
            "El estudiante aprende a distinguir entre el gasto típico y el "
            "gasto promedio, y comprende por qué la cola derecha importa para "
            "la facturación. Previene el error de planear con el promedio "
            "cuando la mayoría gasta menos que eso."
        ),
        "errores": (
            "Afirmar que la barra más alta significa que esa gente gasta más, "
            "cuando indica cuántos clientes hay en ese intervalo • Multiplicar "
            "el promedio por el número de clientes sin revisar los nulos de "
            "consumo • Ignorar que cada intervalo agrupa montos distintos y "
            "leer la barra como un valor exacto."
        ),
        "no_aplica": (
            "Una prueba de hipótesis para dos medias no aplica porque no hay "
            "grupos definidos en esta figura. La distribución de Poisson "
            "tampoco: modela conteos de eventos, no montos continuos en "
            "pesos. Ajustar una normal es riesgoso, porque el consumo suele "
            "ser asimétrico y presenta valores extremos legítimos."
        ),
    }
    return _ficha(
        "Histograma del consumo total",
        "Figura del capítulo 5 — Estadística Descriptiva (U1)",
        datos, fallback,
        "Descriptiva de la variable económica: centro, dispersión y cola de "
        "tickets altos.",
        imagen=png.get("hist_consumo"),
        pie_imagen="Consumo total por cliente (MXN)",
    )


def _ficha_d3(ctx: dict) -> dict:
    """D3 — Boxplot del tiempo de servicio."""
    k = ctx.get("kpis") or {}
    png = ctx.get("png") or {}
    f = _fila_descriptiva(_descriptivos_para(ctx, usar_limpios=True),
                          "Tiempo_Servicio_Min")
    out = _fila_por_variable(ctx.get("out_summary"), "Tiempo_Servicio_Min")
    datos = {
        "variable": "Tiempo_Servicio_Min",
        "grafico": "boxplot (diagrama de caja y bigotes)",
        "n": f.get("n", k.get("n_total")),
        "estadisticos": f,
        "cuartiles_iqr": {
            "P25": f.get("P25"), "P50": f.get("P50"), "P75": f.get("P75"),
            "IQR": out.get("IQR"),
            "limite_inferior": out.get("Límite inferior"),
            "limite_superior": out.get("Límite superior"),
        },
        "atipicos_iqr": out.get("N outliers"),
        "servicio_promedio_min": _num(k.get("servicio_prom"), 2),
    }
    fallback = {
        "que_se_ve": (
            f"Boxplot del tiempo de servicio en minutos para {f.get('n', 'N/D')} "
            f"registros. La caja va del primer al tercer cuartil "
            f"({_d(f.get('P25'))} a {_d(f.get('P75'))} min) con la mediana "
            f"marcada en {_d(f.get('P50'))} min; los bigotes cubren el rango "
            f"no atípico y los puntos sueltos señalan servicios "
            f"excepcionalmente largos."
        ),
        "tecnica": (
            f"El IQR es {_d(out.get('IQR'))} min y los límites de Tukey son "
            f"{_d(out.get('Límite inferior'))} y {_d(out.get('Límite superior'))} "
            f"min. Se identifican {out.get('N outliers', 'N/D')} servicios "
            f"atípicos ({_d(out.get('% outliers'))}% de la muestra). La media "
            f"({_d(f.get('Media'))} min) supera a la mediana "
            f"({_d(f.get('P50'))} min), así que los casos lentos estiran la "
            f"distribución hacia la derecha."
        ),
        "simple": (
            f"La mitad de los pedidos se atiende en menos de "
            f"{_d(f.get('P50'))} minutos, pero hay algunos que tardan tanto "
            f"que parecen de otro día. La caja muestra dónde está el grueso "
            f"del trabajo y los puntos aislados son los pedidos que trabaron "
            f"la fila."
        ),
        "temario": (
            "Unidad 1 — Medidas de posición y detección de valores atípicos: "
            "el boxplot resume cuartiles, rango intercuartílico y la regla de "
            "1.5·IQR para señalar observaciones extremas."
        ),
        "pedagogia": (
            "El estudiante aprende a describir una variable con cinco números "
            "y a separar la variación normal del servicio de los casos "
            "excepcionales. Previene el error de eliminar atípicos sin "
            "investigar si corresponden a pedidos grandes reales."
        ),
        "errores": (
            "Borrar los puntos atípicos para 'mejorar' el análisis, cuando "
            "pueden ser la evidencia del problema • Suponer que la caja "
            "contiene al 50% central y que los bigotes son los extremos "
            "absolutos • Comparar cajas de tamaños de muestra muy distintos "
            "sin mencionarlo."
        ),
        "no_aplica": (
            "La prueba t no aplica porque el boxplot describe un solo grupo; "
            "para comparar medianas haría falta un contraste entre grupos. "
            "Tampoco aplica ajustar una distribución teórica: el boxplot no "
            "estima parámetros, solo posiciones. La regresión logística queda "
            "fuera porque no hay variable respuesta binaria."
        ),
    }
    return _ficha(
        "Boxplot del tiempo de servicio",
        "Figura del capítulo 5 — Estadística Descriptiva (U1)",
        datos, fallback,
        "Describir el servicio con cuartiles, IQR y atípicos; distinguir "
        "variación normal de casos extremos.",
        imagen=png.get("box_servicio"),
        pie_imagen="Tiempo de servicio por pedido (minutos)",
    )


def _ficha_d4(ctx: dict) -> dict:
    """D4 — Boxplot del consumo por tipo de cliente."""
    k = ctx.get("kpis") or {}
    png = ctx.get("png") or {}
    por_tipo = _consumo_por_tipo(ctx.get("df")) or _consumo_por_tipo(ctx.get("clean"))
    ordenados = sorted(por_tipo.items(), key=lambda par: par[1].get("media") or 0)
    if ordenados:
        bajo, alto = ordenados[0], ordenados[-1]
        resumen = (
            f"{bajo[0]} gasta en promedio {_d(bajo[1].get('media'))} MXN y "
            f"{alto[0]} {_d(alto[1].get('media'))} MXN"
        )
    else:
        bajo, alto = ("N/D", {}), ("N/D", {})
        resumen = "no hay datos suficientes por tipo de cliente"
    datos = {
        "variable": "Consumo_Total_MXN",
        "grupo": "Tipo_Cliente",
        "grafico": "boxplot comparativo por categoría",
        "consumo_por_tipo": por_tipo,
        "resumen_comparativo": resumen,
        "consumo_promedio_global_mxn": _num(k.get("consumo_prom"), 2),
        "n_total": k.get("n_total"),
    }
    fallback = {
        "que_se_ve": (
            "Boxplots lado a lado del consumo total (MXN) para cada tipo de "
            f"cliente registrado ({', '.join(por_tipo) if por_tipo else 'sin categorías'}). "
            "Cada caja muestra el 50% central de los tickets de ese grupo, la "
            "línea interior la mediana y los puntos los consumos atípicos; "
            "todos los grupos comparten la misma escala vertical para poder "
            "compararse."
        ),
        "tecnica": (
            f"Comparando las medianas y las cajas se observa que {resumen}. "
            f"El promedio global es {_d(k.get('consumo_prom'))} MXN; las cajas "
            f"se solapan en buena parte de su rango, de modo que la diferencia "
            f"entre grupos existe pero no separa por completo a los clientes. "
            f"La dispersión dentro de cada grupo es del mismo orden que la "
            f"diferencia entre grupos."
        ),
        "simple": (
            f"Todos los clientes compran más o menos lo mismo, aunque "
            f"{alto[0]} deja un poco más de dinero que {bajo[0]}. Hay "
            f"estudiantes que gastan como un profesor y profesores que solo "
            f"piden un café, así que la caja de cada grupo se parece "
            f"bastante."
        ),
        "temario": (
            "Unidad 1 — Estadística descriptiva bivariada: comparar una "
            "variable cuantitativa entre categorías mediante gráficos de caja "
            "es el paso previo a cualquier prueba de comparación de medias."
        ),
        "pedagogia": (
            "El estudiante aprende a comparar grupos mirando centro, "
            "dispersión y solapamiento, no solo promedios. Previene el error "
            "de afirmar que un grupo gasta más porque su promedio es mayor, "
            "sin considerar la variabilidad interna."
        ),
        "errores": (
            "Concluir que la diferencia es significativa sin haber hecho una "
            "prueba de hipótesis • Comparar medianas ignorando el tamaño de "
            "cada grupo • Leer la altura de los puntos atípicos como errores "
            "de captura cuando suelen ser tickets familiares reales."
        ),
        "no_aplica": (
            "La correlación de Pearson no aplica: el eje de grupos es "
            "categórico nominal, no una escala numérica. Poisson tampoco, "
            "porque el consumo es continuo en pesos. Un ajuste normal por "
            "grupo requeriría muestras mayores y verificar simetría; con "
            "grupos pequeños el boxplot es más informativo."
        ),
    }
    return _ficha(
        "Boxplot del consumo por tipo de cliente",
        "Figura del capítulo 5 — Estadística Descriptiva (U1)",
        datos, fallback,
        "Comparación descriptiva entre categorías: centro, dispersión y "
        "solapamiento de los grupos.",
        imagen=png.get("box_consumo_tipo"),
        pie_imagen="Consumo total por tipo de cliente (MXN)",
    )


def _ficha_d5(ctx: dict) -> dict:
    """D5 — Matriz de correlación de Pearson."""
    png = ctx.get("png") or {}
    corr = ctx.get("corr")
    variables = list(corr.columns) if corr is not None and not getattr(corr, "empty", True) else []
    pares = _pares_correlacion(corr, 3)
    if pares:
        principal = pares[0]
        fuerte = f"{principal['par']} (r = {principal['r']})"
    else:
        principal, fuerte = {}, "no se pudo calcular la matriz"
    datos = {
        "variable": "matriz de correlación de Pearson",
        "variables_incluidas": variables,
        "n_variables": len(variables),
        "pares_mas_fuertes": pares,
        "par_principal": fuerte,
        "n_registros": (ctx.get("kpis") or {}).get("n_total"),
        "nota": "Los valores van de -1 a 1; el signo indica la dirección.",
    }
    fallback = {
        "que_se_ve": (
            "Matriz cuadrada de correlaciones de Pearson entre las variables "
            f"numéricas del estudio ({', '.join(variables) if variables else 'sin variables'}). "
            "La diagonal vale 1 por definición, los colores van de azul "
            "(correlación negativa) a rojo (positiva) y cada celda contiene el "
            "coeficiente r con dos decimales."
        ),
        "tecnica": (
            f"El par con mayor asociación lineal es {fuerte}. El resto de los "
            f"coeficientes se mantiene cerca de cero, de modo que las "
            f"variables describen dimensiones distintas del servicio "
            f"(tiempo, dinero y satisfacción). La matriz se estima con los "
            f"pares completos, por lo que los nulos de una variable no "
            f"eliminan las correlaciones de los demás."
        ),
        "simple": (
            f"La matriz dice qué cosas se mueven juntas: {fuerte}. Las demás "
            f"casillas casi no se relacionan, así que subir el consumo no "
            f"explica por sí solo la espera. Es como comparar el clima con "
            f"las ventas: coinciden a veces, pero no son lo mismo."
        ),
        "temario": (
            "Unidad 1 — Correlación lineal y análisis bivariado: la matriz de "
            "Pearson cuantifica la fuerza y la dirección de la asociación "
            "entre pares de variables cuantitativas."
        ),
        "pedagogia": (
            "El estudiante aprende a leer una matriz de correlación, a "
            "distinguir fuerza de dirección y a no confundir asociación con "
            "causa. Previene el error clásico de proponer políticas basadas "
            "en un coeficiente sin diseño experimental."
        ),
        "errores": (
            "Interpretar una correlación alta como relación causal • Comparar "
            "coeficientes calculados con distinto número de pares válidos • "
            "Ignorar que Pearson solo detecta asociación lineal y no capta "
            "relaciones curvas."
        ),
        "no_aplica": (
            "La prueba t de Student y el ANOVA no aplican: no se comparan "
            "medias entre grupos, se mide asociación entre variables "
            "continuas. El ajuste exponencial de tiempos tampoco procede "
            "porque aquí no se modela el tiempo hasta un evento, y Poisson no "
            "describe montos."
        ),
    }
    return _ficha(
        "Matriz de correlación",
        "Figura del capítulo 5 — Estadística Descriptiva (U1)",
        datos, fallback,
        "Asociación lineal entre variables numéricas: fuerza, dirección y "
        "lectura crítica (correlación no implica causalidad).",
        imagen=png.get("corr_png"),
        pie_imagen="Correlaciones de Pearson entre variables numéricas",
    )


def _ficha_d6(ctx: dict) -> dict:
    """D6 — Ajuste exponencial del tiempo de espera."""
    png = ctx.get("png") or {}
    f = _fila_descriptiva(ctx.get("desc"), "Tiempo_Espera_Min")
    bondad = _fila_por_variable(ctx.get("bondad"), "Tiempo_Espera_Min")
    media = f.get("Media")
    lam = (1.0 / float(media)) if isinstance(media, (int, float)) and media else None
    p_valor = bondad.get("p-value")
    acepta = bondad.get("¿Se acepta? (α=0.05)", "N/D")
    datos = {
        "variable": "Tiempo_Espera_Min",
        "distribucion_teorica": "Exponencial",
        "parametro_lambda": _num(lam, 4),
        "media_muestral_min": _num(media, 3),
        "bondad_ajuste": bondad,
        "ks": bondad.get("KS"),
        "p_value": p_valor,
        "se_acepta_alpha_005": acepta,
        "n": f.get("n"),
    }
    fallback = {
        "que_se_ve": (
            "Gráfico compuesto de dos paneles: a la izquierda el histograma "
            f"del tiempo de espera con la curva exponencial teórica "
            f"(λ = {_d(lam, '{:.4f}')} por minuto) superpuesta; a la derecha el "
            "Q-Q plot que compara los cuantiles observados contra los "
            "teóricos. Los puntos del Q-Q se alinean sobre la recta cuando el "
            "ajuste es razonable."
        ),
        "tecnica": (
            f"La prueba de Kolmogorov-Smirnov reporta D = {_d(bondad.get('KS'), '{:.4f}')} "
            f"con p = {_d(p_valor, '{:.4f}')}, por lo que el ajuste exponencial "
            f"{'se acepta' if acepta == 'Sí' else 'se rechaza'} al nivel "
            f"α = 0.05. El parámetro estimado es λ = {_d(lam, '{:.4f}')}, "
            f"equivalente a una espera media de {_d(media)} min. La desviación "
            f"del Q-Q en la cola derecha indica que los episodios de "
            f"saturación no los captura del todo el modelo."
        ),
        "simple": (
            f"El modelo exponencial dice que la mayoría de las esperas son "
            f"cortas y que las largas son raras pero posibles, con un promedio "
            f"de {_d(media)} minutos. Es como esperar el camión: casi siempre "
            f"pasa pronto, pero de vez en cuando se junta la mala suerte y "
            f"esperas el doble."
        ),
        "temario": (
            "Unidad 2 — Distribución exponencial y prueba de bondad de "
            "ajuste: se estima λ con la media muestral y se contrasta el "
            "ajuste con Kolmogorov-Smirnov."
        ),
        "pedagogia": (
            "El estudiante aprende a estimar un parámetro por el método de "
            "los momentos, a plantear H0 de bondad de ajuste y a leer un Q-Q "
            "plot. Previene el error de declarar un modelo válido solo porque "
            "la curva se ve parecida."
        ),
        "errores": (
            "Aceptar el modelo mirando solo el histograma y no el p-value • "
            "Confundir λ de la exponencial (1/media) con λ de Poisson "
            "(llegadas por minuto) • Ignorar que una muestra grande vuelve "
            "significativa cualquier desviación pequeña."
        ),
        "no_aplica": (
            "La prueba t de Student no aplica: no se comparan medias de "
            "grupos. La regresión logística tampoco, porque no hay respuesta "
            "binaria. Una normal no es adecuada para tiempos de espera, que "
            "son positivos y asimétricos por construcción."
        ),
    }
    return _ficha(
        "Ajuste exponencial del tiempo de espera",
        "Figura del capítulo 6 — Distribuciones de Probabilidad (U2)",
        datos, fallback,
        "Estimar λ por momentos y evaluar la bondad de ajuste exponencial "
        "con KS y Q-Q.",
        imagen=png.get("ajuste_espera"),
        pie_imagen="Ajuste exponencial y Q-Q del tiempo de espera",
    )


def _ficha_d7(ctx: dict) -> dict:
    """D7 — Ajuste normal del tiempo de servicio."""
    png = ctx.get("png") or {}
    f = _fila_descriptiva(_descriptivos_para(ctx, usar_limpios=True),
                          "Tiempo_Servicio_Min")
    bondad = _fila_por_variable(ctx.get("bondad"), "Tiempo_Servicio_Min")
    mu, sigma = f.get("Media"), f.get("Desv_Std")
    acepta = bondad.get("¿Se acepta? (α=0.05)", "N/D")
    datos = {
        "variable": "Tiempo_Servicio_Min",
        "distribucion_teorica": "Normal",
        "parametros": {"mu": _num(mu, 3), "sigma": _num(sigma, 3)},
        "bondad_ajuste": bondad,
        "ks": bondad.get("KS"),
        "p_value": bondad.get("p-value"),
        "se_acepta_alpha_005": acepta,
        "n": f.get("n"),
    }
    fallback = {
        "que_se_ve": (
            "Dos paneles: el histograma del tiempo de servicio con la curva "
            f"normal ajustada (μ = {_d(mu)} min, σ = {_d(sigma)} min) y el Q-Q "
            "plot de los cuantiles observados frente a los normales teóricos, "
            "con la recta de referencia. La simetría de la campana contrasta "
            "con el histograma de la espera."
        ),
        "tecnica": (
            f"El ajuste normal se evalúa con Kolmogorov-Smirnov: "
            f"D = {_d(bondad.get('KS'), '{:.4f}')} y p = "
            f"{_d(bondad.get('p-value'), '{:.4f}')}, de modo que el supuesto "
            f"{'no se rechaza' if acepta == 'Sí' else 'se rechaza'} con "
            f"α = 0.05. Los parámetros estimados son μ = {_d(mu)} min y "
            f"σ = {_d(sigma)} min; bajo normalidad, el 95% de los servicios "
            f"caería entre {_d((mu or 0) - 1.96 * (sigma or 0))} y "
            f"{_d((mu or 0) + 1.96 * (sigma or 0))} min."
        ),
        "simple": (
            f"El tiempo de atención se comporta como una campana: casi todos "
            f"los pedidos tardan alrededor de {_d(mu)} minutos, muy pocos "
            f"tardan mucho menos y muy pocos mucho más. Eso permite prometer "
            f"un tiempo de servicio razonable sin exagerar."
        ),
        "temario": (
            "Unidad 2 — Distribución normal, estandarización y bondad de "
            "ajuste: se estiman μ y σ y se verifica el supuesto con KS y Q-Q, "
            "paso obligatorio antes de usar pruebas paramétricas."
        ),
        "pedagogia": (
            "El estudiante aprende que la normalidad es un supuesto que se "
            "verifica, no una suposición automática, y usa la regla empírica "
            "68-95-99.7 para interpretar σ. Previene aplicar pruebas "
            "paramétricas sin revisar el ajuste."
        ),
        "errores": (
            "Asumir normalidad porque la muestra es grande, aunque la forma "
            "sea asimétrica • Confundir el p-value de KS con la probabilidad "
            "de que el modelo sea verdadero • Usar σ muestral sin indicar si "
            "es poblacional o insesgada."
        ),
        "no_aplica": (
            "La distribución de Poisson no aplica: el tiempo de servicio es "
            "continuo y no un conteo de eventos. La exponencial tampoco es "
            "adecuada, porque describe tiempos hasta un evento con media igual "
            "a la desviación, cosa que aquí no se cumple. Bayes no interviene: "
            "no hay actualización de creencias."
        ),
    }
    return _ficha(
        "Ajuste normal del tiempo de servicio",
        "Figura del capítulo 6 — Distribuciones de Probabilidad (U2)",
        datos, fallback,
        "Verificar el supuesto de normalidad del tiempo de servicio y "
        "traducir μ y σ a tiempos prometidos.",
        imagen=png.get("ajuste_servicio"),
        pie_imagen="Ajuste normal y Q-Q del tiempo de servicio",
    )


def _ficha_d8(ctx: dict) -> dict:
    """D8 — Simulación de Poisson de llegadas por minuto."""
    png = ctx.get("png") or {}
    poisson = ctx.get("poisson") or {}
    lam = poisson.get("lambda")
    p_valor = poisson.get("p_value")
    p_ok = isinstance(p_valor, (int, float)) and p_valor is not None
    datos = {
        "modelo": "Poisson",
        "lambda_llegadas_por_minuto": _num(lam, 4),
        "media_empirica": _num(poisson.get("media_empirica"), 4),
        "media_simulada": _num(poisson.get("media_simulada"), 4),
        "ks_estadistico": _num(poisson.get("ks_stat"), 4),
        "ks_p_value": _num(p_valor, 4),
        "minutos_observados": poisson.get("n_minutos"),
        "total_llegadas": poisson.get("total_llegadas"),
        "llegadas_por_hora_estimadas": _num((lam or 0) * 60, 2),
        "interpretacion_ks": (
            "No se rechaza Poisson (p > 0.05)" if p_ok and p_valor > 0.05
            else "Se rechaza Poisson (p <= 0.05)" if p_ok
            else "No disponible"
        ),
    }
    fallback = {
        "que_se_ve": (
            "Dos gráficos: un histograma comparativo del número de llegadas "
            "por minuto —barras empíricas frente a las simuladas con "
            f"Poisson(λ = {_d(lam, '{:.3f}')})— y un Q-Q plot Poisson. El eje "
            "horizontal cuenta llegadas en un minuto y el vertical la "
            "densidad; se observan valores bajos la mayor parte del tiempo y "
            "pocos minutos con varias llegadas."
        ),
        "tecnica": (
            f"λ estimado es {_d(lam, '{:.4f}')} llegadas por minuto "
            f"({_d((lam or 0) * 60, '{:.1f}')} por hora) sobre "
            f"{poisson.get('n_minutos', 'N/D')} minutos y "
            f"{poisson.get('total_llegadas', 'N/D')} llegadas totales. La "
            f"media empírica ({_d(poisson.get('media_empirica'), '{:.4f}')}) y "
            f"la simulada ({_d(poisson.get('media_simulada'), '{:.4f}')}) "
            f"coinciden. Kolmogorov-Smirnov da D = "
            f"{_d(poisson.get('ks_stat'), '{:.4f}')} con p = "
            f"{_d(p_valor, '{:.4f}')}, así que {datos['interpretacion_ks']}."
        ),
        "simple": (
            f"En un minuto normal llega {_d(lam, '{:.2f}')} persona, pero "
            f"nunca es igual: hay minutos vacíos y minutos con cuatro o cinco "
            f"clientes de golpe. Poisson describe justo eso, la suerte de "
            f"llegadas al azar, y permite dimensionar cuánta caja hace falta "
            f"en el peor minuto."
        ),
        "temario": (
            "Unidad 2 — Proceso de Poisson: modela conteos de eventos en un "
            "intervalo fijo, con λ como tasa media de ocurrencia, y se valida "
            "con la distribución empírica de llegadas por minuto."
        ),
        "pedagogia": (
            "El estudiante aprende a estimar λ desde datos reales, a simular "
            "el proceso y a comparar simulación contra realidad con KS. "
            "Previene el error de usar el promedio de llegadas como si fuera "
            "un valor fijo."
        ),
        "errores": (
            "Suponer llegadas perfectamente uniformes (λ constante), cuando "
            "existe hora pico • Confundir λ de Poisson con λ de la "
            "exponencial • Ignorar que los minutos del periodo deben ser "
            "disjuntos para que el conteo tenga sentido."
        ),
        "no_aplica": (
            "La distribución normal no aplica: el conteo por minuto es "
            "discreto y muy asimétrico con λ pequeño. La prueba t tampoco, "
            "porque no hay dos grupos que comparar. El ajuste exponencial "
            "modelaría el tiempo entre llegadas, no el número de llegadas."
        ),
    }
    return _ficha(
        "Simulación Poisson de llegadas por minuto",
        "Figura del capítulo 6 — Distribuciones de Probabilidad (U2)",
        datos, fallback,
        "Estimar λ de llegadas, simular el proceso y contrastarlo con la "
        "realidad mediante KS.",
        imagen=png.get("poisson_hist"),
        pie_imagen="Llegadas por minuto: empírico vs Poisson simulada",
    )


def _ficha_d9(ctx: dict) -> dict:
    """D9 — Prueba t de Student del consumo por tipo de cliente."""
    png = ctx.get("png") or {}
    hip = ctx.get("hip") or {}
    grupos = ctx.get("hip_grupos") or []
    g_a = hip.get("grupo_a") or (grupos[0] if grupos else "Grupo A")
    g_b = hip.get("grupo_b") or (grupos[1] if len(grupos) > 1 else "Grupo B")
    p_t = hip.get("ttest_p")
    p_u = hip.get("mannwhitney_p")
    t_ok = isinstance(p_t, (int, float))
    datos = {
        "variable_respuesta": hip.get("valor_col", "Consumo_Total_MXN"),
        "grupo": hip.get("grupo_col", "Tipo_Cliente"),
        "grupo_a": g_a,
        "grupo_b": g_b,
        "n_a": hip.get("n_a"),
        "n_b": hip.get("n_b"),
        "media_a": _num(hip.get("media_a"), 2),
        "media_b": _num(hip.get("media_b"), 2),
        "shapiro_p_a": _num(hip.get("shapiro_a_p"), 4),
        "shapiro_p_b": _num(hip.get("shapiro_b_p"), 4),
        "levene_p": _num(hip.get("levene_p"), 4),
        "t_estadistico": _num(hip.get("ttest_stat"), 4),
        "t_p_value": _num(p_t, 4),
        "welch_o_student": (
            "Welch (varianzas desiguales)" if hip.get("ttest_equal_var") is False
            else "Student (varianzas iguales)" if t_ok else "N/D"
        ),
        "mannwhitney_u": _num(hip.get("mannwhitney_u"), 2),
        "mannwhitney_p": _num(p_u, 4),
        "cohen_d": _num(hip.get("cohen_d"), 4),
        "interpretacion_d": hip.get("interpretacion_d", "N/D"),
        "decision_alpha_005": (
            "Se rechaza H0 de igualdad de medias" if t_ok and p_t < 0.05
            else "No se rechaza H0 de igualdad de medias" if t_ok else "N/D"
        ),
        "error": hip.get("error"),
    }
    fallback = {
        "que_se_ve": (
            f"Boxplots comparativos del consumo total (MXN) entre {g_a} y "
            f"{g_b}, con las medias marcadas y los bigotes de variabilidad. "
            f"El grupo {g_a} aporta {hip.get('n_a', 'N/D')} observaciones y el "
            f"grupo {g_b} {hip.get('n_b', 'N/D')}; ambos comparten la escala "
            f"para que la comparación sea legítima."
        ),
        "tecnica": (
            f"Medias: {g_a} = {_d(hip.get('media_a'))} MXN y {g_b} = "
            f"{_d(hip.get('media_b'))} MXN. Shapiro-Wilk por grupo da p = "
            f"{_d(hip.get('shapiro_a_p'), '{:.4f}')} y "
            f"{_d(hip.get('shapiro_b_p'), '{:.4f}')}; Levene p = "
            f"{_d(hip.get('levene_p'), '{:.4f}')}, por eso se aplica "
            f"t de {datos['welch_o_student']}: t = "
            f"{_d(hip.get('ttest_stat'), '{:.4f}')} con p = {_d(p_t, '{:.4f}')}. "
            f"La alternativa no paramétrica (Mann-Whitney) da U = "
            f"{_d(hip.get('mannwhitney_u'), '{:.1f}')} y p = {_d(p_u, '{:.4f}')}, "
            f"con Cohen d = {_d(hip.get('cohen_d'), '{:.3f}')} "
            f"({hip.get('interpretacion_d', 'N/D')})."
        ),
        "simple": (
            f"En promedio {g_a} gasta {_d(hip.get('media_a'))} pesos y {g_b} "
            f"gasta {_d(hip.get('media_b'))} pesos. La prueba estadística "
            f"revisa si esa diferencia es real o solo casualidad de la "
            f"muestra; con un tamaño de efecto "
            f"{hip.get('interpretacion_d', 'N/D')}, la diferencia práctica es "
            f"pequeña."
        ),
        "temario": (
            "Unidad 3 — Pruebas de hipótesis para dos muestras: supuestos de "
            "normalidad y homocedasticidad, prueba t de Student o Welch, "
            "alternativa de Mann-Whitney y tamaño del efecto con Cohen d."
        ),
        "pedagogia": (
            "El estudiante aprende la secuencia completa: verificar supuestos, "
            "elegir la variante de la prueba, reportar el p-value y traducirlo "
            "a tamaño del efecto. Previene concluir con solo comparar "
            "promedios."
        ),
        "errores": (
            "Elegir t de Student sin revisar Levene y la normalidad • "
            "Interpretar p > 0.05 como prueba de que los grupos son iguales • "
            "Reportar significancia estadística sin tamaño del efecto ni "
            "intervalos de confianza."
        ),
        "no_aplica": (
            "Una prueba de bondad de ajuste no responde esta pregunta: no se "
            "evalúa la forma de una distribución sino la diferencia de "
            "medias. Poisson no aplica al consumo en pesos. La regresión "
            "logística sirve para predecir una clase, no para contrastar "
            "medias de un monto."
        ),
    }
    return _ficha(
        "Prueba t de Student del consumo por tipo de cliente",
        "Figura del capítulo 7 — Pruebas de Hipótesis (U3)",
        datos, fallback,
        "Contraste de medias entre dos grupos con verificación de supuestos y "
        "tamaño del efecto.",
        imagen=png.get("hip_box"),
        pie_imagen=f"Consumo por tipo de cliente: {g_a} vs {g_b}",
    )


def _ficha_d10(ctx: dict) -> dict:
    """D10 — Calculadora de Bayes (prior vs posterior)."""
    png = ctx.get("png") or {}
    bayes = ctx.get("bayes") or {}
    res = bayes.get("resultado") or {}
    etiqueta_a = bayes.get("etiqueta_a", "A")
    etiqueta_b = bayes.get("etiqueta_b", "B")
    prior = res.get("p_a", bayes.get("p_a_empirico"))
    posterior = res.get("p_a_dado_b")
    datos = {
        "evento_a": etiqueta_a,
        "evento_b_evidencia": etiqueta_b,
        "prior": _num(prior, 4),
        "verosimilitud_b_dado_a": _num(res.get("p_b_dado_a"), 4),
        "verosimilitud_b_dado_no_a": _num(res.get("p_b_dado_no_a"), 4),
        "evidencia_total_p_b": _num(res.get("p_b"), 4),
        "posterior": _num(posterior, 4),
        "posterior_complementario": _num(res.get("p_no_a_dado_b"), 4),
        "prior_empirico_sin_acotar": _num(bayes.get("p_a_empirico"), 4),
        "cambio_porcentual": _num(
            ((posterior - prior) * 100) if isinstance(posterior, (int, float))
            and isinstance(prior, (int, float)) else None, 2),
        "n_registros": (ctx.get("kpis") or {}).get("n_total"),
    }
    fallback = {
        "que_se_ve": (
            f"Gráfico de barras que compara la probabilidad a priori de "
            f"'{etiqueta_a}' con la posterior después de observar la evidencia "
            f"'{etiqueta_b}'. La primera barra es la creencia inicial "
            f"({_d(prior, '{:.3f}', '0.000')}) y la segunda la probabilidad "
            f"actualizada ({_d(posterior, '{:.3f}', '0.000')})."
        ),
        "tecnica": (
            f"Con P(A) = {_d(prior, '{:.4f}')}, P(B|A) = "
            f"{_d(res.get('p_b_dado_a'), '{:.4f}')} y P(B|¬A) = "
            f"{_d(res.get('p_b_dado_no_a'), '{:.4f}')}, la evidencia total es "
            f"P(B) = {_d(res.get('p_b'), '{:.4f}')} y el posterior resulta "
            f"P(A|B) = {_d(posterior, '{:.4f}')}. La evidencia "
            f"{'aumenta' if isinstance(posterior, (int, float)) and isinstance(prior, (int, float)) and posterior > prior else 'reduce'} "
            f"la probabilidad de '{etiqueta_a}' en "
            f"{_d(abs((posterior - prior) * 100) if isinstance(posterior, (int, float)) and isinstance(prior, (int, float)) else None, '{:.1f}')} "
            f"puntos porcentuales."
        ),
        "simple": (
            f"Antes de ver nada sabíamos que '{etiqueta_a}' tenía una "
            f"probabilidad de {_d(prior, '{:.1%}')}. Al ver que el cliente "
            f"'{etiqueta_b}', esa probabilidad pasa a "
            f"{_d(posterior, '{:.1%}')}: la evidencia mueve nuestra creencia, "
            f"no la reemplaza por completo."
        ),
        "temario": (
            "Unidad 4 — Teorema de Bayes: actualización de probabilidades con "
            "la regla de la probabilidad total en el denominador y la "
            "verosimilitud en el numerador."
        ),
        "pedagogia": (
            "El estudiante aprende a separar prior, verosimilitud y evidencia, "
            "y a comprobar que el posterior es coherente con la fórmula. "
            "Previene confundir P(A|B) con P(B|A), el error base del "
            "razonamiento probabilístico."
        ),
        "errores": (
            "Invertir P(A|B) y P(B|A), la falacia de la condicional "
            "traspuesta • Ignorar la tasa base y quedarse solo con la "
            "verosimilitud • Creer que un posterior alto implica causalidad "
            "entre la evidencia y el evento."
        ),
        "no_aplica": (
            "Una prueba t no aplica: no se comparan medias, se actualiza una "
            "probabilidad. La bondad de ajuste de Poisson tampoco, porque no "
            "se modelan conteos. Una regresión lineal no corresponde a una "
            f"variable respuesta binaria como '{etiqueta_a}'."
        ),
    }
    return _ficha(
        "Calculadora de Bayes: prior frente a posterior",
        "Figura del capítulo 8 — Estadística Bayesiana (U4)",
        datos, fallback,
        "Actualización prior→posterior con verosimilitudes empíricas y "
        "lectura crítica de la condicional traspuesta.",
        imagen=png.get("bayes_barras"),
        pie_imagen=f"P({etiqueta_a}) antes y después de observar '{etiqueta_b}'",
    )


def _especificaciones(ctx: dict) -> list:
    """Arma las diez fichas (D1-D10) con payload, respaldo e imagen."""
    return [
        _ficha_d1(ctx), _ficha_d2(ctx), _ficha_d3(ctx), _ficha_d4(ctx),
        _ficha_d5(ctx), _ficha_d6(ctx), _ficha_d7(ctx), _ficha_d8(ctx),
        _ficha_d9(ctx), _ficha_d10(ctx),
    ]


# ============================================================
# CONTENIDO CON IA (paralelo) Y RESPALDO POR BLOQUE
# ============================================================
def _ia_disponible() -> bool:
    """True si hay librería y API key: sin ellos el reintento no sirve."""
    try:
        import mod_ia_analista as _mod
        if not getattr(_mod, "HAS_GENAI", False):
            return False
        return bool(_mod._get_api_key())
    except Exception:
        return False


def _campos_de_ficha(spec: dict) -> tuple:
    """Genera los siete bloques de una ficha y reporta si usó la IA.

    Devuelve `(campos, fuente)`. Si la IA no está disponible, falla o
    responde un JSON incompleto, `analizar_seccion_ia` ya devuelve el respaldo
    curado bloque por bloque, de modo que esta función nunca lanza. Cuando la
    respuesta equivale al respaldo se reintenta una vez (8 s): así un 429
    transitorio del plan gratuito no deja la ficha sin texto de IA.
    """
    fallback = dict(spec["fallback"])
    respuesta = None

    for intento in range(_FICHA_INTENTOS):
        try:
            respuesta = analizar_seccion_ia(
                f"Ficha didáctica · {spec['titulo']}",
                spec["datos"],
                fallback.get("tecnica", ""),
                imagen_png_bytes=spec.get("imagen"),
                instruccion_extra=spec["instruccion"],
                doble_capa=False,   # el prompt didáctico ya pide el bloque simple
                tipo="didactico",
                fallback_campos=fallback,
            )
        except Exception:
            respuesta = None

        if isinstance(respuesta, dict) and respuesta and respuesta != fallback:
            break
        if intento + 1 < _FICHA_INTENTOS and _ia_disponible():
            time.sleep(_FICHA_ESPERA)

    if isinstance(respuesta, dict) and respuesta:
        completo = {
            clave: str(respuesta.get(clave) or fallback.get(clave)
                       or _BLOQUE_SIN_TEXTO)
            for clave, _etiqueta in _CAMPOS_FICHA
        }
        return completo, ("ia" if completo != fallback else "respaldo")

    return fallback, "respaldo"


def _fichas_didacticas(ctx: dict) -> list:
    """Construye las diez fichas del reporte didáctico con IA en paralelo.

    Cada ficha envía a Gemini su propio payload acotado (las métricas reales
    del objeto que explica), su PNG y su respaldo curado, así que un fallo
    puntual no contamina a las demás. Con `_FICHA_WORKERS = 4` las diez
    llamadas tardan lo que unas tres o cuatro, y la caché de
    `mod_ia_analista` evita repetir cuota al regenerar con los mismos datos.
    """
    especificaciones = _especificaciones(ctx or {})
    campos = [dict(spec["fallback"]) for spec in especificaciones]
    fuentes = ["respaldo"] * len(especificaciones)

    with ThreadPoolExecutor(max_workers=_FICHA_WORKERS) as executor:
        futuros = {
            executor.submit(_campos_de_ficha, spec): i
            for i, spec in enumerate(especificaciones)
        }
        for fut in as_completed(futuros):
            i = futuros[fut]
            try:
                resultado, fuente = fut.result()
            except Exception:
                continue
            if isinstance(resultado, dict) and resultado:
                campos[i] = resultado
                fuentes[i] = fuente

    return [
        {
            "titulo": spec["titulo"],
            "referencia": spec["referencia"],
            "campos": campos[i],
            "fuente": fuentes[i],
            "imagen": spec.get("imagen"),
            "pie_imagen": spec.get("pie_imagen"),
        }
        for i, spec in enumerate(especificaciones)
    ]


# ============================================================
# CLASE PDF: especializa ReporteCafeteriaEst con la ficha didáctica
# ============================================================
def _dimensiones_png(datos: bytes | None) -> tuple | None:
    """(ancho, alto) en píxeles del PNG, o None si no se puede leer."""
    if not datos:
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(datos)) as img:
            return img.size
    except Exception:
        return None


class ReporteDidacticoEst(ReporteCafeteriaEst):
    """PDF didáctico: portada, índice de fichas y fichas D1-D10.

    Hereda de `ReporteCafeteriaEst` (fuentes Unicode, header/footer, índices
    y tablas). El reporte institucional no se toca: aquí solo se agrega el
    método de dibujo de la ficha y la portada en su variante didáctica.
    """

    def portada_institucional(self):
        """Portada institucional en su variante didáctica."""
        self._saltar_header_footer(True)
        self.add_page()

        logo_tec = _get_asset_path("logo_tecnm_virtual.png")
        logo_ia = _get_asset_path("logo_maestria_ia.png")
        if logo_tec:
            self.image(logo_tec, x=92, y=18, w=26)
        if logo_ia:
            self.image(logo_ia, x=91, y=232, w=28)

        self.set_y(66)
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(*AZUL)
        self.cell(0, 8, "INSTITUTO TECNOLÓGICO DE TIJUANA", 0, 1, "C")
        self.set_font(self.FONT, "", 11)
        self.cell(0, 6, PROGRAMA, 0, 1, "C")
        self.ln(6)

        self.set_draw_color(*AZUL)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)

        self.set_font(self.FONT, "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 9, "Reporte Didáctico\nEstadística Aplicada a la "
                              "Cafetería", 0, "C")
        self.ln(3)
        self.set_font(self.FONT, "I", 12)
        self.cell(0, 7, "Fichas de las figuras y tablas — Unidades 1 a 4",
                  0, 1, "C")
        self.ln(6)

        self.set_draw_color(*AZUL)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(14)

        self.set_font(self.FONT, "", 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, "Autores:", 0, 1, "C")
        self.set_font(self.FONT, "B", 11)
        for autor in AUTORES:
            self.cell(0, 6, autor, 0, 1, "C")
        self.ln(7)

        self.set_font(self.FONT, "", 10)
        self.cell(0, 6, f"Docente: {DOCENTE}", 0, 1, "C")
        self.cell(0, 6, INSTITUCION, 0, 1, "C")

        self.ln(8)
        self.set_font(self.FONT, "I", 10)
        self.cell(0, 6, "Tijuana, Baja California", 0, 1, "C")
        self.cell(0, 6, f"Noviembre {datetime.now().year}", 0, 1, "C")

        self._saltar_header_footer(False)

    def indice_fichas(self):
        """Índice de fichas con folio alineado y puntos guía."""
        self._cabecera_indice("Índice de Fichas Didácticas")
        for titulo, pagina in self.indice_caps.items():
            self._fila_indice(titulo, pagina)
        self._saltar_header_footer(False)

    def ficha_didactica(self, numero: int, titulo: str, campos: dict,
                        referencia: str | None = None,
                        cita_fuente: str | None = None,
                        imagen_bytes: bytes | None = None,
                        pie_imagen: str | None = None):
        """Dibuja una ficha didáctica de siete bloques.

        `campos` trae las siete claves de `CAMPOS_DIDACTICOS`; si falta una se
        dibuja un aviso neutral para que la ficha nunca quede vacía.
        `referencia` indica de dónde sale el objeto explicado, `cita_fuente`
        la procedencia del texto y `imagen_bytes` el PNG de la figura.
        """
        ancho_util = self.w - self.l_margin - self.r_margin
        if self.get_y() > self.h - self.b_margin - 45:
            self.add_page()

        # --- Banda de título ---
        encabezado = f"  Ficha D{numero}. {titulo}"
        self.set_font(self.FONT, "B", 11)
        if self.get_string_width(encabezado) > ancho_util - 4:
            self.set_font(self.FONT, "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.set_x(self.l_margin)
        self.cell(ancho_util, 9, encabezado, 0, 1, "L", True)
        self.set_text_color(0, 0, 0)

        # --- Procedencia del objeto explicado ---
        notas = [t for t in (referencia, cita_fuente) if t]
        if notas:
            self.set_font(self.FONT, "I", 7.5)
            self.set_text_color(*GRIS)
            for nota in notas:
                self.set_x(self.l_margin)
                self.multi_cell(ancho_util, 4, nota, 0, "L")
            self.set_text_color(0, 0, 0)
        self.ln(2)

        # --- Figura explicada (si la ficha la tiene) ---
        if imagen_bytes:
            dim = _dimensiones_png(imagen_bytes)
            ancho_img, alto_img = 145.0, 0.0
            if dim and dim[0]:
                alto_img = ancho_img * dim[1] / dim[0]
                if alto_img > 68:
                    alto_img = 68.0
                    ancho_img = alto_img * dim[0] / dim[1]
            if alto_img and self.get_y() + alto_img + 10 > self.h - self.b_margin:
                self.add_page()
            try:
                self.set_x(self.l_margin)
                self.image(io.BytesIO(imagen_bytes),
                           x=(self.w - ancho_img) / 2, w=ancho_img)
                self.ln(1)
            except Exception:
                pass
            if pie_imagen:
                self.set_font(self.FONT, "I", 7.5)
                self.set_text_color(*GRIS)
                self.multi_cell(ancho_util, 4, pie_imagen, 0, "C")
                self.set_text_color(0, 0, 0)
            self.ln(2)

        # --- Siete bloques de contenido ---
        for clave, etiqueta in CAMPOS_DIDACTICOS:
            texto = str((campos or {}).get(clave) or _BLOQUE_SIN_TEXTO).strip()

            # Salto de página preventivo antes de cada bloque
            if self.get_y() > self.h - self.b_margin - 28:
                self.add_page()
                self.set_x(self.l_margin)
                self.set_font(self.FONT, "I", 7.5)
                self.set_text_color(*GRIS)
                self.cell(0, 4, f"Ficha D{numero} (continuación)", 0, 1, "L")
                self.set_text_color(0, 0, 0)
                self.ln(1)

            color = _COLOR_CAMPO_DIDACTICO.get(clave, (0, 51, 102))
            self.set_x(self.l_margin)
            if clave == "no_aplica":
                # Bloque crítico: fondo ámbar para que no pase desapercibido
                self.set_fill_color(255, 243, 205)
                self.set_text_color(*color)
                self.set_font(self.FONT, "B", 9)
                self.cell(ancho_util, 6, f" {etiqueta}", 0, 1, "L", True)
                self.set_x(self.l_margin)
                self.set_font(self.FONT, "", 9)
                self.multi_cell(ancho_util, 4.6, texto, 0, "L")
            else:
                self.set_text_color(*color)
                self.set_font(self.FONT, "B", 9)
                self.multi_cell(ancho_util, 5, etiqueta, 0, "L")
                self.set_x(self.l_margin)
                self.set_text_color(0, 0, 0)
                self.set_font(self.FONT, "", 9)
                self.multi_cell(ancho_util, 4.6, texto, 0, "L")
            self.set_text_color(0, 0, 0)
            self.ln(2.5)


# ============================================================
# CUERPO DEL DOCUMENTO Y TRES PASADAS
# ============================================================
def _cuerpo_didactico(pdf, fichas: list, ctx: dict):
    """Dibuja la introducción y las diez fichas; registra el índice."""
    pdf.add_page()
    pdf.cap_actual = ""
    pdf.set_x(pdf.l_margin)
    pdf.set_font(pdf.FONT, "B", 16)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 10, "Reporte Didáctico", 0, 1, "L")
    pdf.set_font(pdf.FONT, "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, (
        "Este documento acompaña al reporte técnico y explica, una por una, "
        "las figuras y tablas que aparecen en él. La idea es que cualquier "
        "persona pueda reconocer el concepto estadístico que hay detrás de "
        "cada resultado, sin necesidad de dominar la notación matemática de "
        "antemano."
    ))
    pdf.ln(2)
    pdf.subtitulo("Cómo leer cada ficha")
    pdf.parrafo(
        "Cada ficha sigue siempre el mismo orden de siete bloques. Los seis "
        "primeros construyen la comprensión: qué se ve, qué significa "
        "técnicamente, cómo se explicaría en palabras simples, a qué unidad "
        "del temario corresponde, por qué importa aprenderlo y qué errores se "
        "cometen al interpretarlo. El séptimo bloque responde una pregunta "
        "que casi nunca se plantea: qué técnicas del temario NO se aplican en "
        "esa figura o tabla y por qué. Ese bloque evita el error de creer que "
        "toda técnica sirve para cualquier problema."
    )
    pdf.subtitulo("Estructura de cada ficha")
    pdf.tabla_metrica(
        [(etiqueta, _DESCRIPCION_CAMPO.get(clave, ""))
         for clave, etiqueta in CAMPOS_DIDACTICOS],
        ancho1=58, ancho2=132,
    )
    pdf.subtitulo("Mapa de fichas incluidas")
    pdf.tabla_metrica(
        [(f"D{i}. {ficha['titulo']}", ficha.get("referencia") or "")
         for i, ficha in enumerate(fichas, 1)],
        ancho1=72, ancho2=118,
    )
    con_ia = sum(1 for ficha in fichas if ficha.get("fuente") == "ia")
    pdf.set_font(pdf.FONT, "I", 8)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(0, 4, (
        f"Contenido generado con IA a partir de las métricas reales de este "
        f"proyecto ({con_ia} de {len(fichas)} fichas redactadas por Gemini); "
        f"el resto usa el respaldo redactado sobre las mismas cifras."
    ))
    pdf.set_text_color(0, 0, 0)

    # --- Fichas: una por página, registradas en el índice ---
    pdf.cap_actual = "Reporte Didáctico · Fichas"
    for i, ficha in enumerate(fichas, 1):
        pdf.add_page()
        pdf.indice_caps[f"D{i}. {ficha['titulo']}"] = pdf.page_no()
        pdf.ficha_didactica(
            i,
            ficha["titulo"],
            ficha.get("campos") or {},
            referencia=ficha.get("referencia"),
            cita_fuente=(
                "Texto redactado por IA (Gemini) a partir de las métricas."
                if ficha.get("fuente") == "ia"
                else "Respaldo redactado sobre las mismas métricas."
            ),
            imagen_bytes=ficha.get("imagen"),
            pie_imagen=ficha.get("pie_imagen"),
        )


def _render_fichas(pdf, fichas: list, ctx: dict, indice: dict | None = None,
                   total_pages: int = 0):
    """Dibuja un documento didáctico completo; `indice` precarga el índice."""
    pdf._init_registros()
    pdf.total_pages = total_pages
    if indice:
        pdf.indice_caps = dict(indice)

    pdf.portada_institucional()
    pdf.indice_fichas()

    # A partir de aquí se reconstruye el índice con las páginas reales.
    pdf.indice_caps = {}
    _cuerpo_didactico(pdf, fichas, ctx)


def _render_didactico(fichas: list, ctx: dict):
    """Tres pasadas: borrador, paginado y final con el folio correcto."""
    borrador = ReporteDidacticoEst()
    borrador.modo = "borrador"
    borrador.set_auto_page_break(auto=True, margin=15)
    _render_fichas(borrador, fichas, ctx)

    paginado = ReporteDidacticoEst()
    paginado.modo = "borrador"
    paginado.set_auto_page_break(auto=True, margin=15)
    _render_fichas(paginado, fichas, ctx, indice=dict(borrador.indice_caps))

    final = ReporteDidacticoEst()
    final.modo = "final"
    final.set_auto_page_break(auto=True, margin=15)
    _render_fichas(final, fichas, ctx, indice=dict(paginado.indice_caps),
                   total_pages=paginado.page_no())
    return final


def generar_reporte_didactico_est_pdf(df: pd.DataFrame | None = None,
                                      min_registros: int = 20
                                      ) -> tuple[bytes | None, str | None]:
    """Genera el Reporte Didáctico y devuelve `(pdf_bytes, error)`.

    `error` es None cuando el documento se construyó; en caso contrario
    explica el motivo (pocos registros, Supabase sin conexión, excepción).
    El contenido de cada ficha se redacta con
    `analizar_seccion_ia(tipo="didactico")` en paralelo y cae a un respaldo
    curado bloque por bloque si la IA no está disponible.
    """
    try:
        if df is None:
            df, error_carga = dm.load_data()
            if error_carga:
                return None, f"No se pudo cargar de Supabase: {error_carga}"
        disponibles = 0 if df is None else len(df)
        if disponibles < min_registros:
            return None, (
                f"Se requieren al menos {min_registros} registros para el "
                f"reporte didáctico (disponibles: {disponibles})."
            )

        ctx = _contexto(df)
        fichas = _fichas_didacticas(ctx)
        pdf = _render_didactico(fichas, ctx)
        return bytes(pdf.output()), None
    except Exception as exc:  # pragma: no cover - salvaguarda para la UI
        traceback.print_exc()
        return None, f"Error al generar reporte didáctico: {exc}"
