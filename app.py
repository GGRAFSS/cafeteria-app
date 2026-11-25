"""
App Streamlit: Cafetería Universitaria
Pestañas:
  1) Captura
  2) Registros
  3) Estadísticas (Unidad 1)
  4) Distribuciones (Unidad 2)
  5) Hipótesis (Unidad 3) — próximamente
  6) Bayes (Unidad 4) — próximamente
"""
import matplotlib

matplotlib.use("Agg")  # sin GUI: Streamlit y el reporte PDF corren en servidor
import streamlit as st
import pandas as pd
import numpy as np
import data_manager as dm
import stats_utils as su
import viz_utils as vu
import mod_distribuciones as md
import mod_hipotesis
import mod_bayes
from mod_reporte_est import generar_reporte_est_pdf
# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Cafetería Universitaria — Captura y Análisis",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CARGA DE DATOS CON MANEJO DE ERROR
# ============================================================
df, load_error = dm.load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/coffee-to-go.png",
        width=80,
    )
    st.markdown("### 📋 Información del proyecto")
    st.markdown(
        """
        **Proyecto:** Análisis estadístico de clientes en una cafetería universitaria.

        **Periodo de captura:** 7:00 AM – 11:00 AM.

        **Dataset:** 200+ clientes simulados con 9 variables.

        **Materia:** Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos.

        **Docente:** Dr. Valdez Acosta Fevrier Adolfo.
        """
    )

    st.divider()

    st.markdown("### 👤 Autor")
    st.markdown(
        """
        **Integrantes:** Rodríguez Avilez Selene y Raygoza Flores Gabriel  
        **Institución:** Instituto Tecnológico de Tijuana - TecNM  
        **Periodo:** 2026
        """
    )

   
    st.divider()

    st.markdown("### 🔌 Estado de la conexión")
    if load_error:
        st.error("❌ Sin conexión a Supabase")
    else:
        st.success(f"✅ Conectado ({len(df)} registros)")

    st.divider()

    if st.button("🔄 Recargar datos desde Supabase", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    with st.expander("📖 Diccionario de variables"):
        st.markdown(
            """
            | Variable | Tipo | Descripción |
            |---|---|---|
            | `ID_Cliente` | Entero | Identificador único |
            | `Hora_Llegada` | Float | Hora decimal (8.25 = 8:15 AM) |
            | `Tiempo_Espera_Min` | Float | Minutos antes de ser atendido |
            | `Tiempo_Servicio_Min` | Float | Minutos de atención del barista |
            | `Tipo_Cliente` | Texto | Estudiante / Profesor / Administrativo |
            | `Metodo_Pago` | Texto | Efectivo / Tarjeta / App (permite nulo) |
            | `Consumo_Total_MXN` | Float | Monto gastado en pesos |
            | `Satisfaccion_1_al_5` | Entero | Calificación 1–5 (permite nulo) |
            | `Es_Hora_Pico` | Texto | Sí / No (8:00–9:00 AM) |
            """
        )

    st.divider()
    st.caption("💾 Datos almacenados en Supabase (nube)")

# ============================================================
# REPORTE ESTADÍSTICO EN PDF (sidebar)
# ============================================================
with st.sidebar:
    st.markdown("### 📄 Reporte estadístico")
    st.caption(
        "Documento completo: portada, índices, resumen ejecutivo y las cuatro "
        "unidades del curso."
    )
    if st.button("📄 Generar Reporte PDF", use_container_width=True):
        with st.spinner("Calculando métricas y maquetando el PDF…"):
            pdf_bytes, error_reporte = generar_reporte_est_pdf(df)
        if error_reporte:
            st.session_state.pop("reporte_pdf", None)
            st.error(f"No se pudo generar el reporte: {error_reporte}")
        else:
            st.session_state["reporte_pdf"] = pdf_bytes
            st.success("Reporte generado. Descárgalo abajo.")

    if st.session_state.get("reporte_pdf"):
        st.download_button(
            "⬇️ Descargar reporte PDF",
            data=st.session_state["reporte_pdf"],
            file_name="reporte_cafeteria.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ============================================================
# BANNER DE ERROR
# ============================================================
if load_error:
    st.error(
        f"⚠️ **No se pudo conectar con la base de datos.**\n\n"
        f"Detalle técnico: `{load_error}`\n\n"
        f"Verifica tu conexión a internet y que el proyecto de Supabase esté activo."
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Reintentar", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    st.stop()

# ============================================================
# TÍTULO PRINCIPAL
# ============================================================
st.title("☕ Cafetería Universitaria — Captura y Análisis")
st.markdown(
    """
    **Análisis estadístico de clientes en horario matutino (7:00 AM – 11:00 AM).**
    Esta aplicación cubre las 4 unidades del curso:
    **Unidad 1** (estadística descriptiva y visualización), **Unidad 2** (distribuciones de probabilidad),
    **Unidad 3** (pruebas de hipótesis y validación de modelos) y **Unidad 4** (estadística bayesiana).
    """
)
st.divider()

# ============================================================
# PESTAÑAS (6)
# ============================================================
tabs = st.tabs([
    "📝 Captura",
    "📋 Registros",
    "📊 Estadísticas (U1)",
    "📈 Distribuciones (U2)",
    "🧪 Hipótesis (U3)",
    "🎲 Bayes (U4)",
])

# ============ 1. CAPTURA ============
with tabs[0]:
    st.subheader("Nuevo registro")
    with st.form("form_captura", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            hora = st.number_input("Hora de llegada (7.00–11.00)", 7.00, 11.00, 8.00, 0.25,
                                   help="Formato decimal: 8.25 = 8:15 AM")
            espera = st.number_input("Tiempo de espera (min)", 0.1, 60.0, 3.0, 0.5)
            servicio = st.number_input("Tiempo de servicio (min)", 0.1, 30.0, 4.0, 0.5)
        with c2:
            tipo = st.selectbox("Tipo de cliente", ["Estudiante", "Profesor", "Administrativo"])
            pago = st.selectbox("Método de pago", ["(Nulo)", "Efectivo", "Tarjeta", "App"])
            consumo = st.number_input("Consumo total (MXN)", 0.0, 500.0, 45.0, 1.0)
        with c3:
            sat = st.selectbox("Satisfacción (1–5)", ["(Nulo)", 1, 2, 3, 4, 5])
            pico = st.selectbox("¿Hora pico (8–9 AM)?", ["No", "Sí"])

        submitted = st.form_submit_button("💾 Guardar registro")

    if submitted:
        if st.session_state.get("guardando", False):
            st.warning("⏳ Ya hay un guardado en proceso, espera un momento.")
        else:
            st.session_state["guardando"] = True
            record = {
                "ID_Cliente": dm.next_id(df),
                "Hora_Llegada": float(hora),
                "Tiempo_Espera_Min": float(espera),
                "Tiempo_Servicio_Min": float(servicio),
                "Tipo_Cliente": tipo,
                "Metodo_Pago": None if pago == "(Nulo)" else pago,
                "Consumo_Total_MXN": float(consumo),
                "Satisfaccion_1_al_5": None if sat == "(Nulo)" else int(sat),
                "Es_Hora_Pico": pico,
            }
            errors = dm.validate_record(record)
            if errors:
                for e in errors:
                    st.error(e)
                st.session_state["guardando"] = False
            else:
                ok, save_error = dm.save_record(record)
                if ok:
                    st.success(f"✅ Registro ID {record['ID_Cliente']} guardado en Supabase.")
                    st.session_state["guardando"] = False
                    st.rerun()
                else:
                    st.error(f"❌ Error al guardar en Supabase: {save_error}")
                    st.session_state["guardando"] = False

# ============ 2. REGISTROS ============
with tabs[1]:
    st.subheader(f"Registros capturados ({len(df)})")
    if df.empty:
        st.info("Aún no hay registros.")
    else:
        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("### 🧹 Limpieza de datos")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🧹 Aplicar limpieza de outliers", use_container_width=True):
                clean = dm.clean_data(df)
                st.write(f"Filas originales: {len(df)} → después de limpieza: {len(clean)}")
                st.dataframe(clean, use_container_width=True)

        with col_b:
            if st.button("🗑️ Eliminar filas con NaN", use_container_width=True):
                before = len(df)
                df_no_nan = df.dropna()
                after = len(df_no_nan)
                st.write(f"Filas originales: {before} → después de eliminar NaN: {after}")
                if after == before:
                    st.info("No había filas con valores nulos.")
                else:
                    st.success(f"Se eliminaron {before - after} filas con valores nulos.")
                    st.dataframe(df_no_nan, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv,
                           "cafeteria_universitaria.csv", "text/csv")

# ============ 3. ESTADÍSTICAS (U1) ============
with tabs[2]:
    st.subheader("Estadísticas descriptivas (Unidad 1)")
    if df.empty:
        st.info("Sin datos para analizar.")
    else:
        clean = dm.clean_data(df)
        kpis = su.executive_summary(df, clean)

        st.markdown("### 📌 Resumen ejecutivo")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Registros totales", f"{kpis['n_total']}",
                  delta=f"{kpis['outliers']} outliers" if kpis["outliers"] else "sin outliers",
                  delta_color="inverse")
        k2.metric("Consumo promedio", f"${kpis['consumo_prom']:.2f}")
        k3.metric("Consumo total", f"${kpis['consumo_total']:,.2f}")
        k4.metric("Satisfacción prom.", f"{kpis['sat_prom']:.2f} / 5")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Espera promedio", f"{kpis['espera_prom']:.2f} min")
        k6.metric("Espera máxima", f"{kpis['espera_max']:.2f} min")
        k7.metric("Servicio promedio", f"{kpis['servicio_prom']:.2f} min")
        k8.metric("Clientes en hora pico", f"{kpis['pct_pico']:.1f} %")

        st.divider()

        st.markdown("### 📊 Tablas descriptivas")
        st.markdown("**Dataset completo**")
        st.dataframe(su.descriptive_stats(df), use_container_width=True)

        st.markdown("**Dataset limpio (sin outliers)**")
        st.dataframe(su.descriptive_stats(clean), use_container_width=True)

        st.markdown("**Tablas de frecuencia**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Tipo_Cliente")
            st.dataframe(su.frequency_table(df, "Tipo_Cliente"), use_container_width=True)
        with c2:
            st.write("Metodo_Pago")
            st.dataframe(su.frequency_table(df, "Metodo_Pago"), use_container_width=True)

        st.divider()

        st.markdown("### 🎯 Outliers detectados (método IQR)")
        with st.expander("ℹ️ ¿Cómo se detectan los outliers?", expanded=False):
            st.markdown(
                """
                Se utiliza el **método del Rango Intercuartílico (IQR)**:
                1. Q1 (percentil 25) y Q3 (percentil 75).
                2. IQR = Q3 − Q1.
                3. Outliers: fuera de **[Q1 − 1.5·IQR , Q3 + 1.5·IQR]**.
                """
            )

        out_summary, out_detail = su.detect_outliers_iqr(df)
        if not out_summary.empty:
            st.dataframe(out_summary, use_container_width=True)
            if not out_detail.empty:
                st.markdown(f"**Filas con outliers ({len(out_detail)} de {len(df)})**")
                st.dataframe(out_detail, use_container_width=True)

        st.divider()
        st.markdown("### 📈 Visualizaciones (Unidad 1)")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(vu.plot_hist(df["Tiempo_Espera_Min"],
                                   "Histograma — Tiempo de Espera",
                                   "Tiempo_Espera_Min"))
        with c2:
            st.pyplot(vu.plot_hist(df["Consumo_Total_MXN"],
                                   "Histograma — Consumo Total",
                                   "Consumo_Total_MXN"))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(vu.plot_box(df["Tiempo_Servicio_Min"],
                                  "Boxplot — Tiempo de Servicio",
                                  "Tiempo_Servicio_Min"))
        with c2:
            st.pyplot(vu.plot_box_by_group(df, "Consumo_Total_MXN", "Tipo_Cliente",
                                           "Consumo vs Tipo de Cliente",
                                           "Consumo_Total_MXN"))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(vu.plot_bars(df, "Tipo_Cliente", "Conteo — Tipo_Cliente"))
        with c2:
            st.pyplot(vu.plot_bars(df, "Metodo_Pago", "Conteo — Metodo_Pago"))

# ============ 4. DISTRIBUCIONES (U2) ============
with tabs[3]:
    st.subheader("Distribuciones de probabilidad (Unidad 2)")
    if df.empty:
        st.info("Sin datos para analizar.")
    else:
        clean = dm.clean_data(df)

        st.markdown("### 📊 Ajuste de curvas de densidad")
        st.markdown(
            "Se superpone la curva teórica ajustada (Normal para servicio, "
            "Exponencial para espera) al histograma empírico, con Q-Q plot al lado."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(md.plot_ajuste_con_qq(
                df["Tiempo_Espera_Min"], dist="expon",
                title_prefix="Tiempo de Espera"
            ))
        with c2:
            st.pyplot(md.plot_ajuste_con_qq(
                clean["Tiempo_Servicio_Min"], dist="norm",
                title_prefix="Tiempo de Servicio"
            ))

        st.divider()
        st.markdown("### 🎲 Simulación Poisson — llegadas por minuto")
        st.markdown(
            "Se simula el número de llegadas por minuto con una distribución "
            "**Poisson(λ)**, donde λ se estima a partir de los datos reales. "
            "Se compara la distribución simulada con la empírica."
        )

        fig_hist_pois, fig_qq_pois, resumen_pois = md.simular_poisson_llegadas(df)

        if fig_hist_pois is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_hist_pois)
            with c2:
                st.pyplot(fig_qq_pois)

            st.markdown("**Resumen de la simulación**")
            st.dataframe(pd.DataFrame([resumen_pois]).T.rename(columns={0: "Valor"}),
                         use_container_width=True)
        else:
            st.warning("No se pudo simular Poisson (datos insuficientes o columna faltante).")

        st.divider()
        st.markdown("### 🧪 Validación de supuestos (test KS)")
        st.markdown(
            "El test de **Kolmogorov-Smirnov** contrasta la hipótesis nula de que los datos "
            "provienen de la distribución ajustada. Si **p > 0.05**, no se rechaza H₀ "
            "(los datos son compatibles con la distribución)."
        )

        tabla_ks = md.tabla_bondad_ajuste(df, clean)
        if not tabla_ks.empty:
            st.dataframe(tabla_ks, use_container_width=True)
            st.caption("α = 0.05. Regla: si p-value > 0.05, no se rechaza la distribución propuesta.")
        else:
            st.info("No hay suficientes datos para calcular los tests.")

# ============ 5. HIPÓTESIS (U3) — placeholder ============
# ============ 5. HIPÓTESIS (U3) ============
with tabs[4]:
    st.subheader("Pruebas de hipótesis y validación de modelos (Unidad 3)")
    if df.empty:
        st.info("Sin datos para analizar.")
    else:
        # ---------- PRUEBAS PARA DOS GRUPOS ----------
        st.markdown("### 🧪 Comparación de dos grupos")
        st.markdown(
            "Compara una variable numérica entre dos grupos de clientes usando "
            "pruebas **paramétricas** (t de Student) y **no paramétricas** (Mann-Whitney U). "
            "Se verifica el supuesto de normalidad (Shapiro-Wilk) y homocedasticidad (Levene)."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            valor_col = st.selectbox(
                "Variable a comparar",
                ["Consumo_Total_MXN", "Tiempo_Espera_Min", "Tiempo_Servicio_Min"],
                key="u3_valor"
            )
        with col2:
            grupo_col = st.selectbox(
                "Variable de agrupación",
                ["Tipo_Cliente", "Es_Hora_Pico", "Metodo_Pago"],
                key="u3_grupo"
            )
        with col3:
            # Detectar grupos disponibles dinámicamente
            if grupo_col in df.columns:
                grupos = [g for g in df[grupo_col].dropna().unique() if str(g) != "nan"]
                grupos = sorted([str(g) for g in grupos])
            else:
                grupos = []

        if len(grupos) < 2:
            st.warning("La variable de agrupación no tiene al menos 2 grupos válidos.")
        else:
            gc1, gc2 = st.columns(2)
            with gc1:
                grupo_a = st.selectbox("Grupo A", grupos, index=0, key="u3_grupo_a")
            with gc2:
                grupo_b = st.selectbox("Grupo B", grupos, index=1, key="u3_grupo_b")

            if grupo_a == grupo_b:
                st.error("Selecciona dos grupos distintos.")
            else:
                res = mod_hipotesis.comparar_dos_grupos(
                    df, valor_col, grupo_col, grupo_a, grupo_b
                )

                if "error" in res and res["error"]:
                    st.error(res["error"])
                else:
                    # Resultados descriptivos
                    st.markdown("#### 📊 Descriptivos por grupo")
                    desc = pd.DataFrame({
                        "Grupo": [grupo_a, grupo_b],
                        "n": [res["n_a"], res["n_b"]],
                        "Media": [round(res["media_a"], 3), round(res["media_b"], 3)],
                        "Desv_Std": [round(res["std_a"], 3), round(res["std_b"], 3)],
                    })
                    st.dataframe(desc, use_container_width=True)

                    # Visualización
                    st.pyplot(mod_hipotesis.plot_comparacion_grupos(
                        df, valor_col, grupo_col, grupo_a, grupo_b,
                        f"{valor_col} — {grupo_a} vs {grupo_b}"
                    ))

                    # Validación de supuestos
                    st.markdown("#### 🔍 Validación de supuestos")
                    supuestos = pd.DataFrame({
                        "Prueba": ["Shapiro-Wilk (" + grupo_a + ")", 
                                   "Shapiro-Wilk (" + grupo_b + ")",
                                   "Levene (homocedasticidad)"],
                        "p-value": [
                            round(res["shapiro_a_p"], 4) if not np.isnan(res["shapiro_a_p"]) else "N/A",
                            round(res["shapiro_b_p"], 4) if not np.isnan(res["shapiro_b_p"]) else "N/A",
                            round(res["levene_p"], 4),
                        ],
                        "Conclusión": [
                            "Normal" if res["shapiro_a_p"] > 0.05 else "No normal",
                            "Normal" if res["shapiro_b_p"] > 0.05 else "No normal",
                            "Varianzas iguales" if res["levene_p"] > 0.05 else "Varianzas distintas",
                        ],
                    })
                    st.dataframe(supuestos, use_container_width=True)

                    # Resultados de las pruebas
                    st.markdown("#### 🎯 Resultados de las pruebas")
                    p1, p2 = st.columns(2)

                    with p1:
                        st.markdown("**Prueba t de Student**")
                        tipo_t = "Welch" if not res["ttest_equal_var"] else "clásica"
                        st.metric("Estadístico t", f"{res['ttest_stat']:.4f}")
                        st.metric("p-value", f"{res['ttest_p']:.4f}")
                        st.caption(f"Variante: {tipo_t}")
                        if res["ttest_p"] < 0.05:
                            st.success("✅ Rechaza H₀ (diferencia significativa)")
                        else:
                            st.info("ℹ️ No rechaza H₀ (sin diferencia significativa)")

                    with p2:
                        st.markdown("**Mann-Whitney U**")
                        st.metric("Estadístico U", f"{res['mannwhitney_u']:.1f}")
                        st.metric("p-value", f"{res['mannwhitney_p']:.4f}")
                        st.caption("Alternativa no paramétrica")
                        if res["mannwhitney_p"] < 0.05:
                            st.success("✅ Rechaza H₀ (diferencia significativa)")
                        else:
                            st.info("ℹ️ No rechaza H₀ (sin diferencia significativa)")

                    st.metric("Tamaño del efecto (Cohen's d)",
                              f"{res['cohen_d']:.3f} → {res['interpretacion_d']}")

        st.divider()

        # ---------- MODELO DE CLASIFICACIÓN ----------
        st.markdown("### 🤖 Modelo de clasificación — Regresión Logística")
        st.markdown(
            "Se entrena una **regresión logística** para predecir si un cliente llega en "
            "**hora pico** (`Es_Hora_Pico = Sí`) a partir de las variables numéricas y categóricas. "
            "Se muestran matriz de confusión, métricas (accuracy, precision, recall, F1) y curva ROC."
        )

        if st.button("🚀 Entrenar modelo de clasificación", key="btn_train_logreg"):
            with st.spinner("Entrenando..."):
                res_ml = mod_hipotesis.entrenar_regresion_logistica(df, target_col="Es_Hora_Pico")

            if res_ml.get("error"):
                st.error(res_ml["error"])
            else:
                st.success(
                    f"✅ Modelo entrenado con {res_ml['n_train']} registros de entrenamiento "
                    f"y {res_ml['n_test']} de prueba."
                )

                # Métricas principales
                st.markdown("#### 📊 Métricas del modelo")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{res_ml['metricas']['accuracy']:.3f}")
                m2.metric("Precision", f"{res_ml['metricas']['precision']:.3f}")
                m3.metric("Recall", f"{res_ml['metricas']['recall']:.3f}")
                m4.metric("F1-Score", f"{res_ml['metricas']['f1']:.3f}")
                m5.metric("AUC-ROC", f"{res_ml['metricas']['auc']:.3f}")

                # Gráficas
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(res_ml["fig_cm"])
                with c2:
                    st.pyplot(res_ml["fig_roc"])

                # Reporte detallado
                st.markdown("#### 📋 Reporte por clase")
                st.dataframe(res_ml["reporte"], use_container_width=True)
        else:
            st.info("Haz clic en el botón para entrenar el modelo.")
# ============ 6. BAYES (U4) — placeholder ============
# ============ 6. BAYES (U4) ============
with tabs[5]:
    st.subheader("Estadística bayesiana (Unidad 4)")
    if df.empty:
        st.info("Sin datos para analizar.")
    else:
        # ---------- CALCULADORA DE BAYES ----------
        st.markdown("### 🧮 Calculadora del Teorema de Bayes")
        st.markdown(
            "Calcula **P(A|B)** a partir de la probabilidad a priori **P(A)**, "
            "la verosimilitud **P(B|A)** y la verosimilitud **P(B|¬A)**."
        )

        with st.expander("ℹ️ Fórmula del Teorema de Bayes", expanded=False):
            st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}")

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            p_a = st.slider("P(A) — Prior", 0.0, 1.0, 0.30, 0.01, key="bayes_pa",
                            help="Probabilidad a priori del evento A")
            etiqueta_a = st.text_input("Etiqueta de A", value="Profesor", key="bayes_ea")
        with bc2:
            p_b_dado_a = st.slider("P(B|A) — Verosimilitud", 0.0, 1.0, 0.60, 0.01, key="bayes_pba",
                                   help="Probabilidad de B dado que ocurrió A")
            etiqueta_b = st.text_input("Etiqueta de B", value="Paga con Tarjeta", key="bayes_eb")
        with bc3:
            p_b_dado_no_a = st.slider("P(B|¬A) — Verosimilitud", 0.0, 1.0, 0.40, 0.01, key="bayes_pbna",
                                      help="Probabilidad de B dado que NO ocurrió A")

        # Ejemplos rápidos
        st.markdown("**Ejemplos rápidos (aplica valores predefinidos):**")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if st.button("Profesor → Tarjeta", use_container_width=True, key="ex_bayes1"):
                st.session_state["bayes_pa"] = 0.20
                st.session_state["bayes_pba"] = 0.70
                st.session_state["bayes_pbna"] = 0.30
                st.rerun()
        with ex2:
            if st.button("Satisfacción Alta → Hora Pico", use_container_width=True, key="ex_bayes2"):
                st.session_state["bayes_pa"] = 0.55
                st.session_state["bayes_pba"] = 0.35
                st.session_state["bayes_pbna"] = 0.25
                st.rerun()
        with ex3:
            if st.button("Estudiante → Consumo Alto", use_container_width=True, key="ex_bayes3"):
                st.session_state["bayes_pa"] = 0.70
                st.session_state["bayes_pba"] = 0.20
                st.session_state["bayes_pbna"] = 0.45
                st.rerun()

        resultado = mod_bayes.teorema_bayes(p_a, p_b_dado_a, p_b_dado_no_a)

        st.markdown("#### 📊 Resultado")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"P({etiqueta_a}) — Prior", f"{resultado['p_a']:.4f}")
        r2.metric(f"P({etiqueta_b}) — Evidencia", f"{resultado['p_b']:.4f}")
        r3.metric(f"P({etiqueta_a}|{etiqueta_b})", f"{resultado['p_a_dado_b']:.4f}")
        r4.metric(f"P(¬{etiqueta_a}|{etiqueta_b})", f"{resultado['p_no_a_dado_b']:.4f}")

        st.pyplot(mod_bayes.plot_bayes_barras(resultado, etiqueta_a, etiqueta_b))

        st.divider()

        # ---------- NAIVE BAYES ----------
        st.markdown("### 🤖 Clasificador Naive Bayes")
        st.markdown(
            "Se entrena un **Gaussian Naive Bayes** para predecir si un cliente tendrá "
            "**Satisfacción Alta** (4–5) vs **Baja** (1–3), a partir de las variables numéricas y categóricas. "
            "Se muestran matriz de confusión, métricas de calibración (**Log-loss** y **Brier Score**) "
            "y probabilidades posteriores promedio por clase real."
        )

        if st.button("🚀 Entrenar Naive Bayes", key="btn_train_nb"):
            with st.spinner("Entrenando Naive Bayes..."):
                res_nb = mod_bayes.entrenar_naive_bayes(df)

            if res_nb.get("error"):
                st.error(res_nb["error"])
            else:
                st.success(
                    f"✅ Modelo entrenado con {res_nb['n_train']} registros de entrenamiento "
                    f"y {res_nb['n_test']} de prueba. Features: {res_nb['n_features']}."
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{res_nb['metricas']['accuracy']:.3f}")
                m2.metric("Log-loss", f"{res_nb['metricas']['log_loss']:.3f}",
                          help="Menor es mejor (0 = perfecto)")
                m3.metric("Brier Score", f"{res_nb['metricas']['brier_score']:.3f}",
                          help="Menor es mejor (0 = perfecto)")

                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(res_nb["fig_cm"])
                with c2:
                    st.pyplot(res_nb["fig_proba"])

                st.markdown("#### 📋 Reporte por clase")
                st.dataframe(res_nb["reporte"], use_container_width=True)
        else:
            st.info("Haz clic en el botón para entrenar Naive Bayes.")

        st.divider()

        # ---------- COMPARACIÓN DE MODELOS + FRIEDMAN ----------
        st.markdown("### 📊 Comparación de modelos (prueba de Friedman)")
        st.markdown(
            "Se comparan **3 modelos** de clasificación (Naive Bayes, Regresión Logística, "
            "Árbol de Decisión) con **validación cruzada estratificada de 5 folds**. "
            "Si hay 3 o más modelos, se aplica la **prueba de Friedman** para contrastar "
            "si las diferencias de accuracy son estadísticamente significativas."
        )

        if st.button("📊 Comparar modelos", key="btn_compare"):
            with st.spinner("Entrenando y comparando modelos..."):
                res_comp = mod_bayes.comparar_modelos(df)

            if res_comp.get("error"):
                st.error(res_comp["error"])
            else:
                st.markdown("#### 📋 Resultados de validación cruzada")
                st.dataframe(res_comp["tabla"], use_container_width=True)

                st.pyplot(res_comp["fig"])

                fr = res_comp["friedman"]
                if fr["stat"] is not None:
                    st.markdown("#### 🧪 Prueba de Friedman")
                    f1, f2, f3 = st.columns(3)
                    f1.metric("Estadístico χ²", f"{fr['stat']:.4f}")
                    f2.metric("p-value", f"{fr['p']:.4f}")
                    f3.metric("Grados de libertad", f"{len(res_comp['tabla']) - 1}")
                    if fr["p"] < 0.05:
                        st.success(f"✅ {fr['interpretacion']}")
                    else:
                        st.info(f"ℹ️ {fr['interpretacion']}")
        else:
            st.info("Haz clic en el botón para comparar modelos.")