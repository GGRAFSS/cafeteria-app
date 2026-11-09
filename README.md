# ☕ Cafetería Universitaria — Captura y Análisis

Sistema web desarrollado en **Python + Streamlit** para capturar, almacenar y analizar datos de clientes de una cafetería universitaria en horario matutino (7:00 AM – 11:00 AM). Los datos se almacenan en **Supabase (PostgreSQL en la nube)** y la app se despliega en **Streamlit Community Cloud**, por lo que no requiere instalación por parte del usuario final.

> **🔗 App en vivo:** [https://TU-APP.streamlit.app](https://TU-APP.streamlit.app)  
> **📦 Repositorio:** [https://github.com/TU-USUARIO/cafeteria-app](https://github.com/TU-USUARIO/cafeteria-app)

---

## 📋 Descripción del proyecto

Este proyecto forma parte de la asignatura **Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos** y cubre las 4 unidades del temario:

1. **Unidad 1:** Captura, limpieza, tipos de variables, estadísticas descriptivas y visualización.
2. **Unidad 2:** Distribuciones de probabilidad (Exponencial, Normal, Poisson) con validación KS y Q-Q plots.
3. **Unidad 3:** Pruebas de hipótesis (t de Student, Mann-Whitney U) y validación de modelos (regresión logística, matriz de confusión, ROC).
4. **Unidad 4:** Estadística bayesiana (calculadora de Bayes, Naive Bayes, Log-loss, Brier Score, prueba de Friedman).

El dataset simula **200 clientes** con 9 variables que combinan tipos cuantitativos, cualitativos (nominales, ordinales y binarios), e incluye **outliers inyectados intencionalmente** (`-2.0` en servicio, `300.0` en consumo) para practicar la detección y limpieza.

---

## 🗂️ Estructura del dataset

| # | Variable | Tipo | Clasificación estadística | Descripción |
|---|---|---|---|---|
| 1 | `ID_Cliente` | Entero | Cualitativa nominal | Identificador único del cliente |
| 2 | `Hora_Llegada` | Float | Cuantitativa continua | Hora decimal (8.25 = 8:15 AM) |
| 3 | `Tiempo_Espera_Min` | Float | Cuantitativa continua | Minutos antes de ser atendido |
| 4 | `Tiempo_Servicio_Min` | Float | Cuantitativa continua | Minutos de atención del barista |
| 5 | `Tipo_Cliente` | Texto | Cualitativa nominal | Estudiante / Profesor / Administrativo |
| 6 | `Metodo_Pago` | Texto | Cualitativa nominal | Efectivo / Tarjeta / App (permite nulo) |
| 7 | `Consumo_Total_MXN` | Float | Cuantitativa continua | Monto gastado en pesos |
| 8 | `Satisfaccion_1_al_5` | Entero | Cualitativa ordinal | Calificación 1–5 (permite nulo) |
| 9 | `Es_Hora_Pico` | Texto | Cualitativa binaria | Sí / No (8:00–9:00 AM) |

---

## 🏗️ Arquitectura
┌──────────────────┐ ┌──────────────────┐
│ Streamlit Cloud │────▶│ Supabase (DB) │
│ (frontend web) │ │ (PostgreSQL) │
└──────────────────┘ └──────────────────┘


- **Frontend + lógica:** Streamlit (Python).
- **Base de datos:** Supabase (PostgreSQL en la nube, plan gratuito).
- **Despliegue:** Streamlit Community Cloud (gratuito, conectado a GitHub).

---

## 📁 Estructura del repositorio
cafeteria_app/
├── .streamlit/
│ └── secrets.toml ← credenciales (NO se sube a GitHub)
├── app.py ← interfaz Streamlit (6 pestañas)
├── data_manager.py ← capa Supabase + validaciones + limpieza
├── stats_utils.py ← descriptivas, KPIs, outliers IQR
├── viz_utils.py ← histogramas, boxplots, barras, ajustes
├── mod_distribuciones.py ← Unidad 2: Poisson, Q-Q, KS
├── mod_hipotesis.py ← Unidad 3: t, Mann-Whitney, logística, ROC
├── mod_bayes.py ← Unidad 4: Bayes, Naive Bayes, Friedman
├── seed_supabase.py ← script de carga inicial (200 registros)
├── requirements.txt
├── README.md
└── .gitignore


---

## 🚀 Cómo ejecutar localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU-USUARIO/cafeteria-app.git
cd cafeteria-app