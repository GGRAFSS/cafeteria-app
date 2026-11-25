# ☕ Cafetería Universitaria — Captura y Análisis

Sistema web desarrollado en **Python + Streamlit** para capturar, almacenar y analizar datos de clientes de una cafetería universitaria en horario matutino (7:00 AM – 11:00 AM). Los datos se almacenan en **Supabase (PostgreSQL en la nube)** y la app se despliega en **Streamlit Community Cloud**, por lo que no requiere instalación por parte del usuario final.

> **🔗 App en vivo:** _pendiente de desplegar (Fase 4)_  
> **📦 Repositorio:** [https://github.com/GGRAFSS/cafeteria-app](https://github.com/GGRAFSS/cafeteria-app)

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
├── stats_utils.py ← descriptivas, KPIs, outliers IQR, media ponderada, correlación
├── viz_utils.py ← histogramas, boxplots, barras, ajustes, violín, correlación, árbol bayesiano
├── mod_distribuciones.py ← Unidad 2: Poisson, Q-Q, KS + simulaciones de apoyo
├── mod_hipotesis.py ← Unidad 3: t, Mann-Whitney, ANOVA, chi-cuadrado, logística, ROC
├── mod_bayes.py ← Unidad 4: Bayes, Naive Bayes, Friedman
├── mod_reporte_est.py ← reporte PDF estadístico (portada, índices, 10 capítulos)
├── mod_ia_analista.py ← analista con IA (Gemini) para interpretaciones
├── assets/ ← logos institucionales usados en el PDF
├── seed_supabase.py ← script de carga inicial (200 registros)
├── requirements.txt
├── README.md
└── .gitignore


---

## 🚀 Cómo ejecutar localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/GGRAFSS/cafeteria-app.git
cd cafeteria-app
```

### 2. Crear el entorno e instalar dependencias

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Crear `.streamlit/secrets.toml`

```toml
SUPABASE_URL = "https://<tu-proyecto>.supabase.co"
SUPABASE_KEY = "<tu-anon-key>"
GEMINI_API_KEY = "<tu-api-key-de-google-ai-studio>"
```

> Este archivo está en `.gitignore`: nunca debe subirse al repositorio.

### 4. Ejecutar la app

```bash
streamlit run app.py
```

---

## 🔐 Secrets requeridos

| Clave | Para qué se usa | Dónde se obtiene |
|---|---|---|
| `SUPABASE_URL` | Conexión a la base de datos | Supabase → Project Settings → API |
| `SUPABASE_KEY` | Conexión a la base de datos (anon key) | Supabase → Project Settings → API |
| `GEMINI_API_KEY` | Analista IA (pestaña 6) y textos del reporte | [Google AI Studio](https://aistudio.google.com/apikey) |

Sin `GEMINI_API_KEY` la app **no se rompe**: el módulo IA se desactiva solo y el reporte usa sus textos de respaldo.

---

## ☁️ Despliegue en Streamlit Community Cloud

1. **Subir el código a GitHub** (`main`). Antes de hacer commit, confirma que `.streamlit/secrets.toml` **no** aparece en `git status` (debe estar ignorado).
2. Entra a <https://share.streamlit.io> con la cuenta de GitHub y elige **Create app** → **Deploy a public app from GitHub**.
3. Llena el formulario:
   - **Repository:** `GGRAFSS/cafeteria-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** la que prefieras (ej. `cafeteria-app-ggrafss`)
4. Abre **Advanced settings**:
   - **Python version:** `3.12` (coincide con el archivo `.python-version`).
   - **Secrets:** pega el bloque TOML completo (las 3 claves) y guarda.
5. **Deploy**. La primera instalación tarda ~3–5 min (scipy, matplotlib y scikit-learn son pesadas).
6. **Actualizaciones:** cada `git push` a `main` redespliega la app automáticamente.
7. **Reiniciar la app:**
   - Desde el dashboard: menú **⋮ → Reboot app**.
   - Desde la app abierta: **Manage app → ⋮ → Reboot app**.
   - Al **guardar un secret** (Manage app → Settings → Secrets → Save) la app se reinicia sola; si no ocurre, haz **Reboot app**.

### Secrets en Streamlit Cloud (paso a paso)

1. Abre tu app en <https://share.streamlit.io>.
2. **Manage app** (esquina inferior derecha) → **Settings**.
3. Pestaña **Secrets** y pega exactamente:

```toml
SUPABASE_URL = "https://<tu-proyecto>.supabase.co"
SUPABASE_KEY = "<tu-anon-key>"
GEMINI_API_KEY = "<tu-api-key>"
```

4. **Save** y espera el reinicio (o **Reboot app** manual).
5. Verifica en **Manage app → Logs** que no aparezcan `KeyError: 'SUPABASE_URL'` ni `KeyError: 'GEMINI_API_KEY'`.

### Notas técnicas del despliegue

- El PDF se construye **en memoria** (`pdf.output()` → `bytes`) y se entrega con `st.download_button`; no se escribe nada en disco.
- Las fuentes Unicode del PDF se toman de `matplotlib.get_data_path()/fonts/ttf` (DejaVu), disponibles también en el contenedor Linux de Streamlit Cloud.
- `matplotlib.use("Agg")` ya está forzado en `app.py`, así que no hay dependencias de GUI en el servidor.
- No se requiere `.streamlit/config.toml` ni archivos adicionales para producción.

---

## ✅ Checklist de despliegue

### Antes de hacer commit

- [ ] `git status` **no** muestra `.streamlit/secrets.toml`.
- [ ] `git add -A --dry-run` no lista `_smoke_*.py`, `_verify_*.py` ni `reporte_*.pdf`.
- [ ] `requirements.txt` incluye `fpdf2` y `google-genai`.
- [ ] `assets/logo_maestria_ia.png` y `assets/logo_tecnm_virtual.png` están versionados.
- [ ] `mod_reporte_est.py` y `mod_ia_analista.py` están versionados.

### En Streamlit Cloud (Secrets)

- [ ] `SUPABASE_URL`, `SUPABASE_KEY` y `GEMINI_API_KEY` cargadas en **Settings → Secrets** con formato TOML.
- [ ] La app se reinició después de guardar los secrets.
- [ ] Python declarado como `3.12`.

### En producción (funcionalidad)

- [ ] La app carga el dataset desde Supabase (sin `KeyError` ni `APIError`).
- [ ] El botón **📄 Generar Reporte PDF** termina sin error en un tiempo razonable (~10–30 s).
- [ ] Se descarga `reporte_cafeteria.pdf` (~1 MB) y abre correctamente en el visor.
- [ ] El PDF trae portada con logos, índice general, índice de figuras y de tablas, y los 10 capítulos.
- [ ] La numeración de figuras y tablas es correlativa (sin huecos ni repetidos).
- [ ] **No** aparece ninguna nota `[Pendiente Fase 2]`.
- [ ] La pestaña del analista IA responde (requiere `GEMINI_API_KEY` válida y cuota disponible).
- [ ] **Manage app → Logs** sin `ImportError`, `ModuleNotFoundError`, `KeyError` ni `RuntimeError` del reporte.

### Diagnóstico rápido si algo falla

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'fpdf'` | `requirements.txt` sin `fpdf2` | Agregar la línea, `git push` y **Reboot app** |
| `ModuleNotFoundError: No module named 'google'` | Falta `google-genai` | Ídem (la app funciona sin IA, pero se pierde el analista) |
| `KeyError: 'SUPABASE_URL'` | Secrets vacíos o mal formados | Revisar TOML en Settings → Secrets (comillas dobles) |
| `KeyError: 'GEMINI_API_KEY'` | Secret ausente | Agregarlo y reiniciar la app |
| `RuntimeError: Se requieren al menos 20 registros` | Tabla con menos de 20 filas | Cargar datos (pestaña Captura o `seed_supabase.py`) |
| El PDF tarda y se corta | Límite de recursos del plan gratuito | **Reboot app** y reintentar sin otras pestañas abiertas |


