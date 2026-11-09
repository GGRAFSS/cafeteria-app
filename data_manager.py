"""
Gestión del dataset en Supabase: carga, guardado, validación de reglas de captura
y limpieza (outliers inyectados intencionalmente para la práctica).
Usa el cliente oficial supabase-py (más estable que st-supabase-connection).
"""
import streamlit as st
import pandas as pd
from supabase import create_client, Client

COLUMNS = [
    "ID_Cliente", "Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min",
    "Tipo_Cliente", "Metodo_Pago", "Consumo_Total_MXN",
    "Satisfaccion_1_al_5", "Es_Hora_Pico",
]

TO_DB = {
    "ID_Cliente": "id_cliente",
    "Hora_Llegada": "hora_llegada",
    "Tiempo_Espera_Min": "tiempo_espera_min",
    "Tiempo_Servicio_Min": "tiempo_servicio_min",
    "Tipo_Cliente": "tipo_cliente",
    "Metodo_Pago": "metodo_pago",
    "Consumo_Total_MXN": "consumo_total_mxn",
    "Satisfaccion_1_al_5": "satisfaccion_1_al_5",
    "Es_Hora_Pico": "es_hora_pico",
}
FROM_DB = {v: k for k, v in TO_DB.items()}


@st.cache_resource
def _conn() -> Client:
    """Conexión cacheada a Supabase usando supabase-py directamente."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def load_data() -> tuple[pd.DataFrame, str | None]:
    """
    Carga todos los registros desde Supabase.
    Devuelve (df, error). Si no hay error, error es None.
    """
    try:
        res = _conn().table("cafeteria").select("*").execute()
    except Exception as e:
        return pd.DataFrame(columns=COLUMNS), str(e)

    if not res.data:
        return pd.DataFrame(columns=COLUMNS), None

    try:
        df = pd.DataFrame(res.data).rename(columns=FROM_DB)

        if "Satisfaccion_1_al_5" in df.columns:
            df["Satisfaccion_1_al_5"] = pd.to_numeric(
                df["Satisfaccion_1_al_5"], errors="coerce"
            ).astype("Int64")
        if "ID_Cliente" in df.columns:
            df["ID_Cliente"] = pd.to_numeric(
                df["ID_Cliente"], errors="coerce"
            ).astype("Int64")

        if "ID_Cliente" in df.columns and not df.empty:
            df = df.sort_values("ID_Cliente").reset_index(drop=True)

        return df[COLUMNS], None
    except Exception as e:
        return pd.DataFrame(columns=COLUMNS), f"Error procesando datos: {e}"


def save_record(record: dict) -> tuple[bool, str | None]:
    """
    Inserta un nuevo registro en Supabase.
    Devuelve (ok, error). Si ok=True, error es None.
    """
    try:
        payload = {TO_DB[k]: record[k] for k in COLUMNS if k in record}
        for k, v in list(payload.items()):
            if pd.isna(v):
                payload[k] = None
        _conn().table("cafeteria").insert(payload).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def next_id(df: pd.DataFrame) -> int:
    """Genera el siguiente ID autoincremental (ID de negocio)."""
    if df.empty or df["ID_Cliente"].dropna().empty:
        return 1
    return int(df["ID_Cliente"].max()) + 1


def validate_record(record: dict) -> list[str]:
    """Devuelve lista de errores. Vacía = registro válido."""
    errors = []

    h = record["Hora_Llegada"]
    if not (7.00 <= h <= 11.00):
        errors.append("Hora_Llegada debe estar entre 7.00 y 11.00.")

    if record["Tiempo_Espera_Min"] <= 0:
        errors.append("Tiempo_Espera_Min debe ser > 0.")

    if record["Tiempo_Servicio_Min"] <= 0:
        errors.append("Tiempo_Servicio_Min debe ser > 0 (revisar error de captura).")

    if record["Tipo_Cliente"] not in ("Estudiante", "Profesor", "Administrativo"):
        errors.append("Tipo_Cliente inválido.")

    if record["Metodo_Pago"] not in (None, "", "Efectivo", "Tarjeta", "App"):
        errors.append("Metodo_Pago inválido.")

    if record["Consumo_Total_MXN"] < 0:
        errors.append("Consumo_Total_MXN debe ser >= 0.")

    sat = record["Satisfaccion_1_al_5"]
    if sat is not None and sat != "" and sat not in (1, 2, 3, 4, 5):
        errors.append("Satisfaccion_1_al_5 debe estar entre 1 y 5.")

    if record["Es_Hora_Pico"] not in ("Sí", "No"):
        errors.append("Es_Hora_Pico debe ser 'Sí' o 'No'.")

    return errors


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina outliers para análisis (servicio <= 0 y consumo > 250)."""
    if df.empty:
        return df.copy()
    clean = df.copy()
    clean = clean[clean["Tiempo_Servicio_Min"] > 0]
    clean = clean[clean["Consumo_Total_MXN"] <= 250]
    return clean.reset_index(drop=True)