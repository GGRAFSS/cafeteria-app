"""
Genera 200 registros simulados según el diccionario de datos de la cafetería universitaria
y los inserta en la tabla cafeteria de Supabase.
"""
import os
import random

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Faltan las variables de entorno SUPABASE_URL y SUPABASE_KEY. "
        "Configúralas en un archivo .env"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Parámetros ----------
random.seed(42)  # reproducibilidad
N = 200
tipos = ["Estudiante", "Profesor", "Administrativo"]
pesos_tipo = [0.7, 0.2, 0.1]
pagos = ["Efectivo", "Tarjeta", "App", None]
pesos_pago = [0.30, 0.35, 0.25, 0.10]

def generar_hora_llegada():
    return round(random.triangular(7.0, 11.0, 8.5), 2)

def generar_espera():
    val = random.expovariate(1/4)  # exponencial, media 4 min
    return round(min(max(val, 0.5), 25.0), 1)

def generar_servicio():
    val = random.gauss(4, 1.5)
    return round(max(val, 0.5), 1)

def generar_consumo(tipo):
    base = {"Estudiante": 45, "Profesor": 70, "Administrativo": 55}[tipo]
    return round(max(random.gauss(base, 15), 5.0), 1)

def generar_satisfaccion():
    if random.random() < 0.05:
        return None
    return random.choices([1,2,3,4,5], weights=[0.05,0.10,0.20,0.35,0.30])[0]

registros = []
for i in range(1, N+1):
    tipo = random.choices(tipos, weights=pesos_tipo)[0]
    hora = generar_hora_llegada()
    registros.append({
        "id_cliente": i,
        "hora_llegada": hora,
        "tiempo_espera_min": generar_espera(),
        "tiempo_servicio_min": generar_servicio(),
        "tipo_cliente": tipo,
        "metodo_pago": random.choices(pagos, weights=pesos_pago)[0],
        "consumo_total_mxn": generar_consumo(tipo),
        "satisfaccion_1_al_5": generar_satisfaccion(),
        "es_hora_pico": "Sí" if 8.0 <= hora <= 9.0 else "No",
    })

# Inyectar outlier de consumo (300.0). El -2.0 en servicio se omite
# porque la BD lo rechaza correctamente por la restricción > 0.
registros[120]["consumo_total_mxn"] = 300.0

# Insertar en lotes de 50
print(f"Insertando {len(registros)} registros en Supabase...")
for i in range(0, len(registros), 50):
    batch = registros[i:i+50]
    supabase.table("cafeteria").insert(batch).execute()
    print(f"  Lote {i//50 + 1}: insertados {len(batch)}")

print("✅ Carga completada. Verifica en Table Editor.")