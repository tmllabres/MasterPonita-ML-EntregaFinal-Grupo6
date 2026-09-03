"""Parámetros y configuración del proyecto.

Todo lo que sea un número, una ruta o una decisión vive aquí y en ningún otro sitio.
La regla: si vas a cambiarlo en la defensa para enseñar algo, tiene que estar en este fichero.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw" / "dataset_practica_final.csv"
DATA_PROC = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
OUTPUTS = RAIZ / "outputs"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
# Una sola semilla para todo: partición, modelos, validación cruzada.
SEMILLA = 42

# ── El problema ──────────────────────────────────────────────────────────────
OBJETIVO = "is_canceled"

# Columnas que determinan el objetivo al 100 %: son fuga y se eliminan siempre.
# reservation_status: Check-Out -> 0 ; Canceled y No-Show -> 1.
FUGAS = ["reservation_status", "reservation_status_date"]

# ── Partición ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20          # 119.390 -> train 95.512 / test 23.878
ESTRATIFICAR = True       # conserva el 62,96 / 37,04 en las dos mitades

# ── Evaluación ───────────────────────────────────────────────────────────────
# Métrica principal declarada ANTES de entrenar. Se justifica en el README.
METRICA_PRINCIPAL = "f1"
METRICAS_SECUNDARIAS = ["accuracy", "precision", "recall", "roc_auc"]

CV_FOLDS = 5              # StratifiedKFold dentro del train
UMBRAL = 0.50             # el de la librería; si lo mueves, dilo y justifícalo

# ── Ajuste de hiperparámetros ────────────────────────────────────────────────
BUSQUEDA = "random"       # "grid" | "random" | None
N_ITER_RANDOM = 30        # solo si BUSQUEDA == "random"
N_JOBS = -1

# ── Qué modelos entran en la comparación ─────────────────────────────────────
# El bucle de model_trainer recorre esta lista. Añadir un modelo = añadir una línea.
MODELOS_ACTIVOS = [
    "logistica",
    "arbol",
    "bosque",
    "boosting",
    "red_keras",
]

# ── Modo demo ────────────────────────────────────────────────────────────────
# La defensa son 30 minutos. Con esto activado el pipeline entrena sobre una
# muestra y con menos folds, para enseñar el circuito completo sin esperar.
DEMO = False
DEMO_FILAS = 20_000
DEMO_FOLDS = 3
