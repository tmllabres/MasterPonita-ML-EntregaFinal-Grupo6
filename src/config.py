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

# Los artefactos, con nombre y apellidos. Sin esto, model_trainer.guardar() y
# predictor.cargar() escriben y leen "ruta=None" y el nombre del fichero acaba
# escrito a mano en dos sitios distintos; el día que uno cambie, el otro falla.
MODELO_PKL = MODELS / "mejor_modelo.pkl"          # el Pipeline completo
MODELO_KERAS = MODELS / "mejor_modelo.keras"      # solo si gana la red
METADATOS = MODELS / "metadatos.json"             # ganador, umbral, columnas, versiones

# Lo que escribe evaluator, con nombre fijo porque el README los enlaza.
FIG_CONFUSION = OUTPUTS / "confusion_matrix.png"
FIG_ROC = OUTPUTS / "roc_curve.png"
FIG_IMPORTANCIAS = OUTPUTS / "feature_importance.png"
TABLA_COMPARATIVA = OUTPUTS / "tabla_comparativa.csv"
METRICAS_TEST = OUTPUTS / "metricas_test.json"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
# Una sola semilla para todo: partición, modelos, validación cruzada.
SEMILLA = 42

# ── El problema ──────────────────────────────────────────────────────────────
OBJETIVO = "is_canceled"

# Columnas que determinan el objetivo al 100 %: son fuga y se eliminan siempre.
# reservation_status: Check-Out -> 0 ; Canceled y No-Show -> 1.
FUGAS = ["reservation_status", "reservation_status_date"]

# ── Partición ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
ESTRATIFICAR = True       # conserva el mismo reparto de clases en las dos mitades

# CUIDADO con las cuentas de filas. Sobre el CSV crudo son 119.390 -> 95.512 / 23.878,
# pero eso es ANTES de limpiar. Con la limpieza que declara data_loader.limpiar()
# (fugas + duplicados exactos + imposibles) quedan 86.971 filas -> 69.577 / 17.394,
# y el reparto pasa de 62,96/37,04 a 72,69/27,31. Los duplicados exactos son 32.252
# filas, el 27 % del dataset, y el 63,4 % de ellas son cancelaciones: por eso al
# quitarlas la clase positiva se hunde diez puntos.
#
# DECISIÓN PENDIENTE, y hay que tomarla antes de entrenar: ¿son esos 32.252 un error
# de registro o reservas legítimamente idénticas (misma noche, mismo precio, mismo
# hotel, grupos)? Tirarlas cambia la prevalencia diez puntos y con ella la
# justificación de la métrica principal. Se decide, se justifica en el README y se
# actualizan los números de los apartados 2, 5 y 8.

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
#
# "baseline" no lo pide el enunciado, pero sin él un F1 de 0,80 no dice si el modelo
# aprendió algo o si el problema era fácil. Es la fila que da sentido a las otras cinco.
MODELOS_ACTIVOS = [
    "baseline",
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
