"""Entrenamiento y comparación de los modelos.

Aquí está la idea que pide el enunciado: imitar por dentro a una librería de AutoML.
Cada modelo es una CLASE con la misma interfaz, viven todas en un REGISTRO, y un único
bucle las recorre sin un solo `if`. Añadir un sexto modelo = escribir su clase y una
línea en el registro; el bucle no cambia ni una letra.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin

from . import config


# ── El contrato ──────────────────────────────────────────────────────────────
class ModeloBase(BaseEstimator, ClassifierMixin):
    """Interfaz común. Todo modelo del registro cumple esto.

    Reglas de scikit-learn que hay que respetar para que clone(), GridSearchCV y
    cross_val_score funcionen:
      · __init__ SOLO guarda hiperparámetros, uno por argumento, sin validar ni
        construir nada. Lo que se aprende no se toca aquí.
      · lo aprendido se guarda en atributos con GUION BAJO FINAL: self.model_
      · fit devuelve self
    """

    nombre = "base"

    def construir(self):
        """Devuelve el estimador o Pipeline sin entrenar. Lo implementa cada hijo."""
        raise NotImplementedError

    def espacio_busqueda(self) -> dict:
        """Rejilla de hiperparámetros con la sintaxis paso__hiperparametro.

        Ej.: {"modelo__max_depth": [4, 6, 8]}. Los dobles guiones bajos son el
        camino hasta el tornillo: cada tramo es la etiqueta que le pusiste al paso.
        """
        return {}

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


# ── La linea del suelo ───────────────────────────────────────────────────────
class Baseline(ModeloBase):
    """DummyClassifier: la referencia contra la que se miden los cinco de verdad.

    No aprende nada, y ese es justo el punto: con strategy="most_frequent" dice
    siempre "no cancela" y ya acierta el 62,96 %. Cualquier modelo que no supere
    claramente esta fila no esta aportando nada, y sin la fila no hay forma de saberlo.

    Ojo con la metrica: en accuracy saca 0,63, pero en F1 de la clase positiva saca
    0,00, porque nunca predice un 1. Las dos cosas dicen lo mismo desde dos sitios.
    """
    nombre = "baseline"


# ── Los cinco obligatorios ───────────────────────────────────────────────────
class Logistica(ModeloBase):
    """Regresión logística. Necesita escalado. C es la inversa de la regularización."""
    nombre = "logistica"


class Arbol(ModeloBase):
    """Árbol de decisión. No necesita escalado. Vigila max_depth y min_samples_leaf."""
    nombre = "arbol"


class Bosque(ModeloBase):
    """Random Forest. Bagging: cada árbol ve ~63,2 % de filas distintas y un sorteo
    de columnas (max_features). Trae feature_importances_ ya calculado."""
    nombre = "bosque"


class Boosting(ModeloBase):
    """Gradient Boosting. learning_rate y n_estimators son un solo mando: si bajas
    uno, sube el otro. Sus árboles son de REGRESIÓN: cada hoja guarda una corrección."""
    nombre = "boosting"


class RedKeras(ModeloBase):
    """MLP con Keras envuelto en la interfaz de scikit-learn.

    Aquí está la trampa del proyecto: la red se construye en FIT, nunca en __init__.
    Si se construye en __init__, clone() reparte el mismo objeto de Keras a los 5
    folds, y como el fit de Keras no reinicia los pesos, el fold 2 arranca habiendo
    visto ya sus datos de validación. El F1 sale inflado y no salta ningún error.
    """
    nombre = "red_keras"


# ── El registro ──────────────────────────────────────────────────────────────
REGISTRO = {
    "baseline": Baseline,
    "logistica": Logistica,
    "arbol": Arbol,
    "bosque": Bosque,
    "boosting": Boosting,
    "red_keras": RedKeras,
}


# ── El comparador ────────────────────────────────────────────────────────────
def entrenar_y_comparar(X_train, y_train, preprocesador) -> tuple:
    """Recorre config.MODELOS_ACTIVOS y devuelve (tabla, modelos).

    Protocolo idéntico para los seis, que es lo que hace justa la comparación:
      · el mismo StratifiedKFold(config.CV_FOLDS) con la misma semilla
      · el preprocesado DENTRO del Pipeline, para que se ajuste en cada fold
      · scoring=config.METRICA_PRINCIPAL — si no lo pones, GridSearchCV optimiza
        accuracy sin decírtelo, y luego presentas F1 en el informe
      · media ± desviación de la validación cruzada, nunca una sola partición

    Devuelve DOS cosas:
      · tabla:   DataFrame ordenado por la métrica principal, una fila por modelo.
      · modelos: dict {nombre: Pipeline ajustado con TODO el train}.

    Por qué también los pipelines, y no solo la tabla: el enunciado pide la curva
    ROC de los cinco modelos EN LOS MISMOS EJES. Para dibujar cinco curvas hacen
    falta cinco `predict_proba` sobre el test, o sea los cinco modelos ajustados.
    Si aquí solo saliera la tabla, main.py se quedaría con el ganador y la figura
    tendría una sola línea. El coste es un refit por modelo sobre el train
    completo, después de la validación cruzada.
    """
    raise NotImplementedError("TODO")


def elegir_mejor(tabla):
    """Devuelve el nombre del modelo ganador según config.METRICA_PRINCIPAL."""
    raise NotImplementedError("TODO")


def guardar(pipeline, nombre, metricas, ruta=None):
    """Persiste el Pipeline COMPLETO, no solo el estimador, más sus metadatos.

    Si guardas solo el modelo, el preprocesado se pierde y las predicciones salen
    mal sin dar error. Keras es la excepción: config.MODELO_KERAS aparte (pesos y
    arquitectura) y el resto del Pipeline en config.MODELO_PKL.

    Escribe además config.METADATOS con lo que predictor.py necesita para no tener
    que adivinar nada:
      · nombre del modelo ganador
      · umbral realmente usado al predecir (no el que hoy tenga config.UMBRAL:
        el que se congeló al guardar; si luego mueves config.UMBRAL, las
        predicciones de un modelo ya guardado no pueden cambiar en silencio)
      · las columnas crudas que espera de entrada, en orden
      · métrica principal, semilla y versiones de las librerías
    """
    raise NotImplementedError("TODO")
