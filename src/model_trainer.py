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
    "logistica": Logistica,
    "arbol": Arbol,
    "bosque": Bosque,
    "boosting": Boosting,
    "red_keras": RedKeras,
}


# ── El comparador ────────────────────────────────────────────────────────────
def entrenar_y_comparar(X_train, y_train, preprocesador) -> "pd.DataFrame":
    """Recorre config.MODELOS_ACTIVOS y devuelve la tabla comparativa.

    Protocolo idéntico para los cinco, que es lo que hace justa la comparación:
      · el mismo StratifiedKFold(config.CV_FOLDS) con la misma semilla
      · el preprocesado DENTRO del Pipeline, para que se ajuste en cada fold
      · scoring=config.METRICA_PRINCIPAL — si no lo pones, GridSearchCV optimiza
        accuracy sin decírtelo, y luego presentas F1 en el informe
      · media ± desviación de la validación cruzada, nunca una sola partición

    Devuelve un DataFrame ordenado por la métrica principal.
    """
    raise NotImplementedError("TODO")


def elegir_mejor(tabla):
    """Devuelve el nombre del modelo ganador según config.METRICA_PRINCIPAL."""
    raise NotImplementedError("TODO")


def reentrenar_ganador(nombre, X_train, y_train, preprocesador):
    """Reentrena el ganador con TODO el train y devuelve el Pipeline ajustado."""
    raise NotImplementedError("TODO")


def guardar(pipeline, ruta=None):
    """Persiste el Pipeline COMPLETO, no solo el estimador.

    Si guardas solo el modelo, el preprocesado se pierde y las predicciones salen
    mal sin dar error. Keras es la excepción: .keras aparte + el resto en .pkl.
    """
    raise NotImplementedError("TODO")
