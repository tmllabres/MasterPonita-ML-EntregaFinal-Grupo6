"""Predicciones con el modelo ya entrenado: el camino de vuelta.

Esto es la INFERENCIA, y es lo que cierra el «flujo completo desde los datos hasta
la inferencia» que pide el enunciado. Este módulo NO entrena, NO evalúa y NO abre
el CSV de train: carga el artefacto y predice.

La prueba de que el artefacto es usable: alguien que solo tenga models/ y este
fichero puede predecir una reserva nueva sin reentrenar nada.
"""
from __future__ import annotations

from . import config


def cargar(ruta=None):
    """Carga el Pipeline entrenado (preprocesado + modelo) y sus metadatos.

    Si el ganador es la red de Keras son DOS ficheros: el .keras con arquitectura
    y pesos, y el .pkl con el ColumnTransformer ya ajustado.
    """
    raise NotImplementedError("TODO")


def predecir(pipeline, reservas):
    """Devuelve 0/1 por reserva aplicando el umbral de los metadatos.

    Las reservas entran con las MISMAS columnas crudas que el train: el Pipeline
    se encarga del resto. Si tienes que preparar los datos a mano antes de llamar
    aquí, es que el preprocesado se quedó fuera del artefacto.
    """
    raise NotImplementedError("TODO")


def predecir_proba(pipeline, reservas):
    """Devuelve la probabilidad de cancelación, que es lo que el modelo calcula
    de verdad. El 0/1 sale luego de comparar contra el umbral."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    # Demo de inferencia: python -m src.predictor
    # Coge unas cuantas reservas del CSV, quítales la respuesta y predícelas.
    raise SystemExit("TODO: demo de inferencia")
