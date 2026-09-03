"""Tests del registro de modelos.

El registro es la pieza que imita a una librería de AutoML, así que lo que se
prueba es el CONTRATO: que las seis clases son intercambiables. Si una se sale
del contrato, el bucle comparador falla con un mensaje que no dice nada.

    python -m pytest tests/test_model_trainer.py -q
"""
from __future__ import annotations

import pytest
from sklearn.base import clone

from src import config, model_trainer

pytestmark = pytest.mark.skip(reason="TODO: quitar cuando model_trainer esté implementado")


def test_todos_los_modelos_activos_estan_en_el_registro():
    """config.MODELOS_ACTIVOS y REGISTRO no pueden divergir sin que alguien se entere."""
    for nombre in config.MODELOS_ACTIVOS:
        assert nombre in model_trainer.REGISTRO


@pytest.mark.parametrize("nombre", config.MODELOS_ACTIVOS)
def test_cada_modelo_cumple_la_interfaz(nombre):
    """Mismo contrato para los seis: construir(), fit, predict y predict_proba."""
    modelo = model_trainer.REGISTRO[nombre]()
    assert modelo.nombre == nombre
    for metodo in ("construir", "fit", "predict", "predict_proba", "espacio_busqueda"):
        assert callable(getattr(modelo, metodo))


@pytest.mark.parametrize("nombre", config.MODELOS_ACTIVOS)
def test_clone_funciona(nombre):
    """clone() es lo que usa la validación cruzada por dentro para dar a cada fold
    un modelo virgen. Falla si __init__ hace algo más que guardar hiperparámetros:
    justo la trampa de construir la red de Keras en __init__ en vez de en fit."""
    modelo = model_trainer.REGISTRO[nombre]()
    assert clone(modelo) is not modelo


def test_el_espacio_de_busqueda_usa_la_sintaxis_de_pipeline():
    """Las claves van como paso__hiperparametro; sin el prefijo, GridSearchCV no
    encuentra el tornillo y revienta en tiempo de ejecución, no al escribirlo."""
    for nombre in config.MODELOS_ACTIVOS:
        for clave in model_trainer.REGISTRO[nombre]().espacio_busqueda():
            assert "__" in clave, f"{nombre}: la clave «{clave}» no lleva prefijo de paso"
