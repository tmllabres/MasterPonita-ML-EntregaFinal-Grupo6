"""Carga y transformación de datos.

Este módulo hace tres cosas y ninguna más:
  1. leer el CSV crudo,
  2. limpiar lo que BORRA FILAS (duplicados, imposibles, fugas) — fuera del Pipeline,
  3. partir en train/test y construir el ColumnTransformer SIN ajustar.

Por qué la limpieza vive aquí y no en el Pipeline: un transformador de scikit-learn
devuelve tantas filas como recibe. Si tirara filas, la `y` se quedaría descolocada
respecto a la `X` y nadie te avisaría. Todo lo que borra filas va antes de partir;
todo lo que APRENDE un número de los datos (medianas, categorías, medias del escalado)
va dentro del ColumnTransformer, para que lo aprenda solo con el train.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from . import config


def cargar_crudo(ruta=None) -> pd.DataFrame:
    """Lee el CSV tal cual viene, sin tocar nada.

    Ojo con los nulos: `company` y `agent` traen el texto "NULL", que pandas no
    reconoce como ausente salvo que se lo digas con na_values.
    """
    raise NotImplementedError("TODO")


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita fugas, duplicados e imposibles. Devuelve MENOS filas de las que recibe.

    Orden que importa:
      1. eliminar config.FUGAS  (determinan el objetivo al 100 %)
      2. eliminar duplicados exactos
      3. eliminar imposibles: adr negativo, reservas con 0 huéspedes
    Deja registrado cuántas filas caen en cada paso: eso va al README.
    """
    raise NotImplementedError("TODO")


def separar_X_y(df: pd.DataFrame):
    """Devuelve (X, y). X son las 29 columnas predictoras; y es is_canceled.

    La cuenta: 32 columnas del CSV − is_canceled − las 2 de fuga = 29.
    """
    raise NotImplementedError("TODO")


def particionar(X, y):
    """train_test_split con test_size, semilla y stratify de config.

    Se llama UNA vez y desde aquí. Si cada módulo partiera por su cuenta, cada
    modelo se compararía contra un reparto distinto y la tabla no valdría nada.
    """
    raise NotImplementedError("TODO")


def construir_preprocesador(X_train) -> ColumnTransformer:
    """Devuelve el ColumnTransformer SIN ajustar.

    Sin ajustar a propósito: lo ajusta el Pipeline dentro de cada fit, solo con
    el train de ese fold. Eso es lo que impide la fuga de preprocesado.

    Ramas:
      - numéricas    -> imputar (mediana) + escalar
      - categóricas  -> imputar (constante) + one-hot con handle_unknown="ignore"
      - alta cardinalidad (country, agent, company) -> decide y justifica:
        agrupar en top-N + "otros", frecuencia, o target encoding dentro del CV.
    """
    raise NotImplementedError("TODO")


def preparar() -> dict:
    """Punto de entrada del módulo: del CSV a todo lo que necesita el resto.

    Devuelve un dict con X_train, X_test, y_train, y_test y el preprocesador.
    """
    raise NotImplementedError("TODO")
