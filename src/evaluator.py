"""Métricas y visualización de resultados.

Es el ÚNICO módulo que abre el conjunto de test, y lo abre una sola vez, al final.
Cada vez que el test influye en una decisión —qué modelo, qué hiperparámetro, qué
umbral— deja de ser una estimación honesta.
"""
from __future__ import annotations

from . import config


def metricas(y_true, y_pred, y_proba=None) -> dict:
    """Devuelve la métrica principal y las secundarias en un dict.

    Con pos_label=1 ("cancela"): es la clase que quieres detectar, no la que te
    conviene. Define qué es un TP y hacia dónde miran precision y recall.
    """
    raise NotImplementedError("TODO")


def matriz_confusion(y_true, y_pred, ruta=None):
    """Figura obligatoria del enunciado. Guarda en outputs/confusion_matrix.png.

    Pon los números absolutos y el porcentaje por fila: el de la diagonal de TP es
    el recall y el de TN la specificity, y así se lee sola.
    """
    raise NotImplementedError("TODO")


def curva_roc(modelos_proba: dict, y_true, ruta=None):
    """Figura obligatoria del enunciado. Guarda en outputs/roc_curve.png.

    LAS CINCO CURVAS EN LOS MISMOS EJES, con su AUC en la leyenda y la diagonal
    del azar. Cinco gráficos sueltos no demuestran nada.
    """
    raise NotImplementedError("TODO")


def curva_pr(modelos_proba: dict, y_true, ruta=None):
    """Precision-Recall. No es obligatoria, pero con clases desbalanceadas dice
    más que la ROC, porque no usa los TN. Su suelo de azar es la prevalencia (0,370)."""
    raise NotImplementedError("TODO")


def barrido_umbral(y_true, y_proba):
    """Tabla umbral -> precision, recall, F1. Para justificar el umbral elegido.

    Se decide sobre VALIDACIÓN, nunca sobre test.
    """
    raise NotImplementedError("TODO")


def importancias(pipeline, X, y, ruta=None):
    """Gráfico de importancia de variables del ganador.

    Ojo: feature_importances_ del bosque favorece a las columnas con muchos valores
    distintos (country). permutation_importance es más honesta y funciona con
    cualquier modelo. Di cuál usaste.
    """
    raise NotImplementedError("TODO")


def informe(tabla_comparativa, metricas_test: dict, ruta=None):
    """Vuelca la tabla comparativa y las métricas de test a outputs/, listas para
    pegarlas en el README."""
    raise NotImplementedError("TODO")
