"""Código fuente del sistema de comparación de modelos de cancelación de reservas.

    src/
    ├── config.py         parámetros y rutas: todo número que se pueda cambiar
    ├── data_loader.py    cargar, limpiar, partir y construir el preprocesador
    ├── model_trainer.py  el registro de modelos y el bucle que los compara
    ├── evaluator.py      métricas y figuras; el único que abre el test
    └── predictor.py      inferencia con el modelo ya entrenado

El orquestador es main.py, en la raíz.
"""
