"""Orquestador del pipeline completo: de los datos a la inferencia.

    python main.py

Este fichero no calcula nada. Solo llama a los módulos de src/ en orden, y por eso
se lee de un vistazo: la prueba del algodón de una arquitectura modular es que
alguien pueda leer main.py y entender el proceso entero sin abrir nada más.

Es lo que se ejecuta delante del profesor en la defensa.
"""
import argparse

from src import config, data_loader, evaluator, model_trainer


def main(demo: bool = False) -> None:
    config.DEMO = demo or config.DEMO

    # 1. Datos: cargar, limpiar, partir, y preparar el preprocesador sin ajustar.
    print("[1/5] Cargando y preparando datos…")
    d = data_loader.preparar()

    # 2. Entrenar los cinco con el mismo protocolo y compararlos por validación
    #    cruzada dentro del train. El test no se toca aquí.
    print("[2/5] Entrenando y comparando modelos…")
    tabla = model_trainer.entrenar_y_comparar(d["X_train"], d["y_train"], d["preprocesador"])
    print(tabla)

    # 3. Elegir el ganador por la métrica principal y reentrenarlo con todo el train.
    print("[3/5] Eligiendo el mejor modelo…")
    ganador = model_trainer.elegir_mejor(tabla)
    pipeline = model_trainer.reentrenar_ganador(ganador, d["X_train"], d["y_train"],
                                                d["preprocesador"])

    # 4. Evaluar UNA sola vez sobre el test, y generar las figuras obligatorias.
    print(f"[4/5] Evaluando «{ganador}» sobre el test…")
    y_pred = pipeline.predict(d["X_test"])
    y_proba = pipeline.predict_proba(d["X_test"])[:, 1]
    m = evaluator.metricas(d["y_test"], y_pred, y_proba)
    evaluator.matriz_confusion(d["y_test"], y_pred)
    evaluator.curva_roc({ganador: y_proba}, d["y_test"])
    evaluator.importancias(pipeline, d["X_test"], d["y_test"])
    evaluator.informe(tabla, m)
    print(m)

    # 5. Persistir el artefacto para que predictor.py pueda usarlo sin reentrenar.
    print("[5/5] Guardando el modelo…")
    model_trainer.guardar(pipeline)
    print("Listo. Figuras en outputs/, modelo en models/.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline de cancelación de reservas")
    p.add_argument("--demo", action="store_true",
                   help="muestra reducida y menos folds, para la defensa")
    main(**vars(p.parse_args()))
