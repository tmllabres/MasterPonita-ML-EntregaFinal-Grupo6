# Predicción de cancelación de reservas de hotel

Sistema modular que entrena, evalúa y compara cinco modelos de clasificación binaria
sobre un conjunto de 119.390 reservas de hotel, selecciona el mejor según una métrica
justificada, y automatiza el flujo completo desde el CSV crudo hasta la inferencia.

> **Máster en IA, Cloud Computing y DevOps** · Machine Learning y Deep Learning
> Práctica de evaluación final · 2026-02

<!--
Este README es la documentación completa de la práctica. El guion lo permite:
«¿La documentación es el README.md? Si se incluye toda la información necesaria
definida en el apartado de Entregables obligatorios, sin problema, pero deben estar
todos los puntos recogidos.»

Los diez puntos obligatorios son los diez apartados de abajo. No borres ninguno.
Para la plataforma de PontIA hay que subir un PDF: exporta este fichero.
-->

---

## 1. Autores y roles

| Persona | Correo | De qué se hace cargo |
|---|---|---|
| | | |

<!--
OBLIGATORIO. El guion: «En la documentación habrá que indicar en un apartado los
roles llevados a cabo, quién se hace cargo de qué parte», y «en caso de que no se
haga una distinción específica de los roles […] se asignará la misma nota a ambas».
Si trabajas solo, dilo aquí y explica que lo hablaste con el profesor.
-->

---

## 2. El problema y por qué este dataset

<!--
Qué se predice, para quién, y qué decisión cambia el modelo.
Aquí entra la justificación de negocio: qué le cuesta al hotel cada tipo de error.
-->

- **Objetivo:** predecir si una reserva se cancelará (`is_canceled = 1`) o no (`0`).
- **Tipo de problema:** clasificación binaria supervisada.
- **Filas:** 119.390 · **Columnas del CSV:** 32 · **Predictoras reales:** 29
- **Reparto de clases:** 62,96 % no cancela / 37,04 % cancela (razón 1,70 : 1)

El diccionario de variables está en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

**Fuga de datos detectada y eliminada.** `reservation_status` y `reservation_status_date`
determinan el objetivo al 100 % (`Check-Out` → 0; `Canceled` y `No-Show` → 1): son la
etiqueta escrita con otras palabras. Si se dejan, los cinco modelos sacan un AUC ≈ 1,000
y la comparación no distingue nada. De ahí la cuenta: **32 − `is_canceled` − 2 de fuga = 29**
columnas predictoras.

---

## 3. Análisis exploratorio (EDA)

<!--
OBLIGATORIO. Puntúa por lo que CAMBIÓ, no por el número de gráficos.
Notebook: notebooks/finales/eda_final.ipynb
La tabla de abajo es el formato que más rinde: cada hallazgo, con su decisión al lado.
-->

| Hallazgo | Evidencia | Decisión que tomamos |
|---|---|---|
| | | |

---

## 4. Diseño del sistema

<!-- OBLIGATORIO. -->

```
proyecto/
├── main.py                    orquestador: python main.py
├── src/
│   ├── config.py              parámetros, rutas y semilla
│   ├── data_loader.py         cargar, limpiar, partir, preprocesador
│   ├── model_trainer.py       registro de modelos y bucle comparador
│   ├── evaluator.py           métricas y figuras; único que abre el test
│   └── predictor.py           inferencia con el modelo entrenado
├── notebooks/
│   ├── exploracion/           EDA inicial y prototipos
│   └── finales/               EDA y comparativa presentables
├── data/raw/                  el CSV original, intacto
├── models/                    artefactos entrenados
├── outputs/                   figuras y tablas generadas
└── docs/                      documentación adicional
```

**Decisiones de diseño que hay que poder defender:**

- **Un módulo, una responsabilidad.** Cada uno se describe en una frase sin usar «y».
- **La partición se hace una sola vez**, en `data_loader`. Si cada módulo partiera por
  su cuenta, cada modelo se compararía contra un reparto distinto.
- **Todo lo que aprende un número de los datos va dentro del `Pipeline`** (medianas,
  categorías, medias del escalado), para que se ajuste solo con el train de cada fold.
  Lo que **borra filas** (duplicados, imposibles) va fuera y antes de partir.
- **Registro de modelos con interfaz común**, imitando por dentro a una librería de
  AutoML: añadir un sexto modelo es una clase y una línea, sin tocar el bucle.
- **Semilla única** (`config.SEMILLA = 42`) para partición, modelos y validación cruzada.

---

## 5. Métrica principal y por qué

<!--
OBLIGATORIO justificarlo. Elígela ANTES de entrenar y no la cambies después.
-->

**Métrica principal:** _(F1 de la clase positiva)_ · **Secundarias:** accuracy, precision, recall, ROC-AUC.

Con un reparto 62,96 / 37,04, un modelo que dijera siempre «no cancela» ya acierta el
**62,96 %** sin haber aprendido nada: por eso accuracy no sirve como criterio.

<!--
Ojo, «F1 equilibra precision y recall» NO es una justificación: di qué error te duele
más en este negocio y por qué el equilibrio es lo razonable aquí.
-->

---

## 6. Cómo ejecutar el proyecto

<!-- OBLIGATORIO: versión de Python, entorno virtual y pasos exactos. -->

**Requisitos:** Python 3.12

```bash
git clone <URL-del-repositorio>
cd Entrega-Final-ML

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install -r requirements.txt
```

<details>
<summary>Alternativa rápida con <code>uv</code> (opcional)</summary>

Si tienes [uv](https://docs.astral.sh/uv/) instalado, lee el mismo `requirements.txt`
y tarda bastante menos —importa, porque TensorFlow son varios cientos de MB:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

No es necesario: los pasos de arriba con `pip` funcionan igual.
</details>

**Pipeline completo** (carga → limpieza → entrenamiento → comparación → evaluación → artefacto):

```bash
python main.py
```

**Versión corta para la defensa** (muestra reducida y menos folds):

```bash
python main.py --demo
```

**Inferencia con el modelo ya entrenado, sin reentrenar nada:**

```bash
python -m src.predictor
```

Salidas: figuras y tablas en `outputs/`, modelo en `models/`.

---

## 7. Modelos comparados

Los cinco que exige el enunciado, entrenados con el mismo protocolo: mismo
`StratifiedKFold`, misma semilla, preprocesado dentro del `Pipeline` y
`scoring` fijado a la métrica principal.

| Modelo | F1 (CV) | Accuracy | Precision | Recall | ROC-AUC | Tiempo |
|---|---|---|---|---|---|---|
| Baseline (`DummyClassifier`) | | | | | | |
| Regresión logística | | | | | | |
| Árbol de decisión | | | | | | |
| Random Forest | | | | | | |
| Gradient Boosting | | | | | | |
| Red neuronal (Keras) | | | | | | |

<!--
La fila del baseline no es decorativa: sin ella, un número suelto no dice si tus
modelos aportan algo por encima de lo trivial.
Las métricas de CV son media ± desviación sobre el train. La del test se mide UNA vez.
-->

---

## 8. Resultados y elección final

<!-- OBLIGATORIO. -->

**Modelo elegido:** _(…)_ · **Por qué gana:** _(…)_

Métricas del ganador sobre el conjunto de test (23.878 reservas nunca vistas):

| | valor |
|---|---|
| F1 | |
| Accuracy | |
| Precision | |
| Recall | |
| ROC-AUC | |

**Figuras obligatorias:**

- Matriz de confusión → `outputs/confusion_matrix.png`
- Curva ROC de los cinco modelos en los mismos ejes → `outputs/roc_curve.png`
- Importancia de variables del ganador → `outputs/feature_importance.png`

---

## 9. Conclusiones

<!-- OBLIGATORIO en el README. Qué se aprendió, qué se puede hacer con esto. -->

---

## 10. Reflexión crítica: limitaciones y mejoras

<!--
OBLIGATORIO, y es donde más gente deja puntos. Reconocer una limitación tú vale
bastante más que si la detecta el profesor.
Candidatas honestas:
  · la validación cruzada no es anidada: los hiperparámetros se eligieron mirando
    los mismos folds con los que se mide
  · la partición es aleatoria, no temporal: en producción predices el futuro
  · el dataset no trae el coste en euros de cada error, así que el umbral se
    elige por F1 y no por dinero
  · el modelo mide correlación, no causa
-->

---

## Anexos

- [Diccionario de variables](docs/diccionario_datos.md)
- Notebooks: [`notebooks/finales/`](notebooks/finales/)
