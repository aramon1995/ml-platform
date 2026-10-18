# ML Platform

Plataforma local de ML: entrenamiento con búsqueda de hiperparámetros, selección y promoción de un modelo a producción, e inferencia batch con las predicciones guardadas en Postgres.

| Pieza | Herramienta |
| --- | --- |
| Orquestación | Prefect |
| Tracking y registro de modelos | MLflow |
| Búsqueda de hiperparámetros | Optuna |
| Almacenamiento de datos y artefactos | S3 (RustFS) |
| Predicciones, metadatos de MLflow, Optuna y Prefect | Postgres |

Todo se levanta con docker compose.

## Inicio rápido

Requisitos: Docker con Compose v2. Para el notebook y los tests, también [uv](https://docs.astral.sh/uv/).

```bash
make start            # levanta los servicios y sube data.csv a s3://data/raw/
make train-select     # entrena los 5 modelos (30 trials) y promueve el mejor (~10 min)
make infer            # predice con el modelo de producción y guarda en Postgres
```

| Servicio | URL | Credenciales |
| --- | --- | --- |
| Prefect (lanzar y seguir flows) | <http://localhost:4200> | |
| MLflow (experimentos y modelos) | <http://localhost:5000> | |
| Consola S3 (datos y artefactos) | <http://localhost:9101/rustfs/console> | `s3admin` / `s3admin12345` |
| Postgres | `localhost:5432` | `mlp` / `mlp`; bases `app`, `mlflow`, `optuna`, `prefect` |

## Comandos

```bash
make start                               # servicios + dataset en S3 (idempotente)
make seed                                # vuelve a subir data.csv a s3://data/raw/data.csv
make train MODELS=svm,logreg TRIALS=10   # estudios de Optuna; sin MODELS entrena todos, sin TRIALS usa 30
make select                              # champion por modelo y modelo de producción
make train-select                        # train + select en un paso
make infer                               # lote por defecto: el split de test
make infer INPUT=s3://data/inference/lote.csv
make eda                                 # re-ejecuta lab/eda.ipynb (desde el host)
make test                                # tests unitarios
make stop                                # para los servicios
make clean                               # para y borra los volúmenes (datos, runs, modelos)
```

Los comandos `train`, `select` e `infer` lanzan el deployment de Prefect y esperan a que termine. También se pueden lanzar desde la UI de Prefect (*Deployments*). El reentrenamiento diario (`train-and-select`, 03:00) está definido pero desactivado: se activa desde la UI.

Para subir un lote de inferencia: `uv run python -m mlplatform.storage upload lote.csv s3://data/inference/lote.csv`. El lote tiene las mismas columnas que `data.csv`; `Target` se ignora si está.

## Flujo

```plain
lab/eda.ipynb          EDA sobre train → decisiones de features
features.py            features elegidas y preprocesado por familia de modelo
models/<modelo>.py     pipeline (preprocesado + estimador) + espacio de búsqueda
tuning.py              Optuna con split y folds fijos → MLflow
selection.py           champion por modelo → producción (regla 1-SE + complejidad)
inference.py           lote de S3 → classifier@production → Postgres
flows.py               flows de Prefect sobre las piezas anteriores
```

El núcleo es Python plano y no depende de Prefect: cada paso se puede ejecutar con `python -m mlplatform.tuning svm`, `python -m mlplatform.selection` o `python -m mlplatform.inference`.

### 1. EDA

`lab/eda.ipynb` lee el dataset de S3 y analiza **solo el split de train**, para no tomar decisiones viendo el test. Cubre calidad de datos, balance, columnas planas, relación con el target (mutual information, V de Cramér corregida), correlaciones, cardinalidad, features de fecha y una ablación con los folds del proyecto.

Figuras, tablas y conclusiones se registran en MLflow (experimento `eda`). La sección 9 del notebook recoge las decisiones que implementa `features.py`.

### 2. Features

Resultado del analisis exploratorio, si detectamos data skews o decremento del modelo de produccion con el tiempo, debemos rehacer el analisis y sacar nuevas conclusiones si es necesario y aplicarlos en `features.py` para el entrenamiento.

Solo sobreviven tres entradas:

| Feature | Tratamiento | Por qué |
| --- | --- | --- |
| `month` (de `Timestamp`) | One-hot; nulo → categoría `missing` | Determina casi por completo el target (V de Cramér 0,95). One-hot para no asumir orden entre meses |
| `F22` | Imputación por mediana | Única numérica con señal fuerte (MI 0,76) |
| `F17` | Mediana + indicador de nulo | 90 % nulos; el valor informa cuando está |

Hay dos preprocesadores: `linear_preprocessor()` escala las numéricas (logreg, SVM, MLP) y `tree_preprocessor()` no (random forest, XGBoost).

### 3. Modelos

Cada modelo es un módulo en `src/mlplatform/models/` con el mismo contrato:

```python
NAME = "svm"
COMPLEXITY = 2                            # orden de simplicidad y explicabilidad

def suggest(trial) -> dict: ...           # espacio de búsqueda de Optuna
def build_pipeline(params) -> Pipeline: ... # preprocesado de su familia + estimador
```

| Modelo | Complejidad | Notas |
| --- | --- | --- |
| `logreg` | 1 | |
| `svm` | 2 | Kernel RBF |
| `random_forest` | 3 | |
| `xgboost` | 3 | Envoltorio que codifica las etiquetas |
| `mlp` | 4 | Red PyTorch propia: capas, activación y dropout se buscan; early stopping y pérdida por época |

**Añadir un modelo**: crear `models/<nombre>.py` con ese contrato y añadirlo al diccionario `MODELS` de `models/__init__.py`.

### 4. Tuning

`tuning.run_study(modelo)` es igual para todos los modelos:

1. Lee el dataset de S3 y genera el split (80/20 estratificado) y los folds (5×2 estratificados repetidos). Son deterministas: todos los modelos usan exactamente los mismos índices.
2. Optuna (TPE) maximiza el F1 macro medio en CV sobre train.
3. **Re-evaluación**: los `RESCORE_TOP_K` (5) mejores trials se vuelven a puntuar con una partición distinta de los mismos datos de train (otra seed), que no participó en la búsqueda. Gana el de mejor re-evaluación, y su media es el `cv_mean` que usa la selección. Así la medida no arrastra la suerte de los folds con los que se eligió el trial.
4. El ganador se reentrena con todo train y se evalúa **una vez** en test.

### 5. Selección

1. **Champion por modelo**: su último estudio, registrado como `<modelo>@champion`.
2. **Producción**, regla de 1 error estándar: se toma el champion con mejor CV y se consideran empatados todos los que estén a menos de 1 SE de él. De ellos gana el de menor complejidad, y luego el de mayor CV.
3. **Control en test**: se descarta un candidato que quede más de `TEST_TOLERANCE` (0,05) por debajo del mejor test. El test no elige, solo controla: con 100 filas, un error equivale a un punto.
4. El ganador se registra como `classifier@production`. Volver a ejecutar la selección no crea versiones si nada cambió.

### 6. Inferencia

`inference.batch_inference(input_uri)` resuelve `classifier@production`, carga esa versión desde S3 en memoria, predice y guarda una fila por predicción:

| Columna | Contenido |
| --- | --- |
| `prediction_id`, `timestamp`, `predicted_value` | Lo que pide el reto |
| `probabilities` | Probabilidad por clase (JSONB) |
| `model_name`, `model_version`, `model_family`, `run_id` | Qué modelo predijo |
| `batch_id`, `input_uri` | De qué lote viene |
| `features` | Fila de entrada original (JSONB) |

```bash
docker compose exec postgres psql -U mlp -d app -c "select prediction_id, timestamp, predicted_value, model_family, model_version from predictions limit 5"
```

## Dónde mirar los resultados

### MLflow

| Experimento | Contenido |
| --- | --- |
| `eda` | Figuras, tablas y `conclusions.json` del notebook |
| `tuning` | Un run padre por modelo (`svm`) con trials anidados (`svm/trial-012`) y el reentrenamiento final (`svm/final`) |
| `selection` | Un run por selección (`production-xgboost`) con la tabla de candidatos y el umbral 1-SE |
| `inference` | Un run por lote (`batch-1a2b3c4d`) con recuento por clase y confianza media |

En `tuning`, la tabla de runs padre compara de un vistazo `cv_mean`, `cv_se`, `test_f1_macro`, `fit_seconds` y `complexity`. `search_cv_mean` es la puntuación del trial en la búsqueda y `optimism` cuánto pierde el mejor trial de la búsqueda al re-evaluarlo: mide el sesgo de selección. La tabla `rescore.json` del padre muestra la re-evaluación de los mejores trials. Cada run `final` incluye matriz de confusión, reporte por clase, importancia por permutación y el modelo; el MLP añade su curva de pérdida. El padre incluye los gráficos de Optuna y los índices del split.

Modelos registrados: `logreg`, `svm`, `random_forest`, `xgboost`, `mlp` (alias `champion`) y `classifier` (alias `production`).

### S3

| Ruta | Contenido |
| --- | --- |
| `s3://data/raw/data.csv` | Dataset original |
| `s3://data/inference/` | Lotes de inferencia |
| `s3://mlflow/` | Todos los artefactos de MLflow, modelos incluidos |

Nada se guarda en disco local. MLflow solo descarga los modelos a un directorio temporal al cargarlos.

## Configuración

`src/mlplatform/config.py` (pydantic-settings) centraliza los valores globales. Cada uno se cambia con una variable de entorno del mismo nombre en mayúsculas, o en `.env`, que leen tanto el host como los contenedores.

| Variable | Defecto | Uso |
| --- | --- | --- |
| `DATA_URI` | `s3://data/raw/data.csv` | Dataset |
| `TARGET` | `Target` | Columna objetivo |
| `SEED` | 42 | Split, folds, Optuna y modelos |
| `TEST_SIZE`, `N_SPLITS`, `N_REPEATS` | 0,2 / 5 / 2 | Validación |
| `METRIC` | `f1_macro` | Cualquier scorer de sklearn |
| `N_TRIALS` | 30 | Trials por modelo |
| `RESCORE_TOP_K` | 5 | Mejores trials que se re-evalúan con folds nuevos |
| `TEST_TOLERANCE` | 0,05 | Control en test de la selección |

Los hiperparámetros no van aquí: viven en cada módulo de modelo. Cambiar `METRIC`, `SEED` o la validación requiere volver a entrenar todos los modelos para que la comparación siga siendo justa.

## Estructura

```plain
├── lab/eda.ipynb
├── src/mlplatform/
│   ├── config.py      data.py      features.py
│   ├── models/        logreg.py  svm.py  random_forest.py  xgboost.py  mlp.py
│   ├── tuning.py      selection.py   inference.py   reporting.py
│   ├── storage.py     (buckets y subida a S3)
│   └── flows.py       (Prefect)
├── tests/
└── docker-compose.yml   Dockerfile   Makefile   .env
```

## Hallazgos y limitaciones

- **El mes explica el target**: enero → A, abril → B, julio → C, octubre → D. El split aleatorio estratificado mide interpolación: asume datos nuevos de esos mismos periodos. Un split temporal no es viable porque dejaría la clase D (solo octubre) fuera del entrenamiento.
- **≈6 % de filas fuera de patrón**: `F22` es la única pista para ellas.
- **Re-evaluación con los mismos datos**: Se evaluan los modelos con folds nuevos que reparten las mismas 400 filas de otra forma. Eliminan la suerte de la partición, no la de la muestra; para eso sigue estando el test.
- **Modelo no residente**: la inferencia carga el modelo una vez por lote. Para esto, se puede montar un FastAPI dentro de un container y utilizarlo como servicio de inferencia, tendriamos que tener tambien un endpoint de update que se lance desde prefect como ultimo paso del flujo de train-select al actualizar el champion.
- **Ejecución desde el host**: el proxy de artefactos de MLflow devuelve URLs prefirmadas con el host interno `s3:9000`, así que el host no puede descargar modelos. Entrenamiento e inferencia corren en contenedores; desde el host solo se usan el notebook y los tests. Aun asi podemos hacer la descarga manual entrando desde el panel de rustfs, pero no es necesario en ningun momento para correr la plataforma.
- **Entrenamiento secuencial**: los estudios se ejecutan uno tras otro. Se pueden hacer como flujos paralelos tipo map en el entrenamiento y reduce en la seleccion, esto seria mas rapido pero consumiria mas recursos, para pruebas en local mejor secuencial

## Escalado a cloud

| Local | Cloud |
| --- | --- |
| RustFS | S3 / GCS |
| Postgres | RDS / Cloud SQL |
| MLflow | Contenedor gestionado (ECS/Cloud Run) o MLflow gestionado, con backend en RDS |
| Prefect server + serve | Prefect Cloud o servidor propio, con work pools de Kubernetes/ECS: un contenedor por flow run |
| Estudios de Optuna | Trials en paralelo en varios workers que comparten el storage en Postgres |
| Inferencia | Flow batch programado; un servicio FastAPI con autoscaling si se necesita online |
| Infraestructura | Terraform; imágenes en un registry; CI con tests y despliegue |
