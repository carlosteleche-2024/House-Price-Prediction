# Fase 2 — House Price Prediction

Proyecto Integrador de Modelos y Simulación de Sistemas I, Universidad de Antioquia.

## Requisitos

Python 3.11 y dependencias en `fase-2/requirements.txt`. Desde la raíz del repositorio:

```bash
python -m pip install -r fase-2/requirements.txt
python -m pytest fase-2/test_model.py -v
python fase-2/train.py --data fase-1/DataSet_House_Price_Prediction.csv --model fase-2/artifacts/modelo_fase2.joblib --output-dir fase-2/artifacts
python fase-2/predict.py --model fase-2/artifacts/modelo_fase2.joblib --input fase-1/DataSet_House_Price_Prediction.csv --output fase-2/artifacts/predicciones.csv
```

El entrenamiento selecciona el modelo por MAE en validación, reserva el test para evaluación final y genera dos versiones (`modelo_v1.joblib`, `modelo_v2.joblib`) junto con archivos JSON de metadatos y métricas de validación. Los resultados son reproducibles con semilla 42; las marcas de tiempo de metadatos corresponden a cada ejecución. El dataset original se conserva sin modificaciones.

## Docker

Construir desde la **raíz** del repositorio:

```bash
docker build -f fase-2/Dockerfile -t house-price-fase2 .
docker run --rm -v "${PWD}/fase-2/artifacts:/app/artifacts" house-price-fase2
```

En PowerShell `${PWD}` se expande a la ruta actual. El volumen permite conservar modelos y metadatos en el equipo. Para ejecutar pruebas:

```bash
docker run --rm --entrypoint python house-price-fase2 -m pytest /app/test_model.py -v
```

## Alcance

El dataset se utiliza con precios positivos; se simula un 1 % de faltantes en `sqft_living` con semilla fija. Se excluyen `price_per_sqft` (fuga de información) y `street` (alta cardinalidad). El preprocesamiento se ajusta exclusivamente con datos de entrenamiento. La partición es aleatoria; se recomienda evaluar en el futuro particiones temporales y viviendas repetidas.
