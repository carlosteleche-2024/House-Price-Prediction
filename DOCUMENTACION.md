# Documentacion del codigo original

## 1. Alcance de esta entrega

Este documento describe unicamente el estado actual de la primera implementacion del proyecto **House Price Prediction**. La implementacion original esta contenida en el notebook `Entrega_1_Modelos_Simulacion.ipynb`, ubicado en la rama `feature/modelo-predictivo` del repositorio.

En esta etapa se realiza un flujo inicial de exploracion, limpieza, preparacion de datos, entrenamiento y comparacion de modelos de regresion. No se documentan como terminadas las etapas posteriores de mejora, validacion avanzada, despliegue o automatizacion.

## 2. Objetivo

Construir modelos capaces de estimar el precio (`price`) de una vivienda a partir de sus caracteristicas fisicas, de ubicacion y de la fecha de venta.

## 3. Fuente de datos

El notebook espera encontrar un archivo llamado `DataSet_House_Price_Prediction.csv` en el directorio de ejecucion. El archivo no esta incluido en el estado actual de la rama documentada, por lo que debe proporcionarse antes de ejecutar el notebook de forma reproducible.

Las variables utilizadas explicitamente durante el analisis incluyen:

- `price`: variable objetivo.
- `price_per_sqft`: variable derivada excluida de los predictores.
- `sqft_living`, `sqft_lot`, `bedrooms` y `bathrooms`: caracteristicas numericas revisadas durante el analisis.
- `waterfront`, `view` y `condition`: caracteristicas adicionales usadas por el modelo.
- `street`, `city` y `statezip`: variables de ubicacion.
- `date`: fecha de venta, transformada posteriormente en ano y mes.

## 4. Flujo implementado

### 4.1 Carga y copia de los datos

Se carga el CSV mediante `pandas.read_csv` y se crea una copia de trabajo para conservar el conjunto original.

### 4.2 Simulacion de valores faltantes

Para probar el tratamiento de datos incompletos, se fija la semilla aleatoria en `42` y se reemplaza aproximadamente el 1% de los valores de `sqft_living` por valores nulos (`NaN`).

### 4.3 Exploracion inicial

El notebook revisa:

- dimensiones del conjunto de datos;
- tipos de datos;
- estadisticas descriptivas;
- histogramas de `price`, `sqft_living`, `sqft_lot`, `bedrooms` y `bathrooms`;
- valores atipicos mediante el rango intercuartilico (IQR);
- diagramas de caja;
- registros con valores extremos;
- viviendas con precio igual a cero.

### 4.4 Limpieza

Los registros cuyo `price` es igual a cero se eliminan. Los valores faltantes de `sqft_living` se reemplazan con la mediana de esa columna calculada despues de la limpieza.

### 4.5 Preparacion de variables

Se separan los datos en:

- `y`: columna `price`;
- `X`: todas las demas columnas, excepto `price` y `price_per_sqft`.

La columna `date` se convierte a fecha y se descompone en `year_sale` y `month_sale`; despues se elimina la fecha original. La columna `street` se elimina por su alta cardinalidad. Las variables `city` y `statezip` se codifican con `OneHotEncoder(handle_unknown="ignore")`; las variables numericas pasan sin transformacion adicional.

### 4.6 Division del conjunto

Los datos se dividen en 80% para entrenamiento y 20% para prueba mediante `train_test_split`, usando `random_state=42`.

## 5. Modelos entrenados

Los tres modelos se implementan dentro de un `Pipeline` de scikit-learn, de forma que el preprocesamiento se aplica junto con el entrenamiento:

1. **Regresion lineal** (`LinearRegression`).
2. **Random Forest** (`RandomForestRegressor`) con `n_estimators=100`, `random_state=42` y `n_jobs=-1`.
3. **Gradient Boosting** (`GradientBoostingRegressor`) con `n_estimators=100` y `random_state=42`.

## 6. Metricas observadas

El notebook conserva la siguiente tabla de resultados para el conjunto de prueba:

| Modelo | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Regresion lineal | 152702.62 | 243341.29 | 0.60 |
| Random Forest | 133869.33 | 534737.36 | -0.92 |
| Gradient Boosting | 129694.30 | 247623.76 | 0.59 |

El MAE representa el error absoluto medio y el RMSE penaliza con mayor fuerza los errores grandes. El valor de R2 indica la proporcion de variabilidad explicada por el modelo respecto a la referencia de la media.

De acuerdo con estos resultados guardados, Gradient Boosting presenta el menor MAE. Sin embargo, la regresion lineal obtiene un R2 ligeramente superior y un RMSE menor que Gradient Boosting. El resultado de Random Forest debe revisarse con cuidado porque su R2 negativo y su RMSE elevado indican un desempeno deficiente en esta particion de prueba.

Estos valores son resultados registrados en el notebook original; no deben considerarse una validacion definitiva hasta volver a ejecutar el flujo con el CSV disponible y repetir la evaluacion.

## 7. Estado y limitaciones conocidas

- El archivo de datos requerido no esta incluido en el estado actual del repositorio.
- La ejecucion guardada del notebook registra `FileNotFoundError` cuando el CSV no se encuentra en el directorio de trabajo.
- Solo se utiliza una division entrenamiento/prueba; no hay validacion cruzada.
- No se observa ajuste de hiperparametros.
- No se documenta todavia un analisis de importancia de variables ni una estrategia de despliegue.
- La simulacion de faltantes sirve para probar el flujo, pero no reemplaza un analisis de la calidad real de los datos.

## 8. Siguiente etapa sugerida

La siguiente etapa deberia asegurar la disponibilidad y descripcion formal del dataset, ejecutar nuevamente el notebook, verificar que las metricas sean reproducibles y ampliar la evaluacion con validacion cruzada y ajuste de hiperparametros. Esas actividades quedan fuera del alcance de esta documentacion inicial.