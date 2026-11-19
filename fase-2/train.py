from pathlib import Path

import joblib
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "fase-1" / "DataSet_House_Price_Prediction.csv"
MODEL_PATH = BASE_DIR / "modelo_fase2.joblib"


# ---------------------------------------------------------
# Carga y preparación de los datos
# ---------------------------------------------------------

def cargar_datos(ruta=None):
    """Carga y prepara el conjunto de datos utilizado en la Fase 1."""

    ruta = Path(ruta) if ruta is not None else DATA_PATH
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {ruta}"
        )

    df = pd.read_csv(ruta)

    # Mantener únicamente registros con precio válido.
    df = df[df["price"] > 0].copy()

    # Introducir de forma reproducible los valores faltantes
    # utilizados en la Fase 1.
    np.random.seed(RANDOM_STATE)
    cantidad_nulos = int(len(df) * 0.01)

    filas_nulas = np.random.choice(
        df.index,
        cantidad_nulos,
        replace=False
    )

    df.loc[filas_nulas, "sqft_living"] = np.nan

    y = df["price"]

    # price_per_sqft depende del precio y produciría fuga de información.
    X = df.drop(columns=["price", "price_per_sqft"]).copy()

    # Transformación de la fecha.
    X["date"] = pd.to_datetime(X["date"])
    X["year_sale"] = X["date"].dt.year
    X["month_sale"] = X["date"].dt.month
    X = X.drop(columns=["date"])

    # street tiene cardinalidad muy alta.
    X = X.drop(columns=["street"])

    return X, y


# ---------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------

def crear_preprocesador(X):
    """Construye el preprocesamiento usado por todos los modelos."""

    categoricas = ["city", "statezip"]
    numericas = [
        columna
        for columna in X.columns
        if columna not in categoricas
    ]

    pipeline_numerico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median"))
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categoricas
            ),
            (
                "num",
                pipeline_numerico,
                numericas
            )
        ]
    )

    return preprocesador


# ---------------------------------------------------------
# Modelos candidatos
# ---------------------------------------------------------

def crear_modelos(X):
    """Crea los tres modelos candidatos de la Fase 1."""

    modelos = {
        "Regresión Lineal": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    }

    pipelines = {}

    for nombre, modelo in modelos.items():
        pipelines[nombre] = Pipeline(
            steps=[
                ("preprocesamiento", crear_preprocesador(X)),
                ("modelo", modelo)
            ]
        )

    return pipelines


# ---------------------------------------------------------
# Métricas
# ---------------------------------------------------------

def calcular_metricas(y_real, y_pred):
    """Calcula MAE, RMSE y R²."""

    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred)
    }


# ---------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------

def entrenar(data_path=None, model_path=None, output_dir=None):
    X, y = cargar_datos(data_path)
    model_path = Path(model_path) if model_path else MODEL_PATH
    output_dir = Path(output_dir) if output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. Separación entrenamiento / prueba
    # -----------------------------------------------------
    # El conjunto de prueba se reserva para la evaluación final.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # -----------------------------------------------------
    # 2. Separación entrenamiento / validación
    # -----------------------------------------------------
    # La selección del modelo se realiza SIN utilizar test.

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    modelos = crear_modelos(X_train_sub)

    resultados_validacion = {}

    print("\nRESULTADOS DE VALIDACIÓN")
    print("-" * 60)

    for nombre, modelo in modelos.items():

        modelo.fit(X_train_sub, y_train_sub)

        pred_val = modelo.predict(X_val)

        metricas = calcular_metricas(y_val, pred_val)

        resultados_validacion[nombre] = metricas

        print(f"\n{nombre}")
        print(f"MAE:  {metricas['MAE']:.2f}")
        print(f"RMSE: {metricas['RMSE']:.2f}")
        print(f"R²:   {metricas['R2']:.4f}")

    # -----------------------------------------------------
    # 3. Selección del mejor modelo
    # -----------------------------------------------------
    # MAE es la métrica principal, igual que en la Fase 1.

    mejor_nombre = min(
        resultados_validacion,
        key=lambda nombre: resultados_validacion[nombre]["MAE"]
    )

    print("\n" + "=" * 60)
    print("MODELO SELECCIONADO:", mejor_nombre)
    print("=" * 60)

    # -----------------------------------------------------
    # 4. Reentrenamiento con TODO el conjunto de entrenamiento
    # -----------------------------------------------------

    modelo_final = crear_modelos(X_train)[mejor_nombre]

    modelo_final.fit(X_train, y_train)

    # -----------------------------------------------------
    # 5. Evaluación final
    # -----------------------------------------------------
    # Test se utiliza por primera vez aquí.

    pred_test = modelo_final.predict(X_test)

    metricas_test = calcular_metricas(y_test, pred_test)

    print("\nEVALUACIÓN FINAL EN TEST")
    print("-" * 60)
    print(f"MAE:  {metricas_test['MAE']:.2f}")
    print(f"RMSE: {metricas_test['RMSE']:.2f}")
    print(f"R²:   {metricas_test['R2']:.4f}")

    # -----------------------------------------------------
    # 6. Guardar modelo
    # -----------------------------------------------------

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, model_path)
    # Dos artefactos versionados con metadatos y métricas verificables.
    for version, nombre in enumerate(sorted(resultados_validacion, key=lambda n: resultados_validacion[n]["MAE"])[:2], start=1):
        candidato = crear_modelos(X_train)[nombre]
        candidato.fit(X_train, y_train)
        ruta_version = output_dir / f"modelo_v{version}.joblib"
        joblib.dump(candidato, ruta_version)
        meta = {"version": version, "modelo": nombre, "fecha_utc": datetime.now(timezone.utc).isoformat(), "random_state": RANDOM_STATE, "filas_entrenamiento": len(X_train), "metricas_validacion": {k: float(v) for k, v in resultados_validacion[nombre].items()}, "ruta_modelo": ruta_version.name}
        (output_dir / f"modelo_v{version}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nModelo guardado correctamente en:")
    print(model_path)

    return modelo_final


# ---------------------------------------------------------
# Ejecución
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento reproducible de la fase 2")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    entrenar(args.data, args.model, args.output_dir)
