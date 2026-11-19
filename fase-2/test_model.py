
from pathlib import Path

import pandas as pd
import pytest

from train import cargar_datos, calcular_metricas


def test_archivo_datos_existe():
    ruta = Path(__file__).resolve().parent.parent / "fase-1" / "DataSet_House_Price_Prediction.csv"
    assert ruta.exists(), "No se encontró el archivo de datos"


def test_cargar_datos():
    datos = cargar_datos()

    assert isinstance(datos, pd.DataFrame)
    assert not datos.empty
    assert "price" in datos.columns


def test_metricas_modelo_perfecto():
    valores = [100000, 200000, 300000]

    metricas = calcular_metricas(valores, valores)

    assert metricas["MAE"] == pytest.approx(0)
    assert metricas["RMSE"] == pytest.approx(0)
    assert metricas["R2"] == pytest.approx(1)


def test_metricas_error_positivo():
    reales = [100000, 200000, 300000]
    predichos = [110000, 190000, 310000]

    metricas = calcular_metricas(reales, predichos)

    assert metricas["MAE"] > 0
    assert metricas["RMSE"] > 0
