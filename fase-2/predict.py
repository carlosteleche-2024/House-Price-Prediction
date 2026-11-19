"""Predicciones de precios usando un pipeline previamente entrenado."""
import argparse
from pathlib import Path
import joblib
import pandas as pd


def preparar_entrada(df):
    """Replica las transformaciones deterministas de train.py."""
    datos = df.copy()
    requeridas = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                  'floors', 'waterfront', 'view', 'condition', 'sqft_above',
                  'sqft_basement', 'yr_built', 'yr_renovated', 'city', 'statezip']
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError(f'Faltan columnas requeridas: {faltantes}')
    fechas = pd.to_datetime(datos['date'], errors='coerce')
    if fechas.isna().any():
        raise ValueError('La columna date contiene fechas inválidas.')
    datos['year_sale'] = fechas.dt.year
    datos['month_sale'] = fechas.dt.month
    datos = datos.drop(columns=['date', 'street', 'price', 'price_per_sqft'], errors='ignore')
    return datos


def predecir_archivo(modelo_path, entrada_path, salida_path):
    modelo_path = Path(modelo_path)
    if not modelo_path.is_file():
        raise FileNotFoundError(f'No existe el modelo: {modelo_path}')
    entrada = pd.read_csv(entrada_path)
    if entrada.empty:
        raise ValueError('El archivo de entrada no contiene filas.')
    modelo = joblib.load(modelo_path)
    X = preparar_entrada(entrada)
    predicciones = modelo.predict(X)
    resultado = entrada.copy()
    resultado['precio_predicho'] = predicciones
    salida_path = Path(salida_path)
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida_path, index=False)
    print(f'Se generaron {len(resultado)} predicciones en: {salida_path}')
    return resultado


def main():
    parser = argparse.ArgumentParser(description='Predice precios de viviendas con un modelo entrenado.')
    parser.add_argument('--model', required=True, help='Ruta al archivo .joblib')
    parser.add_argument('--input', required=True, help='CSV de viviendas a predecir')
    parser.add_argument('--output', required=True, help='CSV de salida')
    args = parser.parse_args()
    predecir_archivo(args.model, args.input, args.output)


if __name__ == '__main__':
    main()
