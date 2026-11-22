import sys,os
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parent))
import api
from service import REQUIRED

client=TestClient(api.app)

def sample():
    row=pd.read_csv(Path(__file__).resolve().parents[1]/'fase-1'/'DataSet_House_Price_Prediction.csv').iloc[0]
    return {c:(str(row[c]) if c in ('date','city','statezip') else float(row[c])) for c in REQUIRED}

def test_health():
    r=client.get('/health'); assert r.status_code==200 and r.json()['status']=='ok'

def test_valid(tmp_path,monkeypatch):
    import service
    monkeypatch.setattr(service,'MODELS',tmp_path)
    # Modelo de prueba pequeño, con el esquema real.
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from service import preparar_entrada
    import joblib
    X=preparar_entrada(pd.DataFrame([sample(),sample()]))
    from train import crear_preprocesador
    model=Pipeline([('preprocesamiento',crear_preprocesador(X)),('modelo',DummyRegressor())]);model.fit(X,[100000,200000]);joblib.dump(model,tmp_path/'modelo_v1.joblib')
    monkeypatch.setattr(api,'LOG',tmp_path/'log.csv')
    r=client.post('/predict',json=sample())
    assert r.status_code==200 and r.json()['version']=='v1' and isinstance(r.json()['prediccion'],float)
    bad=sample();bad.pop('bedrooms')
    r=client.post('/predict',json=bad)
    assert r.status_code==422 and 'bedrooms' in r.json()['detail']
    logs=pd.read_csv(tmp_path/'log.csv');assert len(logs)==2 and list(logs.estado_http)==[200,422]

def test_missing_required(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'LOG',tmp_path/'log.csv')
    r=client.post('/predict',json={'date':'2014-05-02'})
    assert r.status_code==422 and 'bedrooms' in r.json()['detail']
