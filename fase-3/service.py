"""Versiones inmutables y reutilización del pipeline de Fase 2."""
import json, os, sys, uuid, subprocess
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fase-2'))
from train import cargar_datos, crear_modelos, calcular_metricas, RANDOM_STATE
from predict import preparar_entrada

DATA = Path(os.getenv('DATA_PATH', str(ROOT/'fase-1'/'DataSet_House_Price_Prediction.csv')))
MODELS = Path(os.getenv('MODELS_DIR', str(ROOT/'artifacts')))
REQUIRED = ['date','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated','city','statezip']
CATEGORICAL = {'city','statezip'}

def model_paths(version):
    if not isinstance(version,str) or not version or not all(c.isalnum() or c in '_-' for c in version):
        raise ValueError('Identificador de versión inválido')
    return MODELS/f'modelo_{version}.joblib', MODELS/f'modelo_{version}.json'

def validate(record):
    if not isinstance(record,dict): raise ValueError('Se espera un objeto JSON')
    missing=[c for c in REQUIRED if c not in record]
    if missing: raise ValueError('Faltan campos obligatorios: '+', '.join(missing))
    row={}
    for c in REQUIRED:
        v=record[c]
        if c=='date':
            if not isinstance(v,str) or pd.isna(pd.to_datetime(v,errors='coerce')):
                raise ValueError('date debe ser una fecha válida en texto')
        elif c in CATEGORICAL:
            if not isinstance(v,str) or not v.strip(): raise ValueError(f'{c} debe ser texto no vacío')
        elif v is not None:
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):
                raise ValueError(f'{c} debe ser numérico o null')
        row[c]=v
    return row

def predict(record,version):
    row=validate(record)
    path,_=model_paths(version)
    if not path.is_file(): raise FileNotFoundError(f'No existe la versión {version}')
    model=joblib.load(path)
    x=preparar_entrada(pd.DataFrame([row]))
    return float(model.predict(x)[0])

def train_new(version=None, model_override=None):
    version=version or datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    model_path, meta_path=model_paths(version)
    MODELS.mkdir(parents=True,exist_ok=True)
    if model_path.exists() or meta_path.exists(): raise FileExistsError(f'La versión {version} ya existe')
    X,y=cargar_datos(DATA)
    xtr,xt, ytr,yt=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE)
    xs,xv,ys,yv=train_test_split(xtr,ytr,test_size=.20,random_state=RANDOM_STATE)
    scores={}
    for name, model in crear_modelos(xs).items():
        model.fit(xs,ys)
        scores[name]=float(calcular_metricas(yv,model.predict(xv))['MAE'])
    best=model_override or min(scores,key=scores.get)
    if best not in scores: raise ValueError("Modelo desconocido")
    model=crear_modelos(xtr)[best]
    model.fit(xtr,ytr)
    metrics={k:float(v) for k,v in calcular_metricas(yt,model.predict(xt)).items()}
    baseline={k:float(v) for k,v in calcular_metricas(yt,np.full(len(yt),float(ytr.median()))).items()}
    meta={'version':version,'created_utc':datetime.now(timezone.utc).isoformat(),'model':best,
          'target':'price','features':REQUIRED,'feature_types':{c:('string' if c in CATEGORICAL or c=='date' else 'number|null') for c in REQUIRED},
          'parameters':{'random_state':RANDOM_STATE,'test_size':.20,'validation_size_of_train':.20},
          'data':str(DATA),'rows_train':len(xtr),'rows_test':len(xt),
          'validation_mae':scores,'test_metrics':metrics,'baseline_median_test_metrics':baseline,
          'environment':{'python':sys.version.split()[0],'scikit_learn':__import__('sklearn').__version__},
          'commit':os.getenv('GIT_COMMIT','not-set')}
    # Nunca sobrescribir: escritura exclusiva de reserva y limpieza en error.
    try:
        with model_path.open('xb') as f: joblib.dump(model,f)
        with meta_path.open('x',encoding='utf8') as f: json.dump(meta,f,indent=2,ensure_ascii=False)
    except Exception:
        model_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise
    return meta
