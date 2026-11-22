"""API REST de predicción de precios (FastAPI)."""
import os, sys, csv
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from service import predict, train_new, model_paths

app=FastAPI(title='House Price Prediction - Fase 3 y 4',version='1.0')
LOG=Path(os.getenv('MONITOR_LOG',str(Path(__file__).resolve().parents[1]/'fase-4'/'solicitudes.csv')))

class TrainRequest(BaseModel):
    version: str | None = None

@app.middleware('http')
async def log_predict(request:Request,call_next):
    if request.url.path != '/predict' or request.method!='POST': return await call_next(request)
    request.state.version=''
    request.state.prediccion=''
    try:
        response=await call_next(request)
    except Exception:
        response=JSONResponse(status_code=500,content={'detail':'Error interno de predicción'})
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',newline='',encoding='utf8') as f:
        writer=csv.writer(f)
        if f.tell()==0: writer.writerow(['fecha_hora','estado_http','version','prediccion'])
        writer.writerow([datetime.now(timezone.utc).isoformat(),response.status_code,request.state.version,request.state.prediccion])
    return response

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/predict')
async def prediction(request:Request):
    try:
        data=await request.json()
    except Exception:
        raise HTTPException(status_code=422,detail='JSON inválido')
    if not isinstance(data,dict): raise HTTPException(status_code=422,detail='Se requiere un objeto JSON')
    version=data.get('version') or os.getenv('MODEL_VERSION','v1')
    record=data.get('registro',data)
    try:
        result=predict(record,version)
    except ValueError as e:
        raise HTTPException(status_code=422,detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,detail=str(e))
    request.state.version=version
    request.state.prediccion=result
    return {'prediccion':result,'version':version}

@app.post('/train',status_code=201)
def train(request:TrainRequest=TrainRequest()):
    try: meta=train_new(request.version)
    except FileExistsError as e: raise HTTPException(status_code=409,detail=str(e))
    except ValueError as e: raise HTTPException(status_code=422,detail=str(e))
    return {'version':meta['version'],'metricas':meta['test_metrics']}
