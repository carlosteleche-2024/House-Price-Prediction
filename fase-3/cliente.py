"""Cliente de prueba con una fila realista y sin variable objetivo."""
import json,sys,urllib.request
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
row=pd.read_csv(root/'fase-1'/'DataSet_House_Price_Prediction.csv').iloc[0].drop(labels=['price','price_per_sqft','street']).to_dict()
body=json.dumps(row).encode()
req=urllib.request.Request(sys.argv[1] if len(sys.argv)>1 else 'http://localhost:8000/predict',data=body,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req) as response: print(response.status,response.read().decode())
