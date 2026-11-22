FROM python:3.11-slim
WORKDIR /app
COPY fase-3/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY fase-2/ /app/fase-2/
COPY fase-3/ /app/fase-3/
COPY fase-4/ /app/fase-4/
ENV DATA_PATH=/data/DataSet_House_Price_Prediction.csv MODELS_DIR=/models MONITOR_LOG=/logs/solicitudes.csv MODEL_VERSION=v1
EXPOSE 8000
CMD ["uvicorn","api:app","--app-dir","/app/fase-3","--host","0.0.0.0","--port","8000"]

COPY fase-1/DataSet_House_Price_Prediction.csv /app/fase-1/DataSet_House_Price_Prediction.csv
