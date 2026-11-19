FROM python:3.11-slim
WORKDIR /app
COPY fase-2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fase-2/ /app/
COPY fase-1/DataSet_House_Price_Prediction.csv /fase-1/DataSet_House_Price_Prediction.csv
CMD ["python", "train.py", "--data", "/app/fase-1/DataSet_House_Price_Prediction.csv", "--model", "/app/artifacts/modelo_fase2.joblib", "--output-dir", "/app/artifacts"]
