
FROM python:3.11-slim

WORKDIR /app

COPY fase-2/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY fase-2/train.py .
COPY fase-2/predict.py .

COPY fase-1/DataSet_House_Price_Prediction.csv fase-1/DataSet_House_Price_Prediction.csv

CMD ["python", "train.py"]

