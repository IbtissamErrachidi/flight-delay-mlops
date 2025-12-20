FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements_api.txt

# Copier le modèle et l'encodeur
COPY models/ ./models/

# Copier uniquement X_train et y_train pour les features historiques
COPY data/processed/X_train.csv ./data/processed/X_train.csv
COPY data/processed/y_train.csv ./data/processed/y_train.csv

# Copier le reste du code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
