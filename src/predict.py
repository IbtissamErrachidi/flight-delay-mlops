import pandas as pd
import joblib
from src.features.datetime_features import add_datetime_features_predire
from src.features.historical_features import add_historical_features


# Charger le modèle et l'encodeur
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")

# Charger X_train et y_train pour les features historiques
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")



cat_cols = ['airline', 'origin_airport', 'dest_airport']

def predict(flight_input):
    # Convertir l'entrée en DataFrame
    X_new = pd.DataFrame([flight_input.dict()])

    # Ajouter features datetime
    X_new = add_datetime_features_predire(X_new)

    # Ajouter features historiques
    X_new = add_historical_features(X_train, y_train, X_new)

    # Encoder les colonnes catégorielles
    X_new[cat_cols] = X_new[cat_cols].astype(str)
    X_new[cat_cols] = encoder.transform(X_new[cat_cols])

    # Faire la prédiction
    pred_delay = model.predict(X_new)[0]

    return pred_delay



