import pandas as pd
import joblib
from src.features.datetime_features import add_datetime_features_predire
from src.features.historical_features import add_historical_features
from src.models.utils import add_fold_features_arr

# Charger le modèle et l'encodeur
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")

# Charger X_train et y_train pour les features historiques
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

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

if __name__ == "__main__": 
    idx = 45
    x_example = X_test.iloc[[idx]].copy() 
     
    print(x_example.columns)
    # Ajouter les features dérivées sans fuite en utilisant X_train et y_train
    _, x_example = add_fold_features_arr(X_train, y_train, x_example)

    # Prédiction
    predicted_arr_delay = model.predict(x_example)[0]
    true_arr_delay = y_test.iloc[idx, 0] 

    print(f"Exemple index: {idx}")
    print(f"Predicted arrival delay: {predicted_arr_delay:.2f} minutes")
    print(f"True arrival delay: {true_arr_delay:.2f} minutes")
    print(f"Erreur: {predicted_arr_delay - true_arr_delay:.2f} minutes")

