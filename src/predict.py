import pandas as pd
import joblib
from src.features.datetime_features import add_datetime_features_predire
from src.models.utils import add_fold_features_arr

# Charger les ressources
feature_names = joblib.load("models/feature_names.pkl")
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")


X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze() 

cat_cols = ['airline', 'origin_airport', 'dest_airport']

def predict(flight_input):
    X_new = pd.DataFrame([flight_input.dict()])
    X_new = add_datetime_features_predire(X_new)
    
    X_new[cat_cols] = X_new[cat_cols].astype(str)
    X_new[cat_cols] = encoder.transform(X_new[cat_cols])
    
    _, X_new = add_fold_features_arr(X_train, y_train.to_frame(), X_new)
    
    X_new = X_new[feature_names]
    pred_delay = model.predict(X_new)[0]
    
    return pred_delay