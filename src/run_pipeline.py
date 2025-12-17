import pandas as pd
from src.data.build_dataset import build_dataset 
from src.features.datetime_features import add_datetime_features
from src.features.historical_features import add_historical_features
from src.preprocessing.outliers import filter_outliers
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


# 1) Lire les données
df = pd.read_csv("data/raw/leftover_flights_2000.csv")

# 2) Construire X et y
X, y = build_dataset(df)

# 3) Ajouter features datetime
X = add_datetime_features(X)  # dep_hour, arr_hour, weekday, month...

# 4) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Filtrer outliers (optionnel)
X_train, y_train, X_test, y_test = filter_outliers(X_train, y_train, X_test, y_test)

# 6) Ajouter features historiques
X_train = add_historical_features(X_train, y_train, X_train)
X_test  = add_historical_features(X_train, y_train, X_test)

# 7) Encodage des colonnes catégorielles
cat_cols = ['airline', 'origin_airport', 'dest_airport']


encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])



# 8) Charger modèle et faire prédiction
model = joblib.load("models/xgboost_model.pkl")

# Charger l'ordre des features du modèle
model_features = model.get_booster().feature_names

# Réordonner X_test pour correspondre
X_test = X_test[model_features]

preds = model.predict(X_test)

# 9) Afficher résultats
print(preds[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calcul des métriques
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

# Affichage
print("Test MAE XGBoost:", mae)
print("Test RMSE XGBoost:", rmse)
print("Test R² XGBoost:", r2)

