import pandas as pd
from src.data.build_dataset import build_dataset
from sklearn.model_selection import train_test_split
from src.features.datetime_features import add_datetime_features
from src.preprocessing.encoding import encode_categorical
from src.preprocessing.outliers import filter_outliers
import joblib



# 1) Lire les données
df = pd.read_csv("data/raw/df_final.csv")
# 2) Construire X et y
X, y = build_dataset(df)

# 3) Ajouter features datetime
X = add_datetime_features(X)

# 4) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Filtrer outliers (optionnel)
X_train, y_train, X_test, y_test = filter_outliers(X_train, y_train, X_test, y_test)



# 6) Encoder colonnes catégorielles
cat_cols = ['airline', 'origin_airport', 'dest_airport']
X_train, X_test, encoder = encode_categorical(X_train, X_test, cat_cols)

# Sauvegarder l'encodeur pour l'utiliser dans evaluate.py
joblib.dump(encoder, "models/encoder.pkl")

# 7) Sauvegarder datasets traités
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Préparation des données terminée. X_train, X_test, y_train, y_test et encoder sauvegardés.")
