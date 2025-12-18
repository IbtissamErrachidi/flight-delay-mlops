import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn
from src.models.utils import add_fold_features_arr

def train_all_models(X_train, y_train):
    # -------------------------
    # Définition des modèles
    # -------------------------
    models = [
        ("XGBoost", XGBRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=8,
            random_state=42, tree_method='hist', eval_metric='mae'
        )),
        ("LightGBM", lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=8, random_state=42
        )),
        ("CatBoost", CatBoostRegressor(
            depth=8, learning_rate=0.05, iterations=700,
            loss_function='MAE', random_seed=42, verbose=0
        )),
        ("RandomForest", RandomForestRegressor(random_state=42))
    ]

    results = []

    # -------------------------
    # Cross-validation
    # -------------------------
    for name, model in models:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, rmse_scores, r2_scores = [], [], []

        for train_idx, val_idx in tqdm(kf.split(X_train), total=kf.get_n_splits(), desc=f"CV {name}"):
            X_tr = X_train.iloc[train_idx].copy()
            y_tr = y_train.iloc[train_idx].copy()
            X_val = X_train.iloc[val_idx].copy()
            y_val = y_train.iloc[val_idx].copy()

            X_tr, X_val = add_fold_features_arr(X_tr, y_tr.to_frame(), X_val)

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            mae_scores.append(mean_absolute_error(y_val, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            r2_scores.append(r2_score(y_val, preds))

        cv_mae = np.mean(mae_scores)
        cv_rmse = np.mean(rmse_scores)
        cv_r2 = np.mean(r2_scores)

        print(f"{name} CV MAE: {cv_mae}, CV RMSE: {cv_rmse}, CV R²: {cv_r2}")

        results.append((name, model, cv_mae, cv_rmse, cv_r2))

    # -------------------------
    # Trouver le meilleur modèle (par CV R² ici)
    # -------------------------
    best_model = max(results, key=lambda x: x[4])  # x[4] = cv_r2
    model_name, model_obj, mae, rmse, r2 = best_model
    print(f"Le meilleur modèle est {model_name} avec CV R²: {r2}")

    # -------------------------
    # Sauvegarde locale du meilleur modèle seulement
    # -------------------------
    if model_name == "LightGBM":
        model_obj.booster_.save_model("models/best_model.txt")
    elif model_name == "CatBoost":
        model_obj.save_model("models/best_model.cbm")
    else:  # XGBoost et RandomForest
        joblib.dump(model_obj, "models/best_model.pkl")

    # -------------------------
    # Logging MLflow pour le meilleur modèle
    # -------------------------
    mlflow.set_experiment("flight-delay-best-model")
    if "XGBoost" in model_name:
        model_obj._estimator_type = "regressor"

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        if hasattr(model_obj, "n_estimators"):
            mlflow.log_param("n_estimators", model_obj.n_estimators)
        if hasattr(model_obj, "max_depth"):
            mlflow.log_param("max_depth", model_obj.max_depth)

        mlflow.log_metric("cv_mae", mae)
        mlflow.log_metric("cv_rmse", rmse)
        mlflow.log_metric("cv_r2", r2)

        mlflow.sklearn.log_model(model_obj, artifact_path="model")

    return best_model




if __name__ == "__main__":
    import pandas as pd
    from src.models.train import train_all_models

    # Lire les données
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


    # Entraîner le modèle
    train_all_models(X_train, y_train)
