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
import yaml

def train_all_models(X_train, y_train):
    # Charger les paramètres
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["model"]

    # Définition des modèles
    models = [
        ("XGBoost", XGBRegressor(
            n_estimators=params["xgboost"]["n_estimators"],
            max_depth=params["xgboost"]["max_depth"],
            learning_rate=params["xgboost"]["learning_rate"],
            random_state=42,
            tree_method='hist',
            eval_metric='mae'
        )),
        ("LightGBM", lgb.LGBMRegressor(
            n_estimators=params["lightgbm"]["n_estimators"],
            max_depth=params["lightgbm"]["max_depth"],
            learning_rate=params["lightgbm"]["learning_rate"],
            random_state=42
        )),
        ("CatBoost", CatBoostRegressor(
            iterations=params["catboost"]["iterations"],
            depth=params["catboost"]["depth"],
            learning_rate=params["catboost"]["learning_rate"],
            loss_function='MAE',
            random_seed=42,
            verbose=0
        )),
        ("RandomForest", RandomForestRegressor(
            n_estimators=params["random_forest"]["n_estimators"],
            max_depth=params["random_forest"]["max_depth"],
            random_state=42
        ))
    ]

    # Configuration MLflow
    mlflow.set_experiment("flight-delay-models-experiment")
    
    results = []

    # UNE SEULE BOUCLE avec CV + MLflow logging
    for name, model in models:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, rmse_scores, r2_scores = [], [], []
        
        final_model = None  # Variable pour garder le dernier modèle

        # Démarrer UN SEUL run MLflow par modèle
        with mlflow.start_run(run_name=name):
            # Log des paramètres
            mlflow.log_param("model_name", name)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)

            # Cross-validation
            for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train), 
                                                             total=kf.get_n_splits(), 
                                                             desc=f"CV {name}"), 1):
                X_tr = X_train.iloc[train_idx].copy()
                y_tr = y_train.iloc[train_idx].copy()
                X_val = X_train.iloc[val_idx].copy()
                y_val = y_train.iloc[val_idx].copy()

                # calcule sur X_tr, applique à X_val
                X_tr, X_val = add_fold_features_arr(X_tr, y_tr.to_frame(), X_val)

                model.fit(X_tr, y_tr)
                final_model = model  # Garder le modèle de ce fold
                
                preds = model.predict(X_val)

                fold_mae = mean_absolute_error(y_val, preds)
                fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
                fold_r2 = r2_score(y_val, preds)

                mae_scores.append(fold_mae)
                rmse_scores.append(fold_rmse)
                r2_scores.append(fold_r2)

                # Log des métriques par fold 
                mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)
                mlflow.log_metric(f"fold_{fold}_rmse", fold_rmse, step=fold)
                mlflow.log_metric(f"fold_{fold}_r2", fold_r2, step=fold)

            # Calculer et logger les moyennes CV
            cv_mae = np.mean(mae_scores)
            cv_rmse = np.mean(rmse_scores)
            cv_r2 = np.mean(r2_scores)

            mlflow.log_metric("cv_mae_mean", cv_mae)
            mlflow.log_metric("cv_rmse_mean", cv_rmse)
            mlflow.log_metric("cv_r2_mean", cv_r2)
            mlflow.log_metric("cv_mae_std", np.std(mae_scores))
            mlflow.log_metric("cv_rmse_std", np.std(rmse_scores))
            mlflow.log_metric("cv_r2_std", np.std(r2_scores))

            print(f"{name} CV MAE: {cv_mae:.4f} ± {np.std(mae_scores):.4f}")
            print(f"{name} CV RMSE: {cv_rmse:.4f} ± {np.std(rmse_scores):.4f}")
            print(f"{name} CV R²: {cv_r2:.4f} ± {np.std(r2_scores):.4f}\n")
            
            # Logger le modèle final (du dernier fold)
            mlflow.sklearn.log_model(final_model, artifact_path="model")

            results.append((name, final_model, cv_mae, cv_rmse, cv_r2))

    # Sélectionner et sauvegarder le meilleur modèle
    best_model = max(results, key=lambda x: x[4])  # x[4] = cv_r2
    model_name, model_obj, mae, rmse, r2 = best_model
    print(f"\n🏆 Meilleur modèle: {model_name} avec CV R²: {r2:.4f}")

    # Obtenir les noms de features en simulant add_fold_features_arr
    # On prend un petit échantillon juste pour avoir les noms de colonnes
    X_sample = X_train.iloc[:100].copy()
    y_sample = y_train.iloc[:100].copy()
    X_sample_with_hist, _ = add_fold_features_arr(X_sample, y_sample.to_frame(), X_sample)
    feature_names = X_sample_with_hist.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.pkl")

    # Sauvegarder le meilleur modèle localement
    if model_name == "LightGBM":
        model_obj.booster_.save_model("models/best_model.txt")
    elif model_name == "CatBoost":
        model_obj.save_model("models/best_model.cbm")
    else:
        joblib.dump(model_obj, "models/best_model.pkl")

    return best_model


if __name__ == "__main__":
    import pandas as pd

    # Lire les données
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Entraîner les modèles
    train_all_models(X_train, y_train)