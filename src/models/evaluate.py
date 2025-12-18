import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostRegressor

def load_best_model():
    # Chemins possibles selon le type de modèle
    paths = {
        "xgboost/randomforest": "models/best_model.pkl",
        "lightgbm": "models/best_model.txt",
        "catboost": "models/best_model.cbm"
    }

    for key, path in paths.items():
        if os.path.exists(path):
            if key == "lightgbm":
                return lgb.Booster(model_file=path)
            elif key == "catboost":
                model = CatBoostRegressor()
                model.load_model(path)
                return model
            else:
                return joblib.load(path)
    raise FileNotFoundError("Aucun modèle best_model trouvé !")

def evaluate_model(model, X_test, y_test, output_dir="evaluation"):

    # Prédictions
    if isinstance(model, lgb.Booster):
        preds = model.predict(X_test)
    elif isinstance(model, CatBoostRegressor):
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    # Calcul métriques
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }

    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Sauvegarde predictions
    df_preds = pd.DataFrame({"true": y_test, "pred": preds})
    df_preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Scatter plot true vs pred
    plt.figure(figsize=(6,6))
    sns.scatterplot(x="true", y="pred", data=df_preds, alpha=0.6)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("True vs Predicted")
    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
    plt.close()

    # Confusion matrix binaire (retard > 0)
    y_class = (y_test > 0).astype(int)
    pred_class = (preds > 0).astype(int)
    cm = confusion_matrix(y_class, pred_class)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['On time','Delayed'], yticklabels=['On time','Delayed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return preds, metrics

if __name__ == "__main__":
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Charger automatiquement le meilleur modèle
    model = load_best_model()

    preds, metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation terminée. Metrics:", metrics)
