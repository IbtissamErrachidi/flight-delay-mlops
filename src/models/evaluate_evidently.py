import os
import pandas as pd
import joblib
from evidently import Report
from evidently.presets import RegressionPreset
from evidently.future.datasets import Dataset, DataDefinition, Regression
from src.models.evaluate import load_best_model
from src.models.utils import add_fold_features_arr

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def evaluate_evidently(model, X_test, y_test):
    #  AJOUTÉ : Charger X_train et y_train pour les features historiques
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    
    #  AJOUTÉ : Ajouter les features historiques à X_test
    _, X_test_with_hist = add_fold_features_arr(X_train, y_train.to_frame(), X_test)
    
    # AJOUTÉ : Charger les noms de features attendus par le modèle
    feature_names = joblib.load("models/feature_names.pkl")
    X_test_with_hist = X_test_with_hist[feature_names]
    
    # Préparer les données avec target et prediction
    data = X_test_with_hist.copy()  # ← MODIFIÉ : Utiliser X_test_with_hist au lieu de X_test
    data["target"] = y_test.values  # ← AJOUTÉ .values pour éviter des problèmes d'index
    data["prediction"] = model.predict(X_test_with_hist)  # ← MODIFIÉ : Utiliser X_test_with_hist

    # Créer la Data Definition avec Regression object
    regression = [Regression(target="target", prediction="prediction")]
    data_definition = DataDefinition(regression=regression)
    
    # Créer le Dataset avec la définition
    dataset = Dataset.from_pandas(data, data_definition=data_definition)

    # Créer le rapport avec RegressionPreset
    report = Report(metrics=[RegressionPreset()])

    # Exécuter le rapport avec le Dataset
    my_eval = report.run(current_data=dataset, reference_data=None)

    # Sauvegarder
    html_path = os.path.join(REPORT_DIR, "evidently_regression_report.html")
    json_path = os.path.join(REPORT_DIR, "evidently_regression_report.json")

    my_eval.save_html(html_path)
    my_eval.save_json(json_path)

    print(" Evidently report généré :")
    print(f"- HTML : {html_path}")
    print(f"- JSON : {json_path}")

if __name__ == "__main__":
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    model = load_best_model()
    evaluate_evidently(model, X_test, y_test)