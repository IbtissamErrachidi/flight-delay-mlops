# ✈️ Flight Delay Prediction - MLOps Project

## 📋 Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Données](#données)
- [Phase 1: Développement exploratoire](#phase-1-développement-exploratoire)
- [Phase 2: Industrialisation locale](#phase-2-industrialisation-locale)
- [Phase 3: CI/CD et Orchestration](#phase-3-cicd-et-orchestration)
- [Installation et utilisation](#installation-et-utilisation)
- [Technologies utilisées](#technologies-utilisées)

---

## 🎯 Vue d'ensemble

Ce projet a pour objectif de prédire les retards d’arrivée des vols en combinant des modèles de Machine Learning et des pratiques MLOps modernes.
Il comprend une phase exploratoire sur Kaggle, suivie d’une industrialisation complète avec DVC, MLflow, Airflow, Docker et une API FastAPI pour l’inférence.

**Objectif**: Prédire le retard d'arrivée (`ARR_DELAY`) d'un vol en fonction de caractéristiques comme la compagnie aérienne, l'aéroport de départ/arrivée, l'heure prévue, la distance, etc.

---

## 🏗️ Architecture du projet

```
flight-delay-prediction/
├── .dvc/                       # Configuration DVC
├── .github/
│   └── workflows/
│       ├── ci.yml             # Pipeline CI (tests, validation)
│       └── cd.yaml            # Pipeline CD (build Docker, déploiement)
├── airflow/
│   ├── dags/
│   │   └── dvc_pipeline_dag.py # DAG Airflow pour orchestration DVC
│   ├── Dockerfile             # Image Airflow personnalisée
│   ├── docker-compose.yml     # Configuration Airflow
│   └── requirements-docker.txt # Dépendances Airflow
├── data/
│   ├── raw/                   # Données brutes (non versionnées)
│   └── processed/             # Données transformées (versionnées avec DVC)
├── src/
│   ├── data/
│   │   ├── build_dataset.py   # Construction du dataset
│   │   └── prepare_data.py    # Pipeline de préparation
│   ├── features/
│   │   └── datetime_features.py # Feature engineering temporel
│   ├── preprocessing/
│   │   ├── encoding.py        # Encodage des variables catégorielles
│   │   └── outliers.py        # Filtrage des outliers
│   ├── models/
│   │   ├── train.py           # Entraînement des modèles
│   │   ├── evaluate.py        # Évaluation
│   │   ├── evaluate_evidently.py # Monitoring Evidently
│   │   └── utils.py           # Fonctions utilitaires
│   └── predict.py             # Logique de prédiction 
├── api/
│   └── main.py                # API FastAPI
├── models/                    # Modèles entraînés (versionnés avec DVC)
├── evaluation/                # Métriques et visualisations
├── reports/                   # Rapports Evidently
├── .gitignore                 # Fichiers Git à ignorer
├── .dvcignore                 # Fichiers DVC à ignorer
├── dvc.yaml                   # Pipeline DVC
├── dvc.lock                   # Verrouillage du pipeline DVC
├── params.yaml                # Hyperparamètres
├── Dockerfile                 # Conteneurisation API
├── requirements.txt           # Dépendances principales
├── requirements_api.txt       # Dépendances API
└── requirements_ci.txt        # Dépendances CI/CD
```

---

## 📊 Données

### Source
Dataset officiel disponible sur Kaggle:
- **Nom**: [Airline Delay and Cancellation Data (2009-2018)](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)
- **Période**: 2009 - 2018
- **Taille originale**: Plusieurs millions de vols par année

### Stratégie d'échantillonnage

Vu la taille importante des données (plusieurs Go), un échantillonnage stratifié a été appliqué:

- **100,000 lignes par année** (réparties uniformément sur les 12 mois)
- **~8,333 lignes par mois** pour assurer une distribution temporelle équilibrée
- Échantillonnage aléatoire avec `random_state=42` pour la reproductibilité

#### Variables sélectionnées

| Variable | Description |
|----------|-------------|
| `FL_DATE` | Date du vol |
| `OP_CARRIER` | Code de la compagnie aérienne |
| `OP_CARRIER_FL_NUM` | Numéro du vol |
| `ORIGIN` | Aéroport de départ |
| `DEST` | Aéroport de destination |
| `CRS_DEP_TIME` | Heure prévue de départ |
| `CRS_ARR_TIME` | Heure prévue d'arrivée |
| `CRS_ELAPSED_TIME` | Temps prévu de vol |
| `DISTANCE` | Distance du vol |
| `DEP_DELAY` | Retard au départ |
| `TAXI_OUT` | Temps entre gate et décollage |
| `WHEELS_OFF` | Heure réelle du décollage |
| `WHEELS_ON` | Heure réelle d'atterrissage |
| `TAXI_IN` | Temps entre atterrissage et gate |
| `CANCELLED` | Vol annulé (0/1) |
| `DIVERTED` | Vol détourné (0/1) |
| **`ARR_DELAY`** | **Retard d'arrivée (cible)** |

### Résultat final
- **Shape**: (983,294, 19)
- **Période**: 2009-2018
- **Distribution**: Équilibrée par année et par mois

---

## 🔬 Phase 1: Développement exploratoire

**Environnement**: Kaggle Notebooks

Cette phase initiale a permis de valider la faisabilité du projet dans un environnement flexible.

### Tâches réalisées

| Tâche | Description |
|-------|-------------|
| **EDA** | Exploration des données, analyse des distributions, corrélations |
| **Fusion & nettoyage** | Échantillonnage des 10 années, gestion des valeurs manquantes |
| **Feature engineering** | Extraction de features temporelles (heure, jour, mois) et historiques (moyennes par compagnie/aéroport, ect) |
| **Entraînement** | Test de plusieurs algorithmes (XGBoost, LightGBM, CatBoost, Random Forest) |
| **Documentation** | Notebook propre et reproductible |


### Insights clés
- Les retards au départ (`DEP_DELAY`) sont fortement corrélés avec les retards à l'arrivée
- Les aéroports et compagnies ont des patterns de retard distincts
- Les heures de pointe (matin et soir) présentent plus de retards
- Les vols longue distance sont plus susceptibles de récupérer du retard

### Livrable
✅ Notebook Kaggle finalisé avec modèle entraîné et features calculées

---

## 🚀 Phase 2: Industrialisation locale

**Environnement**: GitHub + Machine locale + Docker

Cette phase a transformé le code exploratoire en un projet MLOps production-ready.

### Tâches réalisées

| Tâche | Description |
|-------|-------------|
| **Structuration** | Organisation du code en modules (`/src`, `/data`, `/models`) |
| **DVC** | Versioning des données et création du pipeline `dvc.yaml` |
| **MLflow** | Tracking des expériences et enregistrement des modèles |
| **API FastAPI** | Développement de l'API d'inférence |
| **Docker** | Conteneurisation complète de l'API  |

### 📦 Pipeline DVC

Le pipeline est défini dans `dvc.yaml` et comprend 4 étapes:

```yaml
stages:
  1. prepare_data    # Préparation et split des données
  2. train_model     # Entraînement avec CV et MLflow
  3. evaluate        # Évaluation sur le test set
  4. evaluate_evidently # Génération de rapports de monitoring
```

**Exécution**:
```bash
dvc repro
```

### 📊 MLflow Tracking

Tous les modèles sont trackés avec MLflow:
- Hyperparamètres
- Métriques de cross-validation (MAE, RMSE, R²)
- Métriques par fold
- Modèles enregistrés

**Interface MLflow**:
```bash
mlflow ui
```

### 📄 Visualisation avec Evidently

Après avoir suivi le pipeline et tracké les modèles avec MLflow, un **rapport Evidently** est généré pour le modèle `best_model` afin d’analyser ses performances et la qualité des prédictions.

**Génération et ouverture du rapport** :  

```bash
# Lance la visualisation du rapport dans le navigateur
start reports/evidently_regression_report.html
```

## Contenu du dossier `reports/` après génération

reports/
├── evidently_regression_report.html  # Rapport interactif
└── evidently_regression_report.json  # Données brutes utilisées par Evidently


### Ce rapport permet de visualiser :

- Distribution des prédictions vs valeurs réelles
- Analyse des features importantes
- Évolution des erreurs par sous-groupes
- Détection de dérives (drifts) ou anomalies


### 🌐 API FastAPI

L'API expose deux endpoints:

#### `GET /`
Vérification de l'état de l'API

#### `POST /predict`
Prédiction du retard d'arrivée

**Exemple de requête**:
```json
{
  "flight_date": "2025-12-19",
  "airline": 19,
  "flight_number": 3202,
  "origin_airport": 305,
  "dest_airport": 56,
  "scheduled_dep_time": 1420.0,
  "dep_delay": 5.0,
  "taxi_out": 18.0,
  "wheels_off": 1432.0,
  "scheduled_arr_time": 1600.0,
  "scheduled_elapsed_time": 100.0,
  "distance": 450.0,
  "year": 2025,
  "month": 12
}
```

**Réponse**:
```json
{
  "predicted_arrival_delay": 8.5
}
```

### 🐳 Docker

L'API est conteneurisé pour un déploiement simplifié:

```dockerfile
FROM python:3.11-slim
# Installation des dépendances
# Copie des modèles et données nécessaires
# Exposition du port 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Livrable
✅ Projet MLOps structuré, pipeline versionné, modèle conteneurisé avec API fonctionnelle

---

## ⚙️ Phase 3: CI/CD et Orchestration

**Environnement**: GitHub Actions + Airflow + Docker

Cette phase a ajouté l'automatisation complète avec des pipelines CI/CD et l'orchestration du workflow MLOps.

### Tâches réalisées

| Tâche | Description |
|-------|-------------|
| **GitHub Actions CI** | Tests automatisés, validation du code, test du pipeline de données |
| **GitHub Actions CD** | Build et déploiement automatique des images Docker |
| **Airflow** | Orchestration du pipeline DVC avec DAG |
| **Docker Compose** | Configuration multi-conteneurs (Airflow + API) |

### 🔄 Pipeline CI (Continuous Integration)

Le workflow CI (`ci.yml`) s'exécute automatiquement à chaque push ou pull request sur la branche `main`.

**Jobs CI**:

1. **build-test**: Vérification de base du code
   - Checkout du code
   - Installation de Python 3.13
   - Installation des dépendances
   - Compilation du code source
   - Exécution des tests unitaires

2. **test-data-pipeline**: Test du pipeline de données
   - Téléchargement des données depuis Google Drive
   - Exécution de `prepare_data.py`
   - Vérification de la création des fichiers traités

3. **test-api**: Test de l'API FastAPI
   - Téléchargement des modèles pré-entraînés
   - Démarrage de l'API avec Uvicorn
   - Test de la route `/` (health check)
   - Test de la route `/predict` avec une requête réelle

**Exécution**:
```bash
# Déclenché automatiquement sur push/PR
# Ou manuellement via GitHub Actions UI
```

### 🚀 Pipeline CD (Continuous Delivery)

Le workflow CD (`cd.yaml`) s'exécute automatiquement après le succès du pipeline CI.
Il permet de préparer les artefacts nécessaires au déploiement, sans déploiement automatique en production.

**Jobs CD** :

1. **build-docker** : Construction des images Docker
   - Téléchargement des données depuis Google Drive
   - Build de l'image API (`flight-delay-api:latest`)
   - Build de l'image Airflow (`flight-delay-airflow:latest`)
   - Sauvegarde des images comme artifacts

2. **prepare-deploy** : Préparation au déploiement
   - Téléchargement des images Docker
   - Chargement des images
   - Images prêtes pour un déploiement manuel ou futur automatisé


### 🎯 Orchestration avec Airflow

Airflow permet d'orchestrer le pipeline MLOps de manière programmée et automatisée.

**DAG principal** (`dvc_pipeline_dag.py`):
```python
# Exécute le pipeline DVC complet
run_dvc_repro = BashOperator(
    task_id="run_dvc_repro",
    bash_command="cd /opt/airflow/project && dvc repro"
)
```

**Configuration Airflow**:
- Image personnalisée avec toutes les dépendances ML
- Montage du projet via volumes Docker
- Exécution manuelle ou programmée du pipeline

**Lancement Airflow**:
```bash
cd airflow

# Initialisation d’Airflow (DB, user admin, permissions)
docker compose up -d --build airflow-init

# Démarrage des services Airflow
docker compose up -d

```

Interface Airflow accessible sur `http://localhost:8080`

### 📦 Configuration Docker Multi-Conteneurs

**docker-compose.yml** configure l'environnement Airflow complet:
- Postgres (base de données Airflow)
- Redis (file d'attente des tâches)
- Airflow Webserver (interface UI)
- Airflow Scheduler (orchestrateur)
- Airflow Worker (exécuteur de tâches)

### 🔐 Gestion des Données

Les données volumineuses sont hébergées sur Google Drive et téléchargées automatiquement dans les workflows CI/CD:
- Données brutes: `df_final.csv`
- Données traitées: `X_train.csv`, `y_train.csv`
- Modèles: `best_model.pkl`, `encoder.pkl`, `feature_names.pkl`

### Livrable
✅ Pipelines CI/CD configurés, orchestration Airflow opérationnelle, déploiement Docker prêt pour l’automatisation

---

## 💻 Installation et utilisation

### Prérequis
- Python 3.11+
- Docker (optionnel)
- DVC
- Git

### Installation locale

```bash
# Cloner le repository
git clone <repo-url>
cd flight-delay-mlops

# Installer les dépendances
pip install -r requirements.txt

# Exécuter le pipeline(remote DVC configuré en local)
dvc repro
```

### Lancement de l'API

**Avec Python**:
```bash
uvicorn api.main:app --reload
```

**Avec Docker**:
```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

L'API sera accessible sur `http://localhost:8000`

### Test de l'API

Une fois l'API lancée, vous pouvez tester une prédiction avec :
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "flight_date": "2025-12-19",
    "airline": 19,
    "flight_number": 3202,
    "origin_airport": 305,
    "dest_airport": 56,
    "scheduled_dep_time": 1420.0,
    "dep_delay": 5.0,
    "taxi_out": 18.0,
    "wheels_off": 1432.0,
    "scheduled_arr_time": 1600.0,
    "scheduled_elapsed_time": 100.0,
    "distance": 450.0,
    "year": 2025,
    "month": 12
  }'
```

Documentation interactive: `http://localhost:8000/docs`

### Lancement d'Airflow

```bash
cd airflow
docker-compose up -d
```

Interface Airflow: `http://localhost:8080`

**Identifiants par défaut**:
- Username: `airflow`
- Password: `airflow`

Pour exécuter le DAG DVC:
1. Accéder à l'interface Airflow
2. Activer le DAG `flight_delay_dvc_pipeline`
3. Déclencher manuellement le DAG ou programmer son exécution selon un schedule approprié

---

## 🛠️ Technologies utilisées

### Data Science & ML
- **pandas** - Manipulation de données
- **scikit-learn** - Preprocessing et métriques
- **XGBoost, LightGBM, CatBoost** - Algorithmes de boosting
- **Random Forest** - Ensemble learning

### MLOps
- **DVC** - Versioning des données et des modèles
- **MLflow** - Tracking des expériences
- **Docker** - Conteneurisation
- **FastAPI** - API REST
- **Uvicorn** - Serveur ASGI
- **Evidently** - Monitoring de la qualité du modèle
- **Airflow** - Orchestration des workflows
- **GitHub Actions** - CI/CD automation

### Versioning & Collaboration
- **Git** - Versioning du code
- **GitHub** - Hébergement du repository
- **GitHub Actions** - CI/CD pipelines

---


## 📈 Résultats

Les modèles ont été évalués sur l’ensemble d’entraînement à l’aide d’une cross-validation 5-fold :

| Modèle        | MAE (Mean) | R² (Mean) | MAE (Std) |
|---------------|------------|-----------|-----------|
| XGBoost       | ~6.18      | ~0.893    | ~0.0085  |
| LightGBM      | ~6.35      | ~0.887    | ~0.0081  |
| CatBoost      | ~6.32      | ~0.883    | ~0.0056  |
| Random Forest | ~6.82      | ~0.874    | ~0.0103  |

Le modèle **XGBoost**, ayant obtenu les meilleures performances en cross-validation, a ensuite été évalué sur l’ensemble de test indépendant :
### 🧪 Résultats sur le jeu de test (XGBoost)

- **MAE** : 6.31  
- **RMSE** : 79.15  
- **R²** : 0.89  

---


## 🔮 Évolutions futures

- Déploiement du projet sur le cloud (AWS, GCP ou Azure)
- Orchestration avec Kubernetes pour plus de scalabilité
- Monitoring en production avec Prometheus et Grafana
- Développement d’une interface web pour les utilisateurs finaux
- Tests de charge et optimisation des performances du modèle et de l’API
