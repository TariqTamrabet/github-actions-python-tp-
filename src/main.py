#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:09:29 2025

@author: ttamrabe
"""
# Importation de la fonction pour charger le jeu de données California Housing
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
mlflow.set_experiment("House Price Prediction - By Tariq TAMRABET")
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt




#Mission 1 : Exploration et préparation des données
# Question 1 : Charger les données et effectuer une analyse exploratoire.

# Chargement des données sous forme de DataFrame grâce à as_frame=True
housing = fetch_california_housing(as_frame=True)

# Affichage des 10 premières lignes du DataFrame pour un aperçu rapide des données
housing.frame.head(10)

# Informations sur les données, y compris les types de colonnes, les valeurs manquantes, etc.
housing.frame.info()

# Statistiques descriptives pour les colonnes numériques (moyenne, médiane, etc.)
housing.frame.describe()

# Création d'histogrammes pour chaque colonne numérique afin de visualiser les distributions
housing.frame.hist(bins=50, figsize=(20, 15))




# Question 2 : Identifier les valeurs manquantes ou aberrantes.

# Vérifier les valeurs manquantes en utilisant la fonction isnull
missing_values = housing.data.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Afficher les lignes contenant des valeurs manquantes
rows_with_missing = housing.data[housing.data.isnull().any(axis=1)]
print("Lignes contenant des valeurs manquantes :\n", rows_with_missing)




# Fonction pour détecter les valeurs aberrantes selon l'IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # Premier quartile
    Q3 = df[column].quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1  # Intervalle interquartile
    lower_bound = Q1 - 1.5 * IQR  # Limite inférieure
    upper_bound = Q3 + 1.5 * IQR  # Limite supérieure

    # Identifie les lignes contenant des valeurs aberrantes
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Appliquer la fonction à chaque colonne numérique
numerical_cols = housing.frame.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(housing.frame, col)
    print(f"Colonne : {col}")
    print(f"Valeurs aberrantes détectées : {len(outliers)}")
    print(f"Limite inférieure : {lower}, Limite supérieure : {upper}")
    print(outliers.head())  # Affiche les premières valeurs aberrantes
    print("-" * 50)







# Question 3 : Normaliser ou standardiser les données si nécessaire.
# 1) Charger les données
housing = fetch_california_housing(as_frame=True).frame  # Extraction complète en DataFrame

# 2) Définir la fonction pour détecter et gérer les valeurs aberrantes selon l'IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # Premier quartile
    Q3 = df[column].quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1                   # Intervalle interquartile
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtrage des données
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 3) Supprimer les valeurs aberrantes pour chaque colonne (y compris la cible)
for col in housing.columns:
    housing = remove_outliers_iqr(housing, col)

# 4) Séparer la cible (target) du reste des features
X = housing.drop("MedHouseVal", axis=1)  # 8 colonnes
y = housing["MedHouseVal"].copy()       # 1 colonne (la target)

# 5) Créer un pipeline pour imputation et standardisation
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # Imputation des valeurs manquantes
    ('std_scaler', StandardScaler()),               # Standardisation
])

# 6) Appliquer le pipeline SUR X (sans la target)
X_num_tr = num_pipeline.fit_transform(X)

# 7) Convertir en DataFrame pour une meilleure lisibilité
housing_num_tr_df = pd.DataFrame(
    X_num_tr,
    columns=X.columns
)

# 8) Afficher un aperçu des données finalisées (hors target)
print("Données standardisées (après gestion des valeurs aberrantes) :")
print(housing_num_tr_df.head())





# Question 4 : Diviser les données en ensembles d’entrainement et de test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher les dimensions des ensembles pour vérification
print(f"Taille de X_train : {X_train.shape}")
print(f"Taille de X_test : {X_test.shape}")
print(f"Taille de y_train : {y_train.shape}")
print(f"Taille de y_test : {y_test.shape}")










# Mission 2 - Modélisation et tracking des expériences
# Question 1 : Créer un modèle de régression linéaire comme baseline.


# Initialiser le modèle
lin_reg = LinearRegression()

# Entraîner le modèle
lin_reg.fit(X, y)

# Faire des prédictions
y_pred = lin_reg.predict(X)

# Calculer les métriques
mse = mean_squared_error(y, y_pred)
mae = np.mean(np.abs(y_pred - y))
rmse = np.sqrt(mse)

# Afficher les résultats
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {lin_reg.score(X, y)}")
print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficients: {lin_reg.coef_}")



# Question 2 : Utiliser des modèles avancés comme Random Forest ou Gradient Boosting.


# Initialisation des modèles
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Entraînement des modèles
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Évaluation des modèles
for model_name, y_pred in zip(["Random Forest", "Gradient Boosting"], [rf_pred, gb_pred]):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Résultats pour {model_name} :")
    print(f"  - RMSE : {rmse:.2f}")
    print(f"  - MAE  : {mae:.2f}")
    print(f"  - R²   : {r2:.2f}")
    print("-" * 40)




# Question 3 : Mettre en place MLflow pour suivre les expériences (métriques comme RMSE, MAE, R²).


# Initialiser les modèles
models = [
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Stocker les performances pour comparer les modèles
best_model = None
best_rmse = float("inf")
best_model_name = ""

# Parcourir les modèles et enregistrer dans MLflow
for model_name, model in models:
    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculer les métriques
        rmse = (mean_squared_error(y_test, y_pred))**0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Enregistrer les hyperparamètres
        mlflow.log_param("n_estimators", 100)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name} enregistré avec RMSE={rmse}, MAE={mae}, R2={r2}")




# Question 4.Enregistrer le meilleur modèle dans le Model Registry de MLflow.

        print(rmse)

         # Comparer les modèles pour trouver le meilleur
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name

# Enregistrer le meilleur modèle dans le Model Registry
if best_model:
    with mlflow.start_run(run_name="Best Model"):
        print(f"Enregistrement du meilleur modèle : {best_model_name} avec RMSE={best_rmse}")
        
        # Ajouter des paramètres et des tags
        mlflow.log_param("Model Type", best_model_name)
        mlflow.log_param("Best RMSE", best_rmse)
        mlflow.set_tag("Best Model", "True")
        mlflow.set_tag("Description", f"Best model is {best_model_name} with RMSE={best_rmse}")

        # Enregistrer le modèle
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="Best_Model_Registry"  # Nom du modèle dans le Model Registry
        )

        # Ajouter une description au modèle dans le Model Registry
        client = MlflowClient()
        client.update_registered_model(
            name="Best_Model_Registry",
            description=f"Best model is {best_model_name}. Achieved a RMSE of {best_rmse}. This model is suitable for production."
        )
        
        
##############################################################

full_pipeline = Pipeline([
    ('preprocessor', num_pipeline),
    ('model', best_model)
])

# Entraîner le pipeline complet sur les données d'entraînement transformées
full_pipeline.fit(X_train, y_train)

# Sauvegarder le pipeline complet
joblib.dump(full_pipeline, "model_pipeline.pkl")

##############################################################

        
# Mission 3 : Analyse des features
# Question 1 : 1. Calculer les importances globales des features (à l’aide de SHAP ou des features importances).

print("Question 1 : Calculer les importances globales des features (à l’aide de SHAP ou des features importances)")

# Par exemple, on prend 1000 lignes au maximum
max_samples = 1000
if len(X_test) > max_samples:
    X_shap = X_test.sample(n=max_samples, random_state=42)
else:
    X_shap = X_test

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)

# Calcul de l'importance moyenne
feature_importances = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean SHAP Value": feature_importances
}).sort_values(by="Mean SHAP Value", ascending=False)

# Affichage rapide (exemple : top 10 features les plus importantes)
top_n = 10
subset = shap_importance_df.head(top_n)

plt.figure(figsize=(8, 6))
plt.barh(subset["Feature"], subset["Mean SHAP Value"])
plt.gca().invert_yaxis()
plt.xlabel("Mean SHAP Value")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances (SHAP)")
plt.show()




print("Question 2 : Analyser l’impact local pour des exemples individuels (à l’aide de SHAP)")

# Ici Choisir une instance à expliquer
i = 0
X_instance = X_test.iloc[[i]]

print("Instance à expliquer :")
print(X_instance)

# 4) Calcul local des valeurs SHAP pour cette instance
shap_values_instance = explainer.shap_values(X_instance)

# 5) Visualisation (force plot) dans un notebook
shap.initjs()
shap.force_plot(
    explainer.expected_value, 
    shap_values_instance,
    X_instance
)



# Mission 4 - Mise en production
# Question 1 : Créer une API avec FastAPI pour recevoir des données en entrée et renvoyer une prédiction.


# Enregistrer le meilleur modèle
joblib.dump(best_model, "model.pkl")




        
