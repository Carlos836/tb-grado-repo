# -*- coding: utf-8 -*-
"""
Train Models with Best Hyperparameters on Training Set
"""

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_models_with_best_params():
    """
    Train all models with their best hyperparameters on training set
    """
    print("ENTRENANDO MODELOS CON MEJORES HIPERPARAMETROS...")
    print("="*60)
    
    # Check if optimizer exists
    if 'optimizer' not in globals():
        print("Variable 'optimizer' not found")
        print("Make sure to run the optimization first")
        return None
    
    # Check if we have the training data
    required_vars = ['X_train', 'y_train']
    missing_vars = [var for var in required_vars if var not in globals()]
    
    if missing_vars:
        print(f"Variables faltantes: {missing_vars}")
        return None
    
    print(f"Entrenando con {X_train.shape[0]} muestras")
    print(f"Variable objetivo: {y_train.value_counts().to_dict()}")
    print()
    
    # Dictionary to store trained models
    trained_models = {}
    
    # Get best parameters from optimizer
    best_params = optimizer.best_params
    print(f"Mejores parametros encontrados:")
    for model_name, params in best_params.items():
        print(f"   {model_name}: {len(params)} parametros")
    print()
    
    # Train XGBoost
    if 'xgboost' in best_params:
        print("Entrenando XGBoost...")
        start_time = time.time()
        
        try:
            xgb_model = xgb.XGBClassifier(
                **best_params['xgboost'],
                random_state=RANDOM_STATE
            )
            xgb_model.fit(X_train, y_train)
            trained_models['XGBoost'] = xgb_model
            
            end_time = time.time()
            print(f"XGBoost entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['xgboost']:.4f}")
            
        except Exception as e:
            print(f"Error entrenando XGBoost: {e}")
    
    # Train LightGBM
    if 'lightgbm' in best_params:
        print("Entrenando LightGBM...")
        start_time = time.time()
        
        try:
            lgb_model = lgb.LGBMClassifier(
                **best_params['lightgbm'],
                random_state=RANDOM_STATE
            )
            lgb_model.fit(X_train, y_train)
            trained_models['LightGBM'] = lgb_model
            
            end_time = time.time()
            print(f"LightGBM entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['lightgbm']:.4f}")
            
        except Exception as e:
            print(f"Error entrenando LightGBM: {e}")
    
    # Train CatBoost
    if 'catboost' in best_params:
        print("Entrenando CatBoost...")
        start_time = time.time()
        
        try:
            cat_model = CatBoostClassifier(
                **best_params['catboost'],
                random_state=RANDOM_STATE,
                verbose=False
            )
            cat_model.fit(X_train, y_train)
            trained_models['CatBoost'] = cat_model
            
            end_time = time.time()
            print(f"CatBoost entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['catboost']:.4f}")
            
        except Exception as e:
            print(f"Error entrenando CatBoost: {e}")
    
    # Train Random Forest
    if 'random_forest' in best_params:
        print("Entrenando Random Forest...")
        start_time = time.time()
        
        try:
            rf_model = RandomForestClassifier(
                **best_params['random_forest'],
                random_state=RANDOM_STATE
            )
            rf_model.fit(X_train, y_train)
            trained_models['RandomForest'] = rf_model
            
            end_time = time.time()
            print(f"Random Forest entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['random_forest']:.4f}")
            
        except Exception as e:
            print(f"Error entrenando Random Forest: {e}")
    
    # Train Logistic Regression
    if 'logistic_regression' in best_params:
        print("Entrenando Logistic Regression...")
        start_time = time.time()
        
        try:
            lr_model = LogisticRegression(
                **best_params['logistic_regression'],
                random_state=RANDOM_STATE
            )
            lr_model.fit(X_train, y_train)
            trained_models['LogisticRegression'] = lr_model
            
            end_time = time.time()
            print(f"Logistic Regression entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['logistic_regression']:.4f}")
            
        except Exception as e:
            print(f"Error entrenando Logistic Regression: {e}")
    
    print(f"\nEntrenamiento completado!")
    print(f"Modelos entrenados: {list(trained_models.keys())}")
    
    return trained_models

def display_model_summary(trained_models):
    """
    Display summary of trained models
    """
    if not trained_models:
        print("No hay modelos entrenados para mostrar")
        return
    
    print("\n" + "="*80)
    print("RESUMEN DE MODELOS ENTRENADOS")
    print("="*80)
    
    for model_name, model in trained_models.items():
        print(f"\n{model_name}:")
        print(f"   Tipo: {type(model).__name__}")
        print(f"   Parametros: {len(model.get_params())} parametros")
        
        # Show some key parameters
        if hasattr(model, 'n_estimators'):
            print(f"   n_estimators: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"   max_depth: {model.max_depth}")
        if hasattr(model, 'learning_rate'):
            print(f"   learning_rate: {model.learning_rate}")
        if hasattr(model, 'C'):
            print(f"   C: {model.C}")

# Execute training
print("INICIANDO ENTRENAMIENTO DE MODELOS...")
print("="*60)

trained_models = train_models_with_best_params()

if trained_models:
    display_model_summary(trained_models)
    
    print(f"\nLos modelos estan listos para evaluacion!")
    print(f"Variable 'trained_models' disponible con {len(trained_models)} modelos")
else:
    print("No se pudieron entrenar los modelos")
    print("Verifica que:")
    print("1. Se haya ejecutado la optimizacion con Optuna")
    print("2. Exista la variable 'optimizer' con mejores parametros")
    print("3. Existan las variables 'X_train' y 'y_train'")