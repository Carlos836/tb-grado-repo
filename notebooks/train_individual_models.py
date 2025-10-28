# -*- coding: utf-8 -*-
"""
Train Individual Models with Best Hyperparameters
Each model will be available as a separate variable for individual predictions
"""

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_individual_models():
    """
    Train each model individually with best hyperparameters
    """
    print("ENTRENANDO MODELOS INDIVIDUALES CON MEJORES HIPERPARAMETROS...")
    print("="*70)
    
    # Check if optimizer exists
    if 'optimizer' not in globals():
        print("Variable 'optimizer' not found")
        print("Make sure to run the optimization first")
        return False
    
    # Check if we have the training data
    required_vars = ['X_train', 'y_train']
    missing_vars = [var for var in required_vars if var not in globals()]
    
    if missing_vars:
        print(f"Variables faltantes: {missing_vars}")
        return False
    
    print(f"Entrenando con {X_train.shape[0]} muestras")
    print(f"Variable objetivo: {y_train.value_counts().to_dict()}")
    print()
    
    # Get best parameters from optimizer
    best_params = optimizer.best_params
    print(f"Mejores parametros encontrados:")
    for model_name, params in best_params.items():
        print(f"   {model_name}: {len(params)} parametros")
    print()
    
    success_count = 0
    
    # Train XGBoost
    if 'xgboost' in best_params:
        print("Entrenando XGBoost...")
        start_time = time.time()
        
        try:
            xgbopt = xgb.XGBClassifier(
                **best_params['xgboost'],
                random_state=RANDOM_STATE
            )
            xgbopt.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"XGBoost entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['xgboost']:.4f}")
            print(f"   Variable: 'xgbopt'")
            success_count += 1
            
        except Exception as e:
            print(f"Error entrenando XGBoost: {e}")
    
    # Train LightGBM
    if 'lightgbm' in best_params:
        print("\nEntrenando LightGBM...")
        start_time = time.time()
        
        try:
            lgbopt = lgb.LGBMClassifier(
                **best_params['lightgbm'],
                random_state=RANDOM_STATE
            )
            lgbopt.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"LightGBM entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['lightgbm']:.4f}")
            print(f"   Variable: 'lgbopt'")
            success_count += 1
            
        except Exception as e:
            print(f"Error entrenando LightGBM: {e}")
    
    # Train CatBoost
    if 'catboost' in best_params:
        print("\nEntrenando CatBoost...")
        start_time = time.time()
        
        try:
            catopt = CatBoostClassifier(
                **best_params['catboost'],
                random_state=RANDOM_STATE,
                verbose=False
            )
            catopt.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"CatBoost entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['catboost']:.4f}")
            print(f"   Variable: 'catopt'")
            success_count += 1
            
        except Exception as e:
            print(f"Error entrenando CatBoost: {e}")
    
    # Train Random Forest
    if 'random_forest' in best_params:
        print("\nEntrenando Random Forest...")
        start_time = time.time()
        
        try:
            rfopt = RandomForestClassifier(
                **best_params['random_forest'],
                random_state=RANDOM_STATE
            )
            rfopt.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"Random Forest entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['random_forest']:.4f}")
            print(f"   Variable: 'rfopt'")
            success_count += 1
            
        except Exception as e:
            print(f"Error entrenando Random Forest: {e}")
    
    # Train Logistic Regression
    if 'logistic_regression' in best_params:
        print("\nEntrenando Logistic Regression...")
        start_time = time.time()
        
        try:
            lropt = LogisticRegression(
                **best_params['logistic_regression'],
                random_state=RANDOM_STATE
            )
            lropt.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"Logistic Regression entrenado en {end_time - start_time:.1f} segundos")
            print(f"   Mejor CV Score: {optimizer.best_scores['logistic_regression']:.4f}")
            print(f"   Variable: 'lropt'")
            success_count += 1
            
        except Exception as e:
            print(f"Error entrenando Logistic Regression: {e}")
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO COMPLETADO!")
    print(f"Modelos entrenados exitosamente: {success_count}")
    print(f"{'='*70}")
    
    return success_count > 0

def show_available_models():
    """
    Show which models are available for predictions
    """
    print("\nMODELOS DISPONIBLES PARA PREDICCIONES:")
    print("-" * 50)
    
    available_models = []
    
    if 'xgbopt' in globals():
        print("xgbopt - XGBoost optimizado")
        available_models.append('xgbopt')
    
    if 'lgbopt' in globals():
        print("lgbopt - LightGBM optimizado")
        available_models.append('lgbopt')
    
    if 'catopt' in globals():
        print("catopt - CatBoost optimizado")
        available_models.append('catopt')
    
    if 'rfopt' in globals():
        print("rfopt - Random Forest optimizado")
        available_models.append('rfopt')
    
    if 'lropt' in globals():
        print("lropt - Logistic Regression optimizado")
        available_models.append('lropt')
    
    if not available_models:
        print("No hay modelos disponibles")
        return
    
    print(f"\nTotal de modelos disponibles: {len(available_models)}")
    
    # Show example predictions
    print(f"\nEJEMPLO DE PREDICCIONES:")
    print("-" * 30)
    if 'X_test' in globals() and available_models:
        first_model = available_models[0]
        model = globals()[first_model]
        try:
            predictions = model.predict(X_test[:5])
            probabilities = model.predict_proba(X_test[:5])[:, 1]
            print(f"Usando {first_model}:")
            print(f"Predicciones: {predictions}")
            print(f"Probabilidades: {probabilities}")
        except Exception as e:
            print(f"Error en prediccion de ejemplo: {e}")

# Execute training
print("INICIANDO ENTRENAMIENTO DE MODELOS INDIVIDUALES...")
print("="*70)

success = train_individual_models()

if success:
    show_available_models()
    
    print(f"\nLos modelos estan listos para predicciones individuales!")
    print(f"Variables disponibles:")
    print(f"   - xgbopt: XGBoost optimizado")
    print(f"   - lgbopt: LightGBM optimizado") 
    print(f"   - catopt: CatBoost optimizado")
    print(f"   - rfopt: Random Forest optimizado")
    print(f"   - lropt: Logistic Regression optimizado")
    print(f"\nEjemplo de uso:")
    print(f"   predictions = xgbopt.predict(X_test)")
    print(f"   probabilities = catopt.predict_proba(X_test)")
else:
    print("No se pudieron entrenar los modelos")
    print("Verifica que:")
    print("1. Se haya ejecutado la optimizacion con Optuna")
    print("2. Exista la variable 'optimizer' con mejores parametros")
    print("3. Existan las variables 'X_train' y 'y_train'")