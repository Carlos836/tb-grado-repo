# -*- coding: utf-8 -*-
"""
Execute Basel Committee Traffic Light Analysis with Already Trained Models
Based on BCBS WP14 methodology
"""

# Load the Basel Traffic Light functions
exec(open('notebooks/basel_traffic_light.py').read())

# Function to get already trained models from notebook
def get_trained_models_from_notebook():
    """
    Get the already trained models from the notebook
    This assumes the models are stored in a dictionary called 'trained_models'
    """
    try:
        # Check if trained_models exists in the global namespace
        if 'trained_models' in globals():
            return trained_models
        else:
            print("❌ Variable 'trained_models' not found in global namespace")
            print("Available variables:", [var for var in globals().keys() if not var.startswith('_')])
            return None
    except Exception as e:
        print(f"❌ Error getting trained models: {e}")
        return None

# Function to create models from best parameters if needed
def create_models_from_best_params():
    """
    Create models using the best parameters from optimization
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        
        models = {}
        
        # Check if optimizer exists and has best parameters
        if 'optimizer' in globals() and hasattr(optimizer, 'best_params'):
            best_params = optimizer.best_params
            
            # XGBoost
            if 'xgboost' in best_params:
                models['XGBoost'] = xgb.XGBClassifier(**best_params['xgboost'], random_state=RANDOM_STATE)
            
            # LightGBM
            if 'lightgbm' in best_params:
                models['LightGBM'] = lgb.LGBMClassifier(**best_params['lightgbm'], random_state=RANDOM_STATE)
            
            # CatBoost
            if 'catboost' in best_params:
                models['CatBoost'] = CatBoostClassifier(**best_params['catboost'], random_state=RANDOM_STATE, verbose=False)
            
            # Random Forest
            if 'random_forest' in best_params:
                models['RandomForest'] = RandomForestClassifier(**best_params['random_forest'], random_state=RANDOM_STATE)
            
            # Logistic Regression
            if 'logistic_regression' in best_params:
                models['LogisticRegression'] = LogisticRegression(**best_params['logistic_regression'], random_state=RANDOM_STATE)
        
        return models
        
    except Exception as e:
        print(f"❌ Error creating models from best parameters: {e}")
        return None

# Main execution
print("🔍 Buscando modelos ya entrenados...")

# Try to get already trained models
trained_models = get_trained_models_from_notebook()

if trained_models is None or len(trained_models) == 0:
    print("⚠️ No se encontraron modelos entrenados, intentando crear desde mejores parámetros...")
    trained_models = create_models_from_best_params()

if trained_models is None or len(trained_models) == 0:
    print("❌ No se pudieron obtener modelos para evaluar")
    print("Asegúrate de que:")
    print("1. Los modelos estén entrenados y guardados en 'trained_models'")
    print("2. O que 'optimizer' tenga los mejores parámetros disponibles")
else:
    print(f"✅ Se encontraron {len(trained_models)} modelos para evaluar:")
    for model_name in trained_models.keys():
        print(f"   - {model_name}")
    
    print("\n" + "="*100)
    print("ANALIZANDO TRAFFIC LIGHT CON METODOLOGIA BASEL COMMITTEE")
    print("Basado en: Studies on the Validation of Internal Rating Systems (BCBS WP14)")
    print("Referencia: Tasche, Dirk (2003), A traffic lights approach to PD validation")
    print("="*100)
    
    # Analyze Traffic Light following Basel Committee methodology
    basel_results = analyze_basel_traffic_light(trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout)
    
    # Display results
    display_basel_traffic_light_tables(basel_results)
    
    print("\n" + "="*100)
    print("ANALISIS COMPLETADO - METODOLOGIA BASEL COMMITTEE")
    print("="*100)
