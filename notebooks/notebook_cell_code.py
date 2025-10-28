# Quick Individual Model Training - Copy this code into a notebook cell

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

print("ENTRENANDO MODELOS INDIVIDUALES...")
print("="*50)

# Check if optimizer exists
if 'optimizer' not in globals():
    print("❌ Variable 'optimizer' not found")
    print("Run the Optuna optimization first!")
else:
    print("✅ Optimizer found")
    
    # Get best parameters
    best_params = optimizer.best_params
    print(f"📋 Found {len(best_params)} optimized models")
    
    # Train XGBoost
    if 'xgboost' in best_params:
        print("\n🔥 Training XGBoost...")
        xgbopt = xgb.XGBClassifier(**best_params['xgboost'], random_state=RANDOM_STATE)
        xgbopt.fit(X_train, y_train)
        print(f"✅ xgbopt ready - CV Score: {optimizer.best_scores['xgboost']:.4f}")
    
    # Train LightGBM
    if 'lightgbm' in best_params:
        print("\n🔥 Training LightGBM...")
        lgbopt = lgb.LGBMClassifier(**best_params['lightgbm'], random_state=RANDOM_STATE)
        lgbopt.fit(X_train, y_train)
        print(f"✅ lgbopt ready - CV Score: {optimizer.best_scores['lightgbm']:.4f}")
    
    # Train CatBoost
    if 'catboost' in best_params:
        print("\n🔥 Training CatBoost...")
        catopt = CatBoostClassifier(**best_params['catboost'], random_state=RANDOM_STATE, verbose=False)
        catopt.fit(X_train, y_train)
        print(f"✅ catopt ready - CV Score: {optimizer.best_scores['catboost']:.4f}")
    
    # Train Random Forest
    if 'random_forest' in best_params:
        print("\n🔥 Training Random Forest...")
        rfopt = RandomForestClassifier(**best_params['random_forest'], random_state=RANDOM_STATE)
        rfopt.fit(X_train, y_train)
        print(f"✅ rfopt ready - CV Score: {optimizer.best_scores['random_forest']:.4f}")
    
    # Train Logistic Regression
    if 'logistic_regression' in best_params:
        print("\n🔥 Training Logistic Regression...")
        lropt = LogisticRegression(**best_params['logistic_regression'], random_state=RANDOM_STATE)
        lropt.fit(X_train, y_train)
        print(f"✅ lropt ready - CV Score: {optimizer.best_scores['logistic_regression']:.4f}")
    
    print(f"\n🎯 All models ready for individual predictions!")
    print(f"📝 Available variables: xgbopt, lgbopt, catopt, rfopt, lropt")
