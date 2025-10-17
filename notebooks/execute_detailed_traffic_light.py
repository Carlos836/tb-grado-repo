# -*- coding: utf-8 -*-
"""
Execute Detailed Traffic Light Analysis for XGBoost
"""

# Load the detailed traffic light functions
exec(open('detailed_traffic_light.py').read())

# Analyze XGBoost Traffic Light
print("Analyzing XGBoost Traffic Light with detailed probability ranges...")
xgb_results = analyze_xgboost_traffic_light(trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout)

# Display results
display_traffic_light_tables(xgb_results)

print("Detailed Traffic Light analysis completed!")
