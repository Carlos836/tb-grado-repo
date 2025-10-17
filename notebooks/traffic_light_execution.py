# -*- coding: utf-8 -*-
"""
Traffic Light Statistical Execution
Execute this file to run the statistical Traffic Light analysis
"""

# Load the statistical functions
exec(open('traffic_light_fix.py').read())

# Evaluate models with statistical Traffic Light
print('Evaluating models with statistical Traffic Light...')
results_statistical = evaluate_models_statistical(
    trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout
)

print('Statistical evaluation completed')
