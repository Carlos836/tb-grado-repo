# -*- coding: utf-8 -*-
# Traffic Light Statistical Fix

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom

def binomial_test(k, n, p, alternative='two-sided'):
    """
    Custom binomial test implementation
    """
    if alternative == 'two-sided':
        # Two-sided test: p-value is 2 * min(P(X <= k), P(X >= k))
        p_lower = binom.cdf(k, n, p)
        p_upper = 1 - binom.cdf(k-1, n, p)
        p_value = 2 * min(p_lower, p_upper)
    elif alternative == 'greater':
        p_value = 1 - binom.cdf(k-1, n, p)
    elif alternative == 'less':
        p_value = binom.cdf(k, n, p)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return p_value

def calculate_traffic_light_statistical(y_true, y_pred_proba, n_groups=10, alpha=0.05):
    """
    Calculate Traffic Light using statistical tests
    """
    # Create DataFrame with data
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred_proba
    })
    
    # Create deciles
    df['decile'] = pd.qcut(df['predicted'], q=n_groups, labels=False, duplicates='drop') + 1
    
    # Calculate metrics per decile
    group_stats = []
    for decile in range(1, n_groups + 1):
        decile_data = df[df['decile'] == decile]
        if len(decile_data) > 0:
            actual_rate = decile_data['actual'].mean()
            predicted_rate = decile_data['predicted'].mean()
            n_obs = len(decile_data)
            
            # Binomial test using custom implementation
            n_success = int(actual_rate * n_obs)
            p_value = binomial_test(n_success, n_obs, predicted_rate, alternative='two-sided')
            
            # Determine color based on p-value
            if p_value > alpha:
                color = 'green'
            elif p_value > alpha/2:
                color = 'yellow'
            else:
                color = 'red'
            
            # Confidence interval
            ci_lower, ci_upper = stats.binom.interval(0.95, n_obs, actual_rate)
            ci_lower_rate = ci_lower / n_obs
            ci_upper_rate = ci_upper / n_obs
            
            group_stats.append({
                'decile': decile,
                'actual_rate': actual_rate,
                'predicted_rate': predicted_rate,
                'difference': abs(actual_rate - predicted_rate),
                'p_value': p_value,
                'color': color,
                'size': n_obs,
                'min_prob': decile_data['predicted'].min(),
                'max_prob': decile_data['predicted'].max(),
                'avg_prob': decile_data['predicted'].mean(),
                'ci_lower': ci_lower_rate,
                'ci_upper': ci_upper_rate,
                'significant': p_value < alpha
            })
    
    # Calculate general statistics
    colors = [stat['color'] for stat in group_stats]
    green_pct = colors.count('green') / len(colors) * 100
    yellow_pct = colors.count('yellow') / len(colors) * 100
    red_pct = colors.count('red') / len(colors) * 100
    
    return {
        'group_stats': group_stats,
        'green_percentage': green_pct,
        'yellow_percentage': yellow_pct,
        'red_percentage': red_pct,
        'total_groups': len(group_stats),
        'alpha': alpha
    }

def evaluate_models_statistical(models, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
    """
    Evaluate models using statistical Traffic Light
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} with statistical Traffic Light...")
        
        # Predictions on each set
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        holdout_proba = model.predict_proba(X_holdout)[:, 1]
        
        # Evaluate on each set
        train_metrics = calculate_traffic_light_statistical(y_train, train_proba)
        test_metrics = calculate_traffic_light_statistical(y_test, test_proba)
        holdout_metrics = calculate_traffic_light_statistical(y_holdout, holdout_proba)
        
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'holdout': holdout_metrics
        }
        
        # Show results
        print(f"   Train  - Green: {train_metrics['green_percentage']:.1f}%, Yellow: {train_metrics['yellow_percentage']:.1f}%, Red: {train_metrics['red_percentage']:.1f}%")
        print(f"   Test   - Green: {test_metrics['green_percentage']:.1f}%, Yellow: {test_metrics['yellow_percentage']:.1f}%, Red: {test_metrics['red_percentage']:.1f}%")
        print(f"   Holdout - Green: {holdout_metrics['green_percentage']:.1f}%, Yellow: {holdout_metrics['yellow_percentage']:.1f}%, Red: {holdout_metrics['red_percentage']:.1f}%")
    
    return results

print("Statistical Traffic Light functions loaded successfully")
