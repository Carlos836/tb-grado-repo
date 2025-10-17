# -*- coding: utf-8 -*-
"""
Detailed Traffic Light Analysis with Probability Ranges
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom

def binomial_test(k, n, p, alternative='two-sided'):
    """
    Custom binomial test implementation
    """
    if alternative == 'two-sided':
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

def create_detailed_traffic_light_table(y_true, y_pred_proba, model_name, dataset_name):
    """
    Create detailed Traffic Light table with probability ranges
    """
    # Create DataFrame
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred_proba
    })
    
    # Define probability ranges
    ranges = [
        (0.00, 0.10), (0.11, 0.20), (0.21, 0.30), (0.31, 0.40), (0.41, 0.50),
        (0.51, 0.60), (0.61, 0.70), (0.71, 0.80), (0.81, 0.90), (0.91, 1.00)
    ]
    
    # Create range column
    df['range'] = pd.cut(df['predicted'], 
                        bins=[0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                        labels=['0.00-0.10', '0.11-0.20', '0.21-0.30', '0.31-0.40', '0.41-0.50',
                               '0.51-0.60', '0.61-0.70', '0.71-0.80', '0.81-0.90', '0.91-1.00'],
                        include_lowest=True)
    
    # Calculate metrics per range
    results = []
    for i, (min_prob, max_prob) in enumerate(ranges):
        range_label = f"{min_prob:.2f}-{max_prob:.2f}"
        range_data = df[df['range'] == range_label]
        
        if len(range_data) > 0:
            total_records = len(range_data)
            count_1 = range_data['actual'].sum()
            count_0 = total_records - count_1
            pct_0 = (count_0 / total_records) * 100
            pct_1 = (count_1 / total_records) * 100
            actual_rate = count_1 / total_records
            
            # Midpoint of the range
            midpoint = (min_prob + max_prob) / 2
            
            # Difference between actual rate and midpoint
            difference = abs(actual_rate - midpoint)
            
            # Statistical test
            p_value = binomial_test(count_1, total_records, midpoint, alternative='two-sided')
            
            # Determine color
            if p_value > 0.05:
                color = 'Verde'
            elif p_value > 0.025:
                color = 'Amarillo'
            else:
                color = 'Rojo'
            
            results.append({
                'Rango_Probabilidad': range_label,
                'Total_Registros': total_records,
                'Cantidad_1': count_1,
                'Cantidad_0': count_0,
                'Porcentaje_0': round(pct_0, 2),
                'Porcentaje_1': round(pct_1, 2),
                'Tasa_Real': round(actual_rate, 4),
                'Punto_Medio_Rango': round(midpoint, 2),
                'Diferencia': round(difference, 4),
                'P_Value': round(p_value, 4),
                'Color': color
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Add summary row
    total_records = result_df['Total_Registros'].sum()
    total_1 = result_df['Cantidad_1'].sum()
    total_0 = result_df['Cantidad_0'].sum()
    overall_rate = total_1 / total_records if total_records > 0 else 0
    
    summary_row = pd.DataFrame({
        'Rango_Probabilidad': ['TOTAL'],
        'Total_Registros': [total_records],
        'Cantidad_1': [total_1],
        'Cantidad_0': [total_0],
        'Porcentaje_0': [round((total_0 / total_records) * 100, 2)],
        'Porcentaje_1': [round((total_1 / total_records) * 100, 2)],
        'Tasa_Real': [round(overall_rate, 4)],
        'Punto_Medio_Rango': ['-'],
        'Diferencia': ['-'],
        'P_Value': ['-'],
        'Color': ['-']
    })
    
    result_df = pd.concat([result_df, summary_row], ignore_index=True)
    
    return result_df

def analyze_xgboost_traffic_light(trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
    """
    Analyze XGBoost Traffic Light for all datasets
    """
    if 'XGBoost' not in trained_models:
        print("XGBoost model not found in trained_models")
        return None
    
    xgb_model = trained_models['XGBoost']
    
    # Get predictions
    train_proba = xgb_model.predict_proba(X_train)[:, 1]
    test_proba = xgb_model.predict_proba(X_test)[:, 1]
    holdout_proba = xgb_model.predict_proba(X_holdout)[:, 1]
    
    # Create detailed tables
    train_table = create_detailed_traffic_light_table(y_train, train_proba, 'XGBoost', 'Train')
    test_table = create_detailed_traffic_light_table(y_test, test_proba, 'XGBoost', 'Test')
    holdout_table = create_detailed_traffic_light_table(y_holdout, holdout_proba, 'XGBoost', 'Holdout')
    
    return {
        'train': train_table,
        'test': test_table,
        'holdout': holdout_table
    }

def display_traffic_light_tables(xgb_results):
    """
    Display Traffic Light tables for XGBoost
    """
    if xgb_results is None:
        print("No XGBoost results to display")
        return
    
    print("=" * 80)
    print("TRAFFIC LIGHT DETALLADO - XGBOOST")
    print("=" * 80)
    
    for dataset_name, table in xgb_results.items():
        print(f"\n{dataset_name.upper()} SET:")
        print("-" * 80)
        print(table.to_string(index=False))
        print("-" * 80)
        
        # Summary statistics
        green_count = len(table[table['Color'] == 'Verde'])
        yellow_count = len(table[table['Color'] == 'Amarillo'])
        red_count = len(table[table['Color'] == 'Rojo'])
        total_ranges = green_count + yellow_count + red_count
        
        if total_ranges > 0:
            print(f"Resumen {dataset_name.upper()}:")
            print(f"  Verde: {green_count}/{total_ranges} ({green_count/total_ranges*100:.1f}%)")
            print(f"  Amarillo: {yellow_count}/{total_ranges} ({yellow_count/total_ranges*100:.1f}%)")
            print(f"  Rojo: {red_count}/{total_ranges} ({red_count/total_ranges*100:.1f}%)")
        print()

print("Detailed Traffic Light functions loaded successfully")
