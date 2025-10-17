# -*- coding: utf-8 -*-
"""
Basel Committee Traffic Light Implementation
Based on: Studies on the Validation of Internal Rating Systems (BCBS WP14)
Reference: Tasche, Dirk (2003), A traffic lights approach to PD validation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom

def basel_binomial_test(k, n, p, alpha=0.05):
    """
    Basel Committee binomial test implementation
    Based on BCBS WP14 methodology
    """
    # Two-sided binomial test
    p_lower = binom.cdf(k, n, p)
    p_upper = 1 - binom.cdf(k-1, n, p)
    p_value = 2 * min(p_lower, p_upper)
    
    return p_value

def create_basel_traffic_light_table(y_true, y_pred_proba, model_name, dataset_name):
    """
    Create Basel Committee Traffic Light table
    Following BCBS WP14 methodology
    """
    # Create DataFrame
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred_proba
    })
    
    # Define probability ranges (Basel methodology)
    # Using deciles for probability of default ranges
    df['decile'] = pd.qcut(df['predicted'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate metrics per decile
    results = []
    for decile in range(1, 11):
        decile_data = df[df['decile'] == decile]
        
        if len(decile_data) > 0:
            total_records = len(decile_data)
            count_1 = decile_data['actual'].sum()
            count_0 = total_records - count_1
            pct_0 = (count_0 / total_records) * 100
            pct_1 = (count_1 / total_records) * 100
            actual_rate = count_1 / total_records
            
            # Predicted PD (average of the decile)
            predicted_pd = decile_data['predicted'].mean()
            
            # Difference between actual and predicted
            difference = abs(actual_rate - predicted_pd)
            
            # Basel binomial test (two-sided)
            p_value = basel_binomial_test(count_1, total_records, predicted_pd, alpha=0.05)
            
            # Basel Traffic Light colors (5 colors system)
            # Determine if overestimation or underestimation
            if actual_rate > predicted_pd:
                # Actual rate higher than predicted (underestimation)
                if p_value > 0.05:
                    color = 'Verde'
                    interpretation = 'Preciso'
                elif p_value > 0.01:
                    color = 'Amarillo'
                    interpretation = 'Subestimacion leve'
                else:
                    color = 'Rojo'
                    interpretation = 'Subestimacion fuerte'
            else:
                # Actual rate lower than predicted (overestimation)
                if p_value > 0.05:
                    color = 'Verde'
                    interpretation = 'Preciso'
                elif p_value > 0.01:
                    color = 'Azul Claro'
                    interpretation = 'Sobrestimacion leve'
                else:
                    color = 'Azul Marino'
                    interpretation = 'Sobrestimacion fuerte'
            
            # Confidence interval for actual rate (95%)
            ci_lower, ci_upper = stats.binom.interval(0.95, total_records, actual_rate)
            ci_lower_rate = ci_lower / total_records
            ci_upper_rate = ci_upper / total_records
            
            # Probability range for this decile
            min_prob = decile_data['predicted'].min()
            max_prob = decile_data['predicted'].max()
            
            results.append({
                'Decil': decile,
                'Rango_Probabilidad': f"{min_prob:.3f}-{max_prob:.3f}",
                'Total_Registros': total_records,
                'Cantidad_1': count_1,
                'Cantidad_0': count_0,
                'Porcentaje_0': round(pct_0, 2),
                'Porcentaje_1': round(pct_1, 2),
                'Tasa_Real': round(actual_rate, 4),
                'PD_Predicha': round(predicted_pd, 4),
                'Diferencia': round(difference, 4),
                'P_Value': round(p_value, 4),
                'Color': color,
                'Interpretacion': interpretation,
                'CI_Lower': round(ci_lower_rate, 4),
                'CI_Upper': round(ci_upper_rate, 4)
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Add summary statistics
    total_records = result_df['Total_Registros'].sum()
    total_1 = result_df['Cantidad_1'].sum()
    total_0 = result_df['Cantidad_0'].sum()
    overall_rate = total_1 / total_records if total_records > 0 else 0
    
    # Count colors (5 color system)
    green_count = len(result_df[result_df['Color'] == 'Verde'])
    yellow_count = len(result_df[result_df['Color'] == 'Amarillo'])
    red_count = len(result_df[result_df['Color'] == 'Rojo'])
    light_blue_count = len(result_df[result_df['Color'] == 'Azul Claro'])
    dark_blue_count = len(result_df[result_df['Color'] == 'Azul Marino'])
    
    summary_row = pd.DataFrame({
        'Decil': ['TOTAL'],
        'Rango_Probabilidad': ['-'],
        'Total_Registros': [total_records],
        'Cantidad_1': [total_1],
        'Cantidad_0': [total_0],
        'Porcentaje_0': [round((total_0 / total_records) * 100, 2)],
        'Porcentaje_1': [round((total_1 / total_records) * 100, 2)],
        'Tasa_Real': [round(overall_rate, 4)],
        'PD_Predicha': ['-'],
        'Diferencia': ['-'],
        'P_Value': ['-'],
        'Color': [f'V:{green_count} A:{yellow_count} R:{red_count} AC:{light_blue_count} AM:{dark_blue_count}'],
        'Interpretacion': ['Resumen'],
        'CI_Lower': ['-'],
        'CI_Upper': ['-']
    })
    
    result_df = pd.concat([result_df, summary_row], ignore_index=True)
    
    return result_df

def analyze_basel_traffic_light(trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
    """
    Analyze Traffic Light following Basel Committee methodology
    """
    results = {}
    
    for model_name, model in trained_models.items():
        print(f"Analizando {model_name} con metodologia Basel Committee...")
        
        # Get predictions
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        holdout_proba = model.predict_proba(X_holdout)[:, 1]
        
        # Create Basel tables
        train_table = create_basel_traffic_light_table(y_train, train_proba, model_name, 'Train')
        test_table = create_basel_traffic_light_table(y_test, test_proba, model_name, 'Test')
        holdout_table = create_basel_traffic_light_table(y_holdout, holdout_proba, model_name, 'Holdout')
        
        results[model_name] = {
            'train': train_table,
            'test': test_table,
            'holdout': holdout_table
        }
    
    return results

def display_basel_traffic_light_tables(results):
    """
    Display Basel Committee Traffic Light tables
    """
    print("=" * 100)
    print("TRAFFIC LIGHT - METODOLOGIA BASEL COMMITTEE")
    print("Basado en: Studies on the Validation of Internal Rating Systems (BCBS WP14)")
    print("Referencia: Tasche, Dirk (2003), A traffic lights approach to PD validation")
    print("=" * 100)
    
    for model_name, model_results in results.items():
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        for dataset_name, table in model_results.items():
            print(f"\n{dataset_name.upper()} SET:")
            print("-" * 100)
            
            # Display main table
            display_cols = ['Decil', 'Rango_Probabilidad', 'Total_Registros', 'Cantidad_1', 
                           'Cantidad_0', 'Porcentaje_1', 'Tasa_Real', 'PD_Predicha', 
                           'Diferencia', 'P_Value', 'Color']
            print(table[display_cols].to_string(index=False))
            
            # Summary statistics (5 color system)
            if dataset_name != 'summary':
                green_count = len(table[table['Color'] == 'Verde'])
                yellow_count = len(table[table['Color'] == 'Amarillo'])
                red_count = len(table[table['Color'] == 'Rojo'])
                light_blue_count = len(table[table['Color'] == 'Azul Claro'])
                dark_blue_count = len(table[table['Color'] == 'Azul Marino'])
                total_ranges = green_count + yellow_count + red_count + light_blue_count + dark_blue_count
                
                if total_ranges > 0:
                    print(f"\nResumen {dataset_name.upper()}:")
                    print(f"  Verde: {green_count}/{total_ranges} ({green_count/total_ranges*100:.1f}%)")
                    print(f"  Amarillo: {yellow_count}/{total_ranges} ({yellow_count/total_ranges*100:.1f}%)")
                    print(f"  Rojo: {red_count}/{total_ranges} ({red_count/total_ranges*100:.1f}%)")
                    print(f"  Azul Claro: {light_blue_count}/{total_ranges} ({light_blue_count/total_ranges*100:.1f}%)")
                    print(f"  Azul Marino: {dark_blue_count}/{total_ranges} ({dark_blue_count/total_ranges*100:.1f}%)")
                    
                    # Basel interpretation
                    if green_count >= 7:
                        print(f"  Interpretacion Basel: Modelo APROBADO (>=70% Verde)")
                    elif green_count >= 5:
                        print(f"  Interpretacion Basel: Modelo ACEPTABLE (50-69% Verde)")
                    else:
                        print(f"  Interpretacion Basel: Modelo RECHAZADO (<50% Verde)")
            
            print("-" * 100)

print("Basel Committee Traffic Light functions loaded successfully")
