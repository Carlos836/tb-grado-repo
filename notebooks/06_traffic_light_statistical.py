# -*- coding: utf-8 -*-
# Traffic Light con Pruebas Estadísticas (Método Correcto)
"""
Código para calcular Traffic Light usando pruebas estadísticas
según el paper de rating bancario
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_traffic_light_statistical(y_true, y_pred_proba, n_groups=10, alpha=0.05):
    """
    Calcula Traffic Light usando pruebas estadísticas para determinar
    si la PD del modelo se desvía significativamente de la PD teórica
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        n_groups: Número de deciles
        alpha: Nivel de significancia
    
    Returns:
        Dict con estadísticas de Traffic Light
    """
    # Crear DataFrame con datos
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred_proba
    })
    
    # Crear deciles basados en probabilidades predichas (descendente)
    df['decile'] = pd.qcut(df['predicted'], q=n_groups, labels=False, duplicates='drop') + 1
    
    # Calcular métricas por decil
    group_stats = []
    for decile in range(1, n_groups + 1):
        decile_data = df[df['decile'] == decile]
        if len(decile_data) > 0:
            actual_rate = decile_data['actual'].mean()
            predicted_rate = decile_data['predicted'].mean()
            n_obs = len(decile_data)
            
            # Prueba estadística: Binomial Test
            # H0: La tasa real = tasa predicha
            # H1: La tasa real ≠ tasa predicha
            
            # Calcular estadístico de prueba
            n_success = int(actual_rate * n_obs)
            
            # Prueba binomial
            p_value = stats.binom_test(n_success, n_obs, predicted_rate, alternative='two-sided')
            
            # Determinar color del semáforo basado en p-value
            if p_value > alpha:
                color = 'green'  # No hay evidencia de desviación significativa
            elif p_value > alpha/2:
                color = 'yellow'  # Desviación moderada
            else:
                color = 'red'  # Desviación significativa
            
            # Calcular intervalo de confianza para la tasa real
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
    
    # Calcular estadísticas generales
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
    Evalúa modelos usando Traffic Light estadístico
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔍 Evaluando {model_name} con Traffic Light estadístico...")
        
        # Predicciones en cada conjunto
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        holdout_proba = model.predict_proba(X_holdout)[:, 1]
        
        # Evaluar en cada conjunto
        train_metrics = calculate_traffic_light_statistical(y_train, train_proba)
        test_metrics = calculate_traffic_light_statistical(y_test, test_proba)
        holdout_metrics = calculate_traffic_light_statistical(y_holdout, holdout_proba)
        
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'holdout': holdout_metrics
        }
        
        # Mostrar resultados
        print(f"   📈 Train  - Green: {train_metrics['green_percentage']:.1f}%, Yellow: {train_metrics['yellow_percentage']:.1f}%, Red: {train_metrics['red_percentage']:.1f}%")
        print(f"   🧪 Test   - Green: {test_metrics['green_percentage']:.1f}%, Yellow: {test_metrics['yellow_percentage']:.1f}%, Red: {test_metrics['red_percentage']:.1f}%")
        print(f"   🔒 Holdout - Green: {holdout_metrics['green_percentage']:.1f}%, Yellow: {holdout_metrics['yellow_percentage']:.1f}%, Red: {holdout_metrics['red_percentage']:.1f}%")
    
    return results

def create_traffic_light_table_statistical(results):
    """
    Crea tabla completa de Traffic Light estadístico
    """
    traffic_light_data = []
    
    for model_name, model_results in results.items():
        for dataset in ['train', 'test', 'holdout']:
            metrics = model_results[dataset]
            traffic_light = metrics
            
            for stat in traffic_light['group_stats']:
                traffic_light_data.append({
                    'Modelo': model_name,
                    'Dataset': dataset.capitalize(),
                    'Decil': stat['decile'],
                    'Actual_Rate': stat['actual_rate'],
                    'Predicted_Rate': stat['predicted_rate'],
                    'Difference': stat['difference'],
                    'P_Value': stat['p_value'],
                    'Color': stat['color'],
                    'Size': stat['size'],
                    'Min_Prob': stat['min_prob'],
                    'Max_Prob': stat['max_prob'],
                    'Avg_Prob': stat['avg_prob'],
                    'CI_Lower': stat['ci_lower'],
                    'CI_Upper': stat['ci_upper'],
                    'Significant': stat['significant']
                })
    
    return pd.DataFrame(traffic_light_data)

def visualize_traffic_light_statistical(traffic_light_df):
    """
    Visualiza Traffic Light estadístico
    """
    for model_name in traffic_light_df['Modelo'].unique():
        model_tl = traffic_light_df[traffic_light_df['Modelo'] == model_name]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{model_name} - Traffic Light Estadístico por Decil', fontsize=14, fontweight='bold')
        
        for idx, dataset in enumerate(['Train', 'Test', 'Holdout']):
            ax = axes[idx]
            dataset_data = model_tl[model_tl['Dataset'] == dataset].sort_values('Decil')
            
            # Crear gráfico de barras con colores
            colors_map = {'green': 'green', 'yellow': 'yellow', 'red': 'red'}
            bar_colors = [colors_map[c] for c in dataset_data['Color']]
            
            bars = ax.bar(dataset_data['Decil'], dataset_data['P_Value'], color=bar_colors, alpha=0.7)
            
            ax.set_title(f'{dataset}')
            ax.set_xlabel('Decil')
            ax.set_ylabel('P-Value')
            ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='α = 0.05')
            ax.axhline(y=0.025, color='red', linestyle='--', alpha=0.5, label='α/2 = 0.025')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Agregar valores en las barras
            for bar, val in zip(bars, dataset_data['P_Value']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("🚦 Traffic Light Estadístico - Código listo para usar")
    print("="*60)
    print("Funciones disponibles:")
    print("- calculate_traffic_light_statistical()")
    print("- evaluate_models_statistical()")
    print("- create_traffic_light_table_statistical()")
    print("- visualize_traffic_light_statistical()")
