# -*- coding: utf-8 -*-
"""
Execute Basel Committee Traffic Light Analysis
Based on BCBS WP14 methodology
"""

# Load the Basel Traffic Light functions
exec(open('notebooks/basel_traffic_light.py').read())

# Analyze Traffic Light following Basel Committee methodology
print("Analizando Traffic Light con metodologia Basel Committee...")
print("Basado en: Studies on the Validation of Internal Rating Systems (BCBS WP14)")
print("Referencia: Tasche, Dirk (2003), A traffic lights approach to PD validation")
print()

basel_results = analyze_basel_traffic_light(trained_models, X_train, y_train, X_test, y_test, X_holdout, y_holdout)

# Display results
display_basel_traffic_light_tables(basel_results)

print("\n" + "="*100)
print("ANALISIS COMPLETADO - METODOLOGIA BASEL COMMITTEE")
print("="*100)
