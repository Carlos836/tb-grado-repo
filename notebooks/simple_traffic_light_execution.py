# -*- coding: utf-8 -*-
"""
Simple Traffic Light Execution for Already Trained Models
"""

# Load the Basel Traffic Light functions
exec(open('notebooks/basel_traffic_light.py').read())

# Check if we have the required variables
required_vars = ['trained_models', 'X_train', 'y_train', 'X_test', 'y_test', 'X_holdout', 'y_holdout']
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print(f"Variables faltantes: {missing_vars}")
    print("Asegurate de ejecutar primero:")
    print("1. La carga y preparacion de datos")
    print("2. El entrenamiento de modelos")
    print("3. La division de datos en train/test/holdout")
else:
    print(f"Todas las variables necesarias estan disponibles")
    print(f"Modelos disponibles: {list(trained_models.keys())}")
    print(f"Tamano de datos:")
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Test: {X_test.shape[0]} muestras") 
    print(f"   - Holdout: {X_holdout.shape[0]} muestras")
    
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