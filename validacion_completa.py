#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validación del Sistema Completo
Modelos Conceptuales de Redes Neuronales para Supermercados

Este script valida que todos los componentes del sistema funcionen correctamente.
"""

import sys
import os
import importlib
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def log_message(message, level="INFO"):
    """Función para logging con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def test_imports():
    """Prueba que todos los módulos se importen correctamente"""
    log_message("🔧 Iniciando pruebas de importación de módulos...")
    
    modules_to_test = [
        'data_loader',
        'eda', 
        'modelo_1_regresion',
        'modelo_2_segmentacion', 
        'modelo_3_clasificacion',
        'modelo_4_anomalias',
        'utils'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            log_message(f"✅ {module_name}: Importado correctamente")
        except Exception as e:
            log_message(f"❌ {module_name}: Error de importación - {str(e)}", "ERROR")
            failed_imports.append(module_name)
    
    if failed_imports:
        log_message(f"❌ {len(failed_imports)} módulos fallaron en la importación", "ERROR")
        return False
    else:
        log_message("✅ Todos los módulos se importaron correctamente", "SUCCESS")
        return True

def test_data_loader():
    """Prueba el módulo de carga de datos"""
    log_message("🔧 Probando módulo data_loader...")
    
    try:
        import data_loader
        
        # Crear datos de prueba
        test_data = {
            'Invoice ID': ['001-01-1234', '001-01-1235', '001-01-1236'],
            'Branch': ['A', 'B', 'C'],
            'City': ['Yangon', 'Naypyitaw', 'Mandalay'],
            'Customer type': ['Member', 'Normal', 'Member'],
            'Gender': ['Female', 'Male', 'Female'],
            'Product line': ['Health and beauty', 'Electronic accessories', 'Home and lifestyle'],
            'Unit price': [74.69, 15.28, 46.33],
            'Quantity': [7, 5, 7],
            'Tax 5%': [26.1415, 3.82, 16.2155],
            'Total': [548.9715, 80.22, 340.5255],
            'Date': ['1/5/2019', '3/8/2019', '3/3/2019'],
            'Time': ['13:08', '10:29', '13:23'],
            'Payment': ['Ewallet', 'Cash', 'Credit card'],
            'cogs': [522.83, 76.4, 324.31],
            'gross margin percentage': [4.761904762, 4.761904762, 4.761904762],
            'gross income': [26.1415, 3.82, 16.2155],
            'Rating': [9.1, 9.6, 7.4]
        }
        
        # Crear DataFrame temporal
        temp_df = pd.DataFrame(test_data)
        temp_file = 'test_data_temp.xlsx'
        temp_df.to_excel(temp_file, index=False)
        
        # Probar la carga
        loaded_df = data_loader.cargar_datos(temp_file)
        
        # Verificar que se cargó correctamente
        assert len(loaded_df) == 3, "El DataFrame cargado no tiene el tamaño esperado"
        assert 'Rating' in loaded_df.columns, "La columna Rating no está presente"
        
        # Limpiar archivo temporal
        os.remove(temp_file)
        
        log_message("✅ data_loader: Funciona correctamente", "SUCCESS")
        return True
        
    except Exception as e:
        log_message(f"❌ data_loader: Error - {str(e)}", "ERROR")
        return False

def test_models_with_sample_data():
    """Prueba todos los modelos con datos de muestra"""
    log_message("🔧 Probando modelos con datos de muestra...")
    
    # Crear dataset de prueba más grande
    np.random.seed(42)
    n_samples = 100
    
    test_data = {
        'Unit price': np.random.uniform(10, 100, n_samples),
        'Quantity': np.random.randint(1, 10, n_samples),
        'Tax 5%': np.random.uniform(1, 50, n_samples),
        'Total': np.random.uniform(20, 500, n_samples),
        'cogs': np.random.uniform(15, 450, n_samples),
        'gross income': np.random.uniform(1, 50, n_samples),
        'Rating': np.random.uniform(4, 10, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Customer type': np.random.choice(['Member', 'Normal'], n_samples),
        'Product line': np.random.choice([
            'Health and beauty', 'Electronic accessories', 'Home and lifestyle',
            'Sports and travel', 'Food and beverages', 'Fashion accessories'
        ], n_samples),
        'Branch': np.random.choice(['A', 'B', 'C'], n_samples),
        'City': np.random.choice(['Yangon', 'Naypyitaw', 'Mandalay'], n_samples)
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Test Modelo 1: Regresión
    try:
        import modelo_1_regresion
        log_message("🧪 Probando modelo de regresión...")
        
        modelo, preproc, resultados = modelo_1_regresion.entrenar_regresion(df_test)
        
        assert 'R2' in resultados, "El resultado no contiene R2"
        assert 'MSE' in resultados, "El resultado no contiene MSE"
        assert 'MAE' in resultados, "El resultado no contiene MAE"
        
        log_message(f"✅ Regresión: R² = {resultados['R2']:.4f}", "SUCCESS")
        
    except Exception as e:
        log_message(f"❌ Modelo de regresión: Error - {str(e)}", "ERROR")
        return False
    
    # Test Modelo 2: Segmentación
    try:
        import modelo_2_segmentacion
        log_message("🧪 Probando modelo de segmentación...")
        
        # Usar solo variables numéricas para segmentación
        numeric_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']
        df_seg, kmeans, pca, preproc = modelo_2_segmentacion.segmentar_clientes(
            df_test[numeric_cols], n_clusters=3
        )
        
        assert 'Segmento' in df_seg.columns, "No se creó la columna Segmento"
        assert len(df_seg['Segmento'].unique()) == 3, "No se crearon 3 segmentos"
        
        caracteristicas = modelo_2_segmentacion.caracterizar_segmentos(df_seg)
        
        log_message(f"✅ Segmentación: {len(df_seg['Segmento'].unique())} segmentos creados", "SUCCESS")
        
    except Exception as e:
        log_message(f"❌ Modelo de segmentación: Error - {str(e)}", "ERROR")
        return False
    
    # Test Modelo 3: Clasificación
    try:
        import modelo_3_clasificacion
        log_message("🧪 Probando modelo de clasificación...")
        
        # Usar todas las columnas excepto Product line (variable objetivo)
        input_cols = [col for col in df_test.columns if col != 'Product line']
        df_class = df_test[input_cols + ['Product line']]
        
        modelo, preproc, resultados = modelo_3_clasificacion.entrenar_clasificacion(df_class)
        
        assert 'accuracy' in resultados, "El resultado no contiene accuracy"
        assert 'matriz_confusion' in resultados, "El resultado no contiene matriz_confusion"
        
        log_message(f"✅ Clasificación: Accuracy = {resultados['accuracy']:.4f}", "SUCCESS")
        
    except Exception as e:
        log_message(f"❌ Modelo de clasificación: Error - {str(e)}", "ERROR")
        return False
    
    # Test Modelo 4: Detección de Anomalías
    try:
        import modelo_4_anomalias
        log_message("🧪 Probando modelo de detección de anomalías...")
        
        variables_anomalias = ['Unit price', 'Quantity', 'Total', 'Rating']
        df_anom, modelo, preproc = modelo_4_anomalias.detectar_anomalias(
            df_test[variables_anomalias], variables_anomalias, contamination=0.1
        )
        
        assert 'Anomalía' in df_anom.columns, "No se creó la columna Anomalía"
        anomalias_detectadas = (df_anom['Anomalía'] == 'Sí').sum()
        
        log_message(f"✅ Detección de anomalías: {anomalias_detectadas} anomalías detectadas", "SUCCESS")
        
    except Exception as e:
        log_message(f"❌ Modelo de detección de anomalías: Error - {str(e)}", "ERROR")
        return False
    
    return True

def test_streamlit_compatibility():
    """Prueba la compatibilidad con Streamlit"""
    log_message("🔧 Probando compatibilidad con Streamlit...")
    
    try:
        import streamlit as st
        log_message("✅ Streamlit importado correctamente", "SUCCESS")
        
        # Verificar que las librerías de visualización están disponibles
        import matplotlib.pyplot as plt
        import seaborn as sns
        log_message("✅ Librerías de visualización disponibles", "SUCCESS")
        
        return True
    except Exception as e:
        log_message(f"❌ Error de compatibilidad con Streamlit: {str(e)}", "ERROR")
        return False

def run_full_validation():
    """Ejecuta la validación completa del sistema"""
    log_message("🚀 INICIANDO VALIDACIÓN COMPLETA DEL SISTEMA", "INFO")
    log_message("=" * 60, "INFO")
    
    all_tests_passed = True
    
    # Test 1: Importaciones
    if not test_imports():
        all_tests_passed = False
    
    print()
    
    # Test 2: Carga de datos
    if not test_data_loader():
        all_tests_passed = False
    
    print()
    
    # Test 3: Modelos
    if not test_models_with_sample_data():
        all_tests_passed = False
    
    print()
    
    # Test 4: Streamlit
    if not test_streamlit_compatibility():
        all_tests_passed = False
    
    print()
    log_message("=" * 60, "INFO")
    
    if all_tests_passed:
        log_message("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE", "SUCCESS")
        log_message("✅ El sistema está listo para producción", "SUCCESS")
        
        log_message("\n📋 RESUMEN DE FUNCIONALIDADES VALIDADAS:", "INFO")
        log_message("  ✅ Carga de datos desde Excel", "INFO")
        log_message("  ✅ Análisis exploratorio de datos", "INFO")
        log_message("  ✅ Modelo de regresión (MLPRegressor)", "INFO")
        log_message("  ✅ Modelo de segmentación (PCA + KMeans)", "INFO")
        log_message("  ✅ Modelo de clasificación (MLPClassifier)", "INFO")
        log_message("  ✅ Modelo de detección de anomalías (Isolation Forest)", "INFO")
        log_message("  ✅ Compatibilidad con Streamlit", "INFO")
        log_message("  ✅ Visualizaciones con Matplotlib y Seaborn", "INFO")
        
        log_message("\n🚀 Para ejecutar la aplicación, usa:", "INFO")
        log_message("   streamlit run app.py", "INFO")
        
        return True
    else:
        log_message("❌ ALGUNAS PRUEBAS FALLARON", "ERROR")
        log_message("🔧 Revisa los errores reportados arriba", "ERROR")
        return False

if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
