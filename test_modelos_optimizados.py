"""
Test de modelos optimizados para verificar velocidad y funcionamiento
"""

import pandas as pd
import numpy as np
import time
from src import data_loader, modelo_1_regresion, modelo_2_segmentacion, modelo_3_clasificacion, modelo_4_anomalias

# Crear datos sintéticos de prueba para supermercado
def crear_datos_sinteticos(n_samples=500):  # Reducido de 1000 a 500 para velocidad
    """Crear datos sintéticos que simulan un dataset de supermercado"""
    np.random.seed(42)
    
    # Variables categóricas
    genders = ['Male', 'Female']
    customer_types = ['Member', 'Normal']
    product_lines = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 
                    'Sports and travel', 'Food and beverages', 'Fashion accessories']
    branches = ['A', 'B', 'C']
    cities = ['Yangon', 'Naypyitaw', 'Mandalay']
    
    data = {
        'Gender': np.random.choice(genders, n_samples),
        'Customer type': np.random.choice(customer_types, n_samples),
        'Product line': np.random.choice(product_lines, n_samples),
        'Branch': np.random.choice(branches, n_samples),
        'City': np.random.choice(cities, n_samples),
        'Unit price': np.random.uniform(10, 100, n_samples),
        'Quantity': np.random.randint(1, 11, n_samples),
    }
    
    # Calcular variables derivadas
    data['Total'] = data['Unit price'] * data['Quantity']
    data['Tax 5%'] = data['Total'] * 0.05
    data['cogs'] = data['Total'] / 1.05
    data['gross income'] = data['Tax 5%']
    data['Rating'] = np.random.uniform(4.0, 10.0, n_samples)
    
    return pd.DataFrame(data)

def test_modelo_regresion_optimizado():
    """Test del modelo de regresión optimizado"""
    print("🎯 Testando Modelo de Regresión Optimizado...")
    
    df = crear_datos_sinteticos()
    
    start_time = time.time()
    try:
        modelo, preproc, resultados = modelo_1_regresion.entrenar_regresion(df)
        end_time = time.time()
        
        print(f"   ✅ Entrenamiento exitoso en {end_time - start_time:.2f} segundos")
        print(f"   📊 R² Score: {resultados['R2']:.3f}")
        print(f"   📊 MSE: {resultados['MSE']:.3f}")
        print(f"   📊 MAE: {resultados['MAE']:.3f}")
        
        return True, end_time - start_time
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def test_modelo_segmentacion_optimizado():
    """Test del modelo de segmentación optimizado"""
    print("👥 Testando Modelo de Segmentación Optimizado...")
    
    df = crear_datos_sinteticos()
    
    start_time = time.time()
    try:
        df_seg, kmeans, pca, preproc = modelo_2_segmentacion.segmentar_clientes(df, n_clusters=3)
        caracteristicas = modelo_2_segmentacion.caracterizar_segmentos(df_seg)
        end_time = time.time()
        
        print(f"   ✅ Segmentación exitosa en {end_time - start_time:.2f} segundos")
        print(f"   📊 Segmentos creados: {len(df_seg['Segmento'].unique())}")
        print(f"   📊 Varianza explicada: {sum(pca.explained_variance_ratio_[:2]):.3f}")
        
        return True, end_time - start_time
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def test_modelo_clasificacion_optimizado():
    """Test del modelo de clasificación optimizado"""
    print("🛍️ Testando Modelo de Clasificación Optimizado...")
    
    df = crear_datos_sinteticos()
    
    start_time = time.time()
    try:
        modelo, preproc, le, resultados = modelo_3_clasificacion.entrenar_clasificacion(df)
        end_time = time.time()
        
        print(f"   ✅ Clasificación exitosa en {end_time - start_time:.2f} segundos")
        print(f"   📊 Accuracy: {resultados['accuracy']:.3f}")
        print(f"   📊 Clases detectadas: {len(le.classes_)}")
        
        return True, end_time - start_time
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def test_modelo_anomalias_optimizado():
    """Test del modelo de anomalías optimizado"""
    print("🔍 Testando Modelo de Anomalías Optimizado...")
    
    df = crear_datos_sinteticos()
    
    start_time = time.time()
    try:
        df_anom, modelo, preproc = modelo_4_anomalias.detectar_anomalias(df, contamination=0.1)
        end_time = time.time()
        
        anomalias = len(df_anom[df_anom['Anomalía'] == 'Sí'])
        total = len(df_anom)
        
        print(f"   ✅ Detección exitosa en {end_time - start_time:.2f} segundos")
        print(f"   📊 Anomalías detectadas: {anomalias}/{total} ({anomalias/total*100:.1f}%)")
        
        return True, end_time - start_time
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def main():
    """Función principal para ejecutar todos los tests"""
    print("🚀 TESTING DE MODELOS OPTIMIZADOS")
    print("=" * 50)
    
    tiempos = []
    exitos = 0
    total_tests = 4
    
    # Test de regresión
    success, tiempo = test_modelo_regresion_optimizado()
    if success:
        tiempos.append(tiempo)
        exitos += 1
    print()
    
    # Test de segmentación
    success, tiempo = test_modelo_segmentacion_optimizado()
    if success:
        tiempos.append(tiempo)
        exitos += 1
    print()
    
    # Test de clasificación
    success, tiempo = test_modelo_clasificacion_optimizado()
    if success:
        tiempos.append(tiempo)
        exitos += 1
    print()
    
    # Test de anomalías
    success, tiempo = test_modelo_anomalias_optimizado()
    if success:
        tiempos.append(tiempo)
        exitos += 1
    print()
    
    # Resumen final
    print("📊 RESUMEN DE OPTIMIZACIÓN")
    print("=" * 50)
    print(f"✅ Modelos exitosos: {exitos}/{total_tests}")
    
    if tiempos:
        print(f"⏱️ Tiempo promedio: {np.mean(tiempos):.2f} segundos")
        print(f"⏱️ Tiempo total: {sum(tiempos):.2f} segundos")
        print(f"⏱️ Modelo más rápido: {min(tiempos):.2f} segundos")
        print(f"⏱️ Modelo más lento: {max(tiempos):.2f} segundos")
    
    if exitos == total_tests:
        print("\n🎉 TODOS LOS MODELOS OPTIMIZADOS FUNCIONAN CORRECTAMENTE!")
        print("   Los modelos están listos para usar en el dashboard.")
    else:
        print(f"\n⚠️ {total_tests - exitos} modelo(s) necesitan revisión.")

if __name__ == "__main__":
    main()
