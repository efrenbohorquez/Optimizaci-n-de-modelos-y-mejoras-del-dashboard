"""
Test de Modelos con Variables Fijas
==================================
Este script prueba que todos los modelos funcionen correctamente 
con variables específicas fijas en lugar de detección automática.
"""

import pandas as pd
import numpy as np
from src import data_loader, modelo_1_regresion, modelo_2_segmentacion, modelo_3_clasificacion, modelo_4_anomalias
import warnings
warnings.filterwarnings('ignore')

def crear_datos_prueba():
    """Crea un dataset sintético para probar los modelos"""
    np.random.seed(42)
    n = 1000
    
    # Crear datos sintéticos del supermercado
    data = {
        'Invoice ID': [f'INV-{i:06d}' for i in range(n)],
        'Branch': np.random.choice(['A', 'B', 'C'], n),
        'City': np.random.choice(['Yangon', 'Naypyitaw', 'Mandalay'], n),
        'Customer type': np.random.choice(['Member', 'Normal'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Product line': np.random.choice([
            'Health and beauty', 'Electronic accessories', 'Home and lifestyle',
            'Sports and travel', 'Food and beverages', 'Fashion accessories'
        ], n),
        'Unit price': np.random.uniform(10, 100, n).round(2),
        'Quantity': np.random.randint(1, 11, n),
        'Tax 5%': np.random.uniform(0.5, 25, n).round(2),
        'cogs': np.random.uniform(10, 500, n).round(2),
        'gross income': np.random.uniform(0.5, 25, n).round(2),
        'Rating': np.random.uniform(4, 10, n).round(1)
    }
    
    # Calcular Total basado en otras variables
    data['Total'] = [data['cogs'][i] + data['Tax 5%'][i] for i in range(n)]
    
    df = pd.DataFrame(data)
    return df

def test_modelo_regresion(df):
    """Prueba el modelo de regresión con variables fijas"""
    print("\n=== PROBANDO MODELO DE REGRESIÓN ===")
    try:
        modelo, preproc, resultados = modelo_1_regresion.entrenar_regresion(df)
        print(f"✅ Regresión exitosa!")
        print(f"   - MSE: {resultados['MSE']:.4f}")
        print(f"   - MAE: {resultados['MAE']:.4f}")
        print(f"   - R²: {resultados['R2']:.4f}")
        print(f"   - Muestras de prueba: {len(resultados['y_test'])}")
        return True
    except Exception as e:
        print(f"❌ Error en regresión: {e}")
        return False

def test_modelo_segmentacion(df):
    """Prueba el modelo de segmentación con variables fijas"""
    print("\n=== PROBANDO MODELO DE SEGMENTACIÓN ===")
    try:
        df_seg, kmeans, pca, preproc = modelo_2_segmentacion.segmentar_clientes(df, n_clusters=3)
        caracteristicas = modelo_2_segmentacion.caracterizar_segmentos(df_seg)
        print(f"✅ Segmentación exitosa!")
        print(f"   - Clientes segmentados: {len(df_seg)}")
        print(f"   - Número de segmentos: {len(df_seg['Segmento'].unique())}")
        print(f"   - Componentes PCA: {pca.n_components_}")
        print(f"   - Características calculadas: {len(caracteristicas.columns)}")
        
        # Mostrar distribución de segmentos
        dist_segmentos = df_seg['Segmento'].value_counts().sort_index()
        print(f"   - Distribución: {dict(dist_segmentos)}")
        return True
    except Exception as e:
        print(f"❌ Error en segmentación: {e}")
        return False

def test_modelo_clasificacion(df):
    """Prueba el modelo de clasificación con variables fijas"""
    print("\n=== PROBANDO MODELO DE CLASIFICACIÓN ===")
    try:
        modelo, preproc, resultados = modelo_3_clasificacion.entrenar_clasificacion(df)
        print(f"✅ Clasificación exitosa!")
        print(f"   - Accuracy: {resultados['accuracy']:.4f}")
        print(f"   - Clases detectadas: {len(resultados['label_encoder'].classes_)}")
        print(f"   - Muestras de prueba: {len(resultados['y_test'])}")
        print(f"   - Clases: {list(resultados['label_encoder'].classes_)}")
        return True
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
        return False

def test_modelo_anomalias(df):
    """Prueba el modelo de detección de anomalías con variables fijas"""
    print("\n=== PROBANDO MODELO DE ANOMALÍAS ===")
    try:
        df_anom, modelo, preproc = modelo_4_anomalias.detectar_anomalias(df, contamination=0.05)
        anomalias = df_anom[df_anom['Anomalía'] == 'Sí']
        normales = df_anom[df_anom['Anomalía'] == 'No']
        
        print(f"✅ Detección de anomalías exitosa!")
        print(f"   - Total registros: {len(df_anom)}")
        print(f"   - Anomalías detectadas: {len(anomalias)}")
        print(f"   - Datos normales: {len(normales)}")
        print(f"   - Porcentaje anomalías: {len(anomalias)/len(df_anom)*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Error en anomalías: {e}")
        return False

def main():
    """Función principal que ejecuta todas las pruebas"""
    print("🧪 INICIANDO PRUEBAS DE MODELOS CON VARIABLES FIJAS")
    print("=" * 60)
    
    # Crear datos de prueba
    print("📊 Creando dataset sintético de supermercado...")
    df = crear_datos_prueba()
    print(f"   - Registros creados: {len(df)}")
    print(f"   - Columnas: {list(df.columns)}")
    
    # Ejecutar pruebas
    resultados = {
        'regresion': test_modelo_regresion(df),
        'segmentacion': test_modelo_segmentacion(df),
        'clasificacion': test_modelo_clasificacion(df),
        'anomalias': test_modelo_anomalias(df)
    }
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS:")
    exitosos = sum(resultados.values())
    total = len(resultados)
    
    for modelo, exito in resultados.items():
        status = "✅ EXITOSO" if exito else "❌ FALLÓ"
        print(f"   - {modelo.capitalize()}: {status}")
    
    print(f"\n🎯 RESULTADO FINAL: {exitosos}/{total} modelos funcionando correctamente")
    
    if exitosos == total:
        print("🎉 ¡TODOS LOS MODELOS FUNCIONAN CORRECTAMENTE!")
        print("💡 Los modelos están listos para mostrar proyecciones y cálculos en el dashboard.")
    else:
        print("⚠️  Algunos modelos necesitan revisión.")
    
    return exitosos == total

if __name__ == "__main__":
    main()
