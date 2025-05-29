# Este módulo está alineado y documentado según la arquitectura conceptual ubicada en:
# C:\Users\efren\Downloads\supermarket_nn_models_entrega\home\ubuntu\supermarket_nn_models\docs\modelos_conceptuales.md

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Función para preparar los datos para clustering/autoencoder
def preparar_datos_segmentacion(df):
    # Variables fijas específicas para segmentación de clientes
    # Variables numéricas más importantes para comportamiento del cliente
    num_features = []
    for col in ['Total', 'Quantity', 'Unit price', 'gross income', 'Tax 5%', 'cogs', 'Rating']:
        if col in df.columns:
            num_features.append(col)
    
    # Variables categóricas relevantes para segmentación
    cat_features = []
    for col in ['Gender', 'Customer type', 'Product line', 'Branch', 'City']:
        if col in df.columns:
            cat_features.append(col)
    
    # Asegurar que tenemos al menos algunas columnas numéricas para segmentación efectiva
    if len(num_features) < 2:
        raise ValueError("Se necesitan al menos 2 variables numéricas para segmentación efectiva. Variables requeridas: Total, Quantity, Unit price, gross income")
    
    # Crear el preprocesador basado en las columnas disponibles
    transformers = []
    if cat_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_features))
    if num_features:
        transformers.append(('num', StandardScaler(), num_features))
    
    preprocessor = ColumnTransformer(transformers)
    X_processed = preprocessor.fit_transform(df)
    return X_processed, preprocessor

# Función para segmentar clientes usando KMeans sobre reducción PCA (simulación de autoencoder)
def segmentar_clientes(df, n_clusters=3):
    X, preprocessor = preparar_datos_segmentacion(df)
    # Reducción de dimensionalidad optimizada (menos componentes para velocidad)
    pca = PCA(n_components=min(4, X.shape[1]), random_state=42)  # Máximo 4 componentes
    X_latent = pca.fit_transform(X)
    # KMeans optimizado con menos iteraciones
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42,
        max_iter=100,  # Reducido de 300 a 100
        n_init=5       # Reducido de 10 a 5
    )
    clusters = kmeans.fit_predict(X_latent)
    df_segmentado = df.copy()
    df_segmentado['Segmento'] = clusters
    return df_segmentado, kmeans, pca, preprocessor

# Función para caracterizar segmentos
def caracterizar_segmentos(df_segmentado):
    # Detectar columnas disponibles para caracterización
    caracteristicas = {}
    
    for col in df_segmentado.columns:
        if col == 'Segmento':
            continue
            
        if pd.api.types.is_numeric_dtype(df_segmentado[col]):
            # Para columnas numéricas, calcular estadísticas
            caracteristicas[f'{col}_mean'] = df_segmentado.groupby('Segmento')[col].mean()
            caracteristicas[f'{col}_std'] = df_segmentado.groupby('Segmento')[col].std()
        else:
            # Para columnas categóricas, obtener la moda (valor más frecuente)
            try:
                caracteristicas[f'{col}_mode'] = df_segmentado.groupby('Segmento')[col].apply(
                    lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'N/A'
                )
            except:
                caracteristicas[f'{col}_mode'] = df_segmentado.groupby('Segmento')[col].apply(lambda x: 'N/A')
    
    # Añadir conteo de elementos por segmento
    caracteristicas['count'] = df_segmentado.groupby('Segmento').size()
    
    return pd.DataFrame(caracteristicas)
