# Modelo 1: Predicción de la Calificación del Cliente (Regresión)
# Este módulo implementa el modelo de regresión descrito en docs/modelos_conceptuales.md
# Este módulo está alineado y documentado según la arquitectura conceptual ubicada en:
# C:\Users\efren\Downloads\supermarket_nn_models_entrega\home\ubuntu\supermarket_nn_models\docs\modelos_conceptuales.md

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

# Función para preparar los datos para regresión
def preparar_datos_regresion(df):
    # Verificar que Rating esté disponible
    if 'Rating' not in df.columns:
        raise ValueError("La columna 'Rating' es requerida para el modelo de regresión")
    
    # Variables fijas específicas para regresión de rating
    # Variables numéricas más importantes para predecir rating
    num_features = []
    for col in ['Unit price', 'Quantity', 'Total', 'Tax 5%', 'cogs', 'gross income']:
        if col in df.columns:
            num_features.append(col)
    
    # Variables categóricas relevantes para rating
    cat_features = []
    for col in ['Gender', 'Customer type', 'Product line', 'Branch', 'City']:
        if col in df.columns:
            cat_features.append(col)
    
    # Asegurar que tenemos al menos algunas variables predictoras
    if not num_features and not cat_features:
        raise ValueError("No se encontraron las variables requeridas para regresión. Se necesitan: Unit price, Quantity, Total, Gender, Customer type, Product line")
    
    target = 'Rating'
    y = df[target]

    # Crear el preprocesador basado en las columnas disponibles
    transformers = []
    if cat_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_features))
    if num_features:
        transformers.append(('num', StandardScaler(), num_features))
    
    preprocessor = ColumnTransformer(transformers)
    
    # Seleccionar solo las columnas que vamos a usar
    feature_columns = cat_features + num_features
    X = df[feature_columns]
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

# Función para crear y entrenar el modelo de regresión
def entrenar_regresion(df):
    X, y, preprocessor = preparar_datos_regresion(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Configuración optimizada: menos capas y iteraciones para entrenamiento rápido
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        max_iter=200,  # Reducido de 500 a 200
        early_stopping=True,  # Para evitar sobreentrenamiento
        validation_fraction=0.1,
        n_iter_no_change=10,  # Para early stopping
        random_state=42,
        alpha=0.001  # Regularización ligera
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    resultados = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }
    return model, preprocessor, resultados
