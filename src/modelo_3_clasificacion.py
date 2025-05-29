# Este módulo está alineado y documentado según la arquitectura conceptual ubicada en:
# C:\Users\efren\Downloads\supermarket_nn_models_entrega\home\ubuntu\supermarket_nn_models\docs\modelos_conceptuales.md

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# Función para preparar los datos para clasificación
def preparar_datos_clasificacion(df):
    # Verificar que Product line esté disponible
    if 'Product line' not in df.columns:
        raise ValueError("La columna 'Product line' es requerida para el modelo de clasificación")
    
    # Variables fijas específicas para clasificación de línea de producto
    # Variables numéricas relevantes para predecir línea de producto
    num_features = []
    for col in ['Unit price', 'Quantity', 'Total', 'Tax 5%', 'cogs', 'gross income', 'Rating']:
        if col in df.columns:
            num_features.append(col)
    
    # Variables categóricas relevantes para clasificación
    cat_features = []
    for col in ['Gender', 'Customer type', 'Branch', 'City']:
        if col in df.columns:
            cat_features.append(col)
    
    # Asegurar que tenemos al menos algunas variables predictoras
    if not num_features and not cat_features:
        raise ValueError("No se encontraron las variables requeridas para clasificación. Se necesitan: Unit price, Quantity, Total, Gender, Customer type, Branch")
    
    target = 'Product line'
    y = df[target]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
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
    
    return X_processed, y_encoded, preprocessor, le

# Función para crear y entrenar el modelo de clasificación
def entrenar_clasificacion(df):
    X, y, preprocessor, le = preparar_datos_clasificacion(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Configuración optimizada para entrenamiento rápido
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Reducido de (128, 64, 32)
        activation='relu',
        max_iter=200,                 # Reducido de 500 a 200
        early_stopping=True,          # Para evitar sobreentrenamiento
        validation_fraction=0.1,
        n_iter_no_change=10,          # Para early stopping
        random_state=42,
        alpha=0.001                   # Regularización ligera
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    resultados = {
        'accuracy': accuracy_score(y_test, y_pred),
        'reporte': classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
        'matriz_confusion': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'label_encoder': le
    }
    return model, preprocessor, resultados
