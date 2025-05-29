# filepath: c:\Users\efren\Desktop\modelo_conceptual\modelos_conceptuales\app.py
# Este archivo está alineado y documentado según la arquitectura conceptual ubicada en:
# C:\Users\efren\Downloads\supermarket_nn_models_entrega\home\ubuntu\supermarket_nn_models\docs\modelos_conceptuales.md

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import data_loader, eda, modelo_1_regresion, modelo_2_segmentacion, modelo_3_clasificacion, modelo_4_anomalias
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Modelos Conceptuales Supermercado", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🛒"
)

# Configurar matplotlib para mejor compatibilidad
plt.style.use('default')
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')  # Fallback para versiones anteriores

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
        padding: 1em 0;
        background: linear-gradient(90deg, #3498db, #2c3e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #34495e;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        margin: 0.2em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #217dbb;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stDataFrame, .stTable {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5em;
        border: 1px solid #e9ecef;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🛒 Modelos Conceptuales de Redes Neuronales para Supermercados</div>', unsafe_allow_html=True)

# Sidebar para carga de datos
st.sidebar.header("📊 Carga de Datos")
try:
    st.sidebar.image("data/logo_uc.png", width=120)
except Exception:
    st.sidebar.info("💡 Coloca el logo institucional en 'data/logo_uc.png'")

st.sidebar.markdown('<div class="subtitle">Maestría en Analítica de Datos<br>Universidad Central</div>', unsafe_allow_html=True)
archivo = st.sidebar.file_uploader("Sube tu archivo de datos (xlsx)", type=["xlsx"])

# Información del proyecto en sidebar
with st.sidebar.expander("ℹ️ Información del Proyecto"):
    st.markdown("""
    **Modelos Implementados:**
    - 🎯 Regresión (MLPRegressor)
    - 👥 Segmentación (PCA + KMeans)
    - 🛍️ Clasificación (MLPClassifier)
    - 🔍 Detección de Anomalías (Isolation Forest)
    
    **Tecnologías:**
    - Streamlit
    - Scikit-learn
    - Pandas & NumPy
    - Matplotlib & Seaborn
    """)

# Cargar datos
df = None
if archivo:
    try:
        df = data_loader.cargar_datos(archivo)
        st.success("✅ Datos cargados correctamente.")
        st.sidebar.success(f"📁 Archivo: {archivo.name}")
        st.sidebar.info(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
else:
    st.info("📤 Por favor, sube un archivo de datos para comenzar el análisis.")

if df is not None:
    # Análisis Exploratorio de Datos
    st.header("📊 Análisis Exploratorio de Datos")
    st.markdown('<div class="subtitle">Explora los datos antes de aplicar los modelos</div>', unsafe_allow_html=True)
    
    try:
        eda.analisis_descriptivo(df)
    except Exception as e:
        st.error(f"Error en el análisis exploratorio: {e}")
    
    st.markdown("---")
    
    # Sección de Modelos
    st.header("🤖 Modelos Propuestos")
    st.markdown('<div class="subtitle">Selecciona y ejecuta el modelo de tu interés</div>', unsafe_allow_html=True)
    
    # Explicación de los modelos
    with st.expander("📖 Explicación de los Modelos y Conceptos de Redes Neuronales", expanded=False):
        st.markdown("""
        ### 🧠 **Conceptos de Redes Neuronales Aplicados:**
        
        **🎯 Regresión (MLPRegressor):**
        - Utiliza un perceptrón multicapa (red neuronal feedforward)
        - Aprende relaciones no lineales complejas
        - Capas ocultas con activación ReLU
        - Predice valores continuos (calificación del cliente)
        
        **👥 Segmentación (PCA + KMeans):**
        - Simula autoencoders mediante PCA para reducción de dimensionalidad
        - Encuentra representaciones latentes de los datos
        - Identifica patrones ocultos en el comportamiento del cliente
        - Agrupa clientes similares usando clustering
        
        **🛍️ Clasificación (MLPClassifier):**
        - Red neuronal multicapa para clasificación multiclase
        - Función de activación softmax en la capa de salida
        - Aprende límites de decisión complejos
        - Predice categorías (líneas de producto)
        
        **🔍 Detección de Anomalías (Isolation Forest):**
        - Inspirado en redes neuronales de autoencoder para detección de outliers
        - Identifica patrones anómalos mediante aislamiento
        - Útil para detectar fraudes o errores en datos
        """)
    
    # Guías de selección de variables
    with st.expander("📋 Guías de Selección de Variables por Modelo", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 **Regresión - Predicción de Rating**
            **Variables recomendadas:**
            - ✅ Variables de transacción: `Unit price`, `Quantity`, `Total`, `Tax 5%`, `cogs`, `gross income`
            - ✅ Variables demográficas: `Gender`, `Customer type`
            - ✅ Variables de producto: `Product line`
            - ✅ Variables de ubicación: `Branch`, `City`
            - ❌ **Excluir**: `Rating` (variable objetivo)
            
            ### 👥 **Segmentación de Clientes**
            **Variables recomendadas:**
            - ✅ Variables de comportamiento: `Total`, `Quantity`, `Unit price`, `gross income`
            - ✅ Variables demográficas: `Gender`, `Customer type`
            - ⚠️ **Mínimo 2 variables numéricas** para análisis efectivo
            """)
        
        with col2:
            st.markdown("""
            ### 🛍️ **Clasificación de Productos**
            **Variables recomendadas:**
            - ✅ Variables de contexto: `Total`, `Quantity`, `Unit price`, `Rating`
            - ✅ Variables demográficas: `Gender`, `Customer type`, `Branch`
            - ✅ Variables temporales: `Date`, `Time` (si disponibles)
            - ❌ **Excluir**: `Product line` (variable objetivo)
            
            ### 🔍 **Detección de Anomalías**
            **Variables recomendadas:**
            - ✅ Variables transaccionales: `Total`, `Quantity`, `Unit price`, `Tax 5%`
            - ✅ Variables de tiempo: `Date`, `Time`
            - ✅ Cualquier variable numérica con posibles outliers
            - ⚠️ **Mínimo 1 variable** requerida
            """)
    
    # Inicializar session state para selección de variables
    if 'var_select' not in st.session_state:
        st.session_state.var_select = list(df.columns)
    
    # Selector de variables mejorado
    st.subheader("🔧 Configuración de Variables")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        variables = st.multiselect(
            "Selecciona las variables para el análisis:",
            options=list(df.columns),
            default=st.session_state.var_select,
            key="var_select_widget",
            help="Selecciona las variables que deseas usar. Consulta las guías arriba para recomendaciones."
        )
        
        # Actualizar session_state
        if variables != st.session_state.var_select:
            st.session_state.var_select = variables
    
    with col2:
        st.markdown("**⚡ Configuración Rápida:**")
        
        if st.button("🎯 Para Regresión", help="Variables óptimas para regresión"):
            regression_vars = [col for col in df.columns if col not in ['Rating']]
            st.session_state.var_select = regression_vars
            st.rerun()
        
        if st.button("👥 Para Segmentación", help="Variables óptimas para segmentación"):
            seg_vars = [col for col in df.columns 
                       if col in ['Total', 'Quantity', 'Unit price', 'gross income', 
                                 'Gender', 'Customer type', 'Branch', 'City']]
            if not seg_vars:
                seg_vars = df.select_dtypes(include=['number']).columns.tolist()
            st.session_state.var_select = seg_vars
            st.rerun()
        
        if st.button("🛍️ Para Clasificación", help="Variables óptimas para clasificación"):
            class_vars = [col for col in df.columns if col not in ['Product line']]
            st.session_state.var_select = class_vars
            st.rerun()
        
        if st.button("🔍 Para Anomalías", help="Variables óptimas para detección de anomalías"):
            anomaly_vars = df.select_dtypes(include=['number']).columns.tolist()
            if 'Date' in df.columns:
                anomaly_vars.append('Date')
            if 'Time' in df.columns:
                anomaly_vars.append('Time')
            st.session_state.var_select = anomaly_vars
            st.rerun()
    
    # Usar variables del session state
    variables = st.session_state.var_select
    
    # Información sobre la selección actual
    if variables:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Variables Seleccionadas", len(variables))
        with col2:
            numeric_vars = len([v for v in variables if pd.api.types.is_numeric_dtype(df[v])])
            st.metric("Variables Numéricas", numeric_vars)
        with col3:
            categorical_vars = len([v for v in variables if not pd.api.types.is_numeric_dtype(df[v])])
            st.metric("Variables Categóricas", categorical_vars)
        with col4:
            missing_data = df[variables].isnull().sum().sum()
            st.metric("Datos Faltantes", missing_data)
    
    # Función para validar variables por modelo
    def validar_variables_modelo(modelo_tipo, variables_seleccionadas, df):
        """Valida si las variables seleccionadas son apropiadas para el modelo"""
        warnings_list = []
        recommendations = []
        
        if modelo_tipo == "regresion":
            if 'Rating' not in df.columns:
                warnings_list.append("❌ La variable 'Rating' es necesaria para el modelo de regresión.")
            
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 1:
                warnings_list.append("⚠️ Se recomienda al menos 1 variable numérica para mejor rendimiento.")
            
            if 'Rating' in variables_seleccionadas:
                recommendations.append("💡 'Rating' se excluirá automáticamente ya que es la variable objetivo.")
        
        elif modelo_tipo == "segmentacion":
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 2:
                warnings_list.append("❌ Se necesitan al menos 2 variables numéricas para segmentación efectiva.")
            
            if len(variables_seleccionadas) > 15:
                recommendations.append("💡 Muchas variables seleccionadas. El PCA ayudará a reducir dimensionalidad.")
        
        elif modelo_tipo == "clasificacion":
            if 'Product line' not in df.columns:
                warnings_list.append("❌ La variable 'Product line' es necesaria para el modelo de clasificación.")
            
            if 'Product line' in variables_seleccionadas:
                recommendations.append("💡 'Product line' se excluirá automáticamente ya que es la variable objetivo.")
        
        elif modelo_tipo == "anomalias":
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 1:
                warnings_list.append("❌ Se necesita al menos 1 variable numérica para detección de anomalías.")
            
            if len(variables_seleccionadas) == 1:
                recommendations.append("💡 Con 1 variable, las anomalías serán unidimensionales. Considera añadir más.")
        
        return warnings_list, recommendations
    
    # Selector de modelo
    st.markdown("---")
    st.subheader("🎯 Selección de Modelo")
    
    opcion = st.selectbox(
        "Elige el modelo a ejecutar:",
        [
            "🎯 Regresión: Predicción de Rating",
            "👥 Segmentación de Clientes", 
            "🛍️ Clasificación de Producto",
            "🔍 Detección de Anomalías"
        ],
        key="modelo_select"
    )
    
    # Validaciones específicas por modelo
    modelo_map = {
        "🎯 Regresión: Predicción de Rating": "regresion",
        "👥 Segmentación de Clientes": "segmentacion",
        "🛍️ Clasificación de Producto": "clasificacion",
        "🔍 Detección de Anomalías": "anomalias"
    }
    
    modelo_tipo = modelo_map.get(opcion, "")
    if modelo_tipo:
        warnings_list, recommendations = validar_variables_modelo(modelo_tipo, variables, df)
        
        # Mostrar validaciones
        if warnings_list:
            for warning in warnings_list:
                st.warning(warning)
        
        if recommendations:
            with st.expander("💡 Recomendaciones para optimizar el modelo", expanded=False):
                for rec in recommendations:
                    st.info(rec)
    
    st.markdown("---")
    
    # === IMPLEMENTACIÓN DE MODELOS ===
    
    if opcion == "🎯 Regresión: Predicción de Rating":
        st.subheader("🎯 Modelo de Regresión (MLPRegressor)")
        
        st.markdown("""
        **Objetivo:** Predecir la calificación del cliente basándose en variables transaccionales y demográficas.
        
        **Algoritmo:** Red neuronal multicapa que aprende relaciones no lineales complejas entre las variables de entrada y la calificación del cliente.
        """)
        
        can_train = 'Rating' in df.columns and len([v for v in variables if pd.api.types.is_numeric_dtype(df[v])]) >= 1
        
        if can_train:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**⚙️ Configuración del Modelo:**")
                capas_ocultas = st.selectbox(
                    "Arquitectura de capas ocultas:",
                    [(100,), (128, 64), (128, 64, 32), (100, 50, 25)],
                    index=2,
                    format_func=lambda x: f"{len(x)} capas: {x}"
                )
                
                max_iter = st.slider("Máximo de iteraciones:", 200, 1000, 500, 100)
                
            with col1:
                if st.button("🚀 Entrenar Modelo de Regresión", type="primary", use_container_width=True):
                    with st.spinner("🔄 Entrenando modelo de regresión..."):
                        try:
                            input_vars = [v for v in variables if v != 'Rating']
                            modelo, preproc, resultados = modelo_1_regresion.entrenar_regresion(
                                df[input_vars + ['Rating']].dropna()
                            )
                            
                            st.success("✅ Entrenamiento completado exitosamente!")
                            
                            # Métricas principales
                            st.subheader("📊 Métricas del Modelo")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "MSE (Error Cuadrático Medio)", 
                                    f"{resultados['MSE']:.4f}",
                                    help="Menor es mejor"
                                )
                            with col2:
                                st.metric(
                                    "MAE (Error Absoluto Medio)", 
                                    f"{resultados['MAE']:.4f}",
                                    help="Menor es mejor"
                                )
                            with col3:
                                r2_score = resultados['R2']
                                st.metric(
                                    "R² (Coeficiente de Determinación)", 
                                    f"{r2_score:.4f}",
                                    delta=f"{(r2_score - 0.5):.4f}" if r2_score > 0.5 else None,
                                    help="Más cercano a 1 es mejor"
                                )
                            
                            # Interpretación del modelo
                            st.subheader("🔍 Interpretación del Modelo")
                            
                            if r2_score > 0.8:
                                st.success("🎉 Excelente rendimiento del modelo (R² > 0.8)")
                            elif r2_score > 0.6:
                                st.info("👍 Buen rendimiento del modelo (R² > 0.6)")
                            elif r2_score > 0.4:
                                st.warning("⚠️ Rendimiento moderado del modelo (R² > 0.4)")
                            else:
                                st.error("❌ Rendimiento bajo del modelo (R² ≤ 0.4)")
                            
                            # Visualizaciones
                            st.subheader("📈 Análisis Visual")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Gráfico de predicciones vs reales
                                fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                                
                                sns.scatterplot(
                                    x=resultados['y_test'], 
                                    y=resultados['y_pred'], 
                                    ax=ax_scatter, 
                                    alpha=0.7, 
                                    color='#3498db'
                                )
                                
                                # Línea de predicción perfecta
                                min_val = min(min(resultados['y_test']), min(resultados['y_pred']))
                                max_val = max(max(resultados['y_test']), max(resultados['y_pred']))
                                ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
                                
                                ax_scatter.set_xlabel("Valores Reales", fontsize=12)
                                ax_scatter.set_ylabel("Valores Predichos", fontsize=12)
                                ax_scatter.set_title("Predicciones vs Valores Reales", fontsize=14, fontweight='bold')
                                ax_scatter.legend()
                                ax_scatter.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig_scatter)
                            
                            with col2:
                                # Análisis de residuos
                                residuos = resultados['y_test'].values - resultados['y_pred']
                                
                                fig_res, ax_res = plt.subplots(figsize=(8, 6))
                                sns.scatterplot(x=resultados['y_pred'], y=residuos, ax=ax_res, alpha=0.7, color='#e74c3c')
                                ax_res.hlines(y=0, xmin=min(resultados['y_pred']), xmax=max(resultados['y_pred']), 
                                            color='black', linestyle='--', linewidth=2)
                                ax_res.set_xlabel("Valores Predichos", fontsize=12)
                                ax_res.set_ylabel("Residuos", fontsize=12)
                                ax_res.set_title("Análisis de Residuos", fontsize=14, fontweight='bold')
                                ax_res.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig_res)
                            
                            # Distribución de residuos
                            st.subheader("📊 Distribución de Residuos")
                            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                            sns.histplot(residuos, kde=True, ax=ax_hist, color='#9b59b6', alpha=0.7, bins=30)
                            ax_hist.axvline(x=0, color='black', linestyle='--', linewidth=2)
                            ax_hist.set_xlabel("Residuos", fontsize=12)
                            ax_hist.set_ylabel("Frecuencia", fontsize=12)
                            ax_hist.set_title("Distribución de Residuos", fontsize=14, fontweight='bold')
                            ax_hist.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_hist)
                            
                            # Estadísticas de residuos
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Media de Residuos", f"{np.mean(residuos):.4f}")
                            with col2:
                                st.metric("Desv. Estándar", f"{np.std(residuos):.4f}")
                            with col3:
                                st.metric("Residuo Mínimo", f"{np.min(residuos):.4f}")
                            with col4:
                                st.metric("Residuo Máximo", f"{np.max(residuos):.4f}")
                                
                        except Exception as e:
                            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                            st.exception(e)
        else:
            st.info("⚠️ Verifica que las variables seleccionadas cumplan con los requisitos.")
    
    elif opcion == "👥 Segmentación de Clientes":
        st.subheader("👥 Segmentación de Clientes (PCA + KMeans)")
        
        st.markdown("""
        **Objetivo:** Agrupar clientes en segmentos homogéneos basándose en patrones de comportamiento.
        
        **Algoritmo:** Reducción de dimensionalidad con PCA seguida de clustering con KMeans para identificar grupos naturales.
        """)
        
        numeric_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        can_segment = len(numeric_vars) >= 2
        
        if can_segment:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**⚙️ Configuración:**")
                n_clusters = st.slider("Número de segmentos:", min_value=2, max_value=8, value=3)
                
            with col1:
                st.info(f"💡 Se usarán {len(numeric_vars)} variables numéricas para la segmentación")
            
            if st.button("🚀 Ejecutar Segmentación", type="primary", use_container_width=True):
                with st.spinner("🔄 Segmentando clientes..."):
                    try:
                        df_seg, kmeans, pca, preproc = modelo_2_segmentacion.segmentar_clientes(
                            df[variables].dropna(), 
                            n_clusters=n_clusters
                        )
                        caracteristicas = modelo_2_segmentacion.caracterizar_segmentos(df_seg)
                        
                        st.success("✅ Segmentación completada exitosamente!")
                        
                        # Métricas de segmentación
                        st.subheader("📊 Métricas de Segmentación")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Segmentos Creados", n_clusters)
                        with col2:
                            st.metric("Clientes Segmentados", len(df_seg))
                        with col3:
                            try:
                                from sklearn.metrics import silhouette_score
                                sil_score = silhouette_score(
                                    pca.transform(preproc.transform(df[variables].dropna())), 
                                    df_seg['Segmento']
                                )
                                st.metric("Silhouette Score", f"{sil_score:.3f}")
                            except:
                                st.metric("Variables Usadas", len(variables))
                        with col4:
                            varianza_explicada = sum(pca.explained_variance_ratio_[:2]) * 100
                            st.metric("Varianza Explicada (PC1+PC2)", f"{varianza_explicada:.1f}%")
                        
                        # Visualización PCA
                        st.subheader("🎯 Visualización de Segmentos (Espacio PCA)")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            pca_coords = pca.transform(preproc.transform(df[variables].dropna()))
                            
                            scatter = sns.scatterplot(
                                x=pca_coords[:,0], 
                                y=pca_coords[:,1], 
                                hue=df_seg['Segmento'], 
                                palette='Set2', 
                                ax=ax,
                                s=100,
                                alpha=0.7
                            )
                            
                            # Añadir centroides
                            for segment in df_seg['Segmento'].unique():
                                mask = df_seg['Segmento'] == segment
                                centroid_x = pca_coords[mask, 0].mean()
                                centroid_y = pca_coords[mask, 1].mean()
                                ax.scatter(centroid_x, centroid_y, c='black', s=300, marker='X', 
                                         linewidths=3, label=f'Centroide {segment}' if segment == df_seg['Segmento'].unique()[0] else "")
                            
                            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)", fontsize=12)
                            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)", fontsize=12)
                            ax.set_title("Segmentación de Clientes (Espacio PCA)", fontsize=14, fontweight='bold')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("**📊 Distribución de Segmentos:**")
                            segment_counts = df_seg['Segmento'].value_counts().sort_index()
                            
                            for segment, count in segment_counts.items():
                                percentage = (count / len(df_seg)) * 100
                                st.markdown(f"""
                                <div class="metric-container">
                                    <strong>Segmento {segment}:</strong><br>
                                    {count} clientes ({percentage:.1f}%)
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Características por segmento
                        st.subheader("📋 Características por Segmento")
                        st.dataframe(caracteristicas.round(3), use_container_width=True)
                        
                        # Análisis detallado por variables
                        st.subheader("📈 Análisis Detallado por Variables")
                        
                        numeric_vars_for_analysis = [col for col in variables if pd.api.types.is_numeric_dtype(df[col])]
                        
                        if numeric_vars_for_analysis:
                            selected_vars_analysis = st.multiselect(
                                "Selecciona variables para análisis detallado:",
                                options=numeric_vars_for_analysis,
                                default=numeric_vars_for_analysis[:min(len(numeric_vars_for_analysis), 3)],
                                key="segment_analysis_vars"
                            )
                            
                            if selected_vars_analysis:
                                cols_per_row = 2
                                for i in range(0, len(selected_vars_analysis), cols_per_row):
                                    cols = st.columns(cols_per_row)
                                    
                                    for j, var_name in enumerate(selected_vars_analysis[i:i+cols_per_row]):
                                        with cols[j]:
                                            fig_box, ax_box = plt.subplots(figsize=(8, 6))
                                            
                                            sns.boxplot(
                                                data=df_seg, 
                                                x='Segmento', 
                                                y=var_name, 
                                                ax=ax_box, 
                                                palette='Set2'
                                            )
                                            sns.stripplot(
                                                data=df_seg, 
                                                x='Segmento', 
                                                y=var_name, 
                                                ax=ax_box, 
                                                color='black', 
                                                alpha=0.3, 
                                                size=3
                                            )
                                            
                                            ax_box.set_title(f"Distribución de {var_name}", fontsize=12, fontweight='bold')
                                            ax_box.grid(True, alpha=0.3)
                                            plt.tight_layout()
                                            st.pyplot(fig_box)
                                
                                # Estadísticas detalladas
                                with st.expander("📊 Estadísticas Detalladas", expanded=False):
                                    for var in selected_vars_analysis:
                                        st.markdown(f"**{var}:**")
                                        stats_by_segment = df_seg.groupby('Segmento')[var].agg([
                                            'count', 'mean', 'std', 'min', 'median', 'max'
                                        ]).round(3)
                                        st.dataframe(stats_by_segment, use_container_width=True)
                        
                        # Opción de descarga
                        st.subheader("💾 Descargar Resultados")
                        csv_data = df_seg.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar datos segmentados (CSV)",
                            data=csv_data,
                            file_name=f"clientes_segmentados_{n_clusters}_grupos.csv",
                            mime="text/csv",
                            help="Descarga el dataset con la asignación de segmentos"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error durante la segmentación: {str(e)}")
                        st.exception(e)
        else:
            st.info("⚠️ Se necesitan al menos 2 variables numéricas para la segmentación.")
    
    elif opcion == "🛍️ Clasificación de Producto":
        st.subheader("🛍️ Clasificación de Producto (MLPClassifier)")
        
        st.markdown("""
        **Objetivo:** Predecir la línea de producto basándose en características del cliente y transacción.
        
        **Algoritmo:** Red neuronal multicapa con clasificación multiclase que aprende patrones de compra complejos.
        """)
        
        can_classify = 'Product line' in df.columns
        
        if can_classify:
            product_lines = df['Product line'].unique()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"💡 Se entrenará para clasificar entre {len(product_lines)} líneas de producto")
                
                with st.expander("📊 Distribución de Líneas de Producto", expanded=False):
                    class_dist = df['Product line'].value_counts()
                    
                    # Gráfico de barras
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                    class_dist.plot(kind='bar', ax=ax_dist, color='skyblue', alpha=0.8)
                    ax_dist.set_title("Distribución de Líneas de Producto", fontsize=14, fontweight='bold')
                    ax_dist.set_xlabel("Línea de Producto", fontsize=12)
                    ax_dist.set_ylabel("Cantidad", fontsize=12)
                    ax_dist.tick_params(axis='x', rotation=45)
                    ax_dist.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_dist)
                    
                    st.dataframe(
                        class_dist.reset_index().rename(columns={'index': 'Línea de Producto', 'Product line': 'Cantidad'}),
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("**⚙️ Configuración:**")
                st.metric("Clases Únicas", len(product_lines))
                
                capas_ocultas_clf = st.selectbox(
                    "Arquitectura de capas:",
                    [(100,), (128, 64), (128, 64, 32)],
                    index=1,
                    format_func=lambda x: f"{len(x)} capas: {x}",
                    key="clf_layers"
                )
            
            if st.button("🚀 Entrenar Modelo de Clasificación", type="primary", use_container_width=True):
                with st.spinner("🔄 Entrenando modelo de clasificación..."):
                    try:
                        input_vars = [v for v in variables if v != 'Product line']
                        modelo, preproc, resultados = modelo_3_clasificacion.entrenar_clasificacion(
                            df[input_vars + ['Product line']].dropna()
                        )
                        
                        st.success("✅ Entrenamiento completado exitosamente!")
                        
                        # Métricas principales
                        st.subheader("📊 Métricas del Modelo")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Exactitud (Accuracy)", f"{resultados['accuracy']:.3f}")
                        
                        with col2:
                            precision_scores = [v['precision'] for v in resultados['reporte'].values() 
                                              if isinstance(v, dict) and 'precision' in v]
                            if precision_scores:
                                precision_avg = np.mean(precision_scores)
                                st.metric("Precisión Promedio", f"{precision_avg:.3f}")
                        
                        with col3:
                            recall_scores = [v['recall'] for v in resultados['reporte'].values() 
                                           if isinstance(v, dict) and 'recall' in v]
                            if recall_scores:
                                recall_avg = np.mean(recall_scores)
                                st.metric("Recall Promedio", f"{recall_avg:.3f}")
                        
                        # Matriz de confusión
                        st.subheader("📋 Matriz de Confusión")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            sns.heatmap(
                                resultados['matriz_confusion'], 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues', 
                                ax=ax,
                                xticklabels=product_lines,
                                yticklabels=product_lines,
                                cbar_kws={'label': 'Cantidad'}
                            )
                            
                            ax.set_xlabel('Predicción', fontsize=12)
                            ax.set_ylabel('Real', fontsize=12)
                            ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
                            plt.xticks(rotation=45)
                            plt.yticks(rotation=0)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("**📈 Métricas por Clase:**")
                            reporte_df = pd.DataFrame(resultados['reporte']).transpose()
                            class_metrics = reporte_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
                            st.dataframe(class_metrics.round(3), use_container_width=True)
                        
                        # Interpretación del modelo
                        st.subheader("🔍 Interpretación del Modelo")
                        accuracy = resultados['accuracy']
                        
                        if accuracy > 0.9:
                            st.success("🎉 Excelente rendimiento del modelo (Accuracy > 90%)")
                        elif accuracy > 0.8:
                            st.info("👍 Buen rendimiento del modelo (Accuracy > 80%)")
                        elif accuracy > 0.7:
                            st.warning("⚠️ Rendimiento moderado del modelo (Accuracy > 70%)")
                        else:
                            st.error("❌ Rendimiento bajo del modelo (Accuracy ≤ 70%)")
                        
                        # Análisis de clases más difíciles de predecir
                        diag = np.diag(resultados['matriz_confusion'])
                        totales_por_clase = resultados['matriz_confusion'].sum(axis=1)
                        accuracy_por_clase = diag / totales_por_clase
                        
                        st.markdown("**🎯 Análisis por Clase:**")
                        for i, product_line in enumerate(product_lines):
                            acc_clase = accuracy_por_clase[i]
                            if acc_clase < 0.7:
                                st.warning(f"⚠️ {product_line}: {acc_clase:.2%} - Clase difícil de predecir")
                            elif acc_clase > 0.9:
                                st.success(f"✅ {product_line}: {acc_clase:.2%} - Excelente predicción")
                            else:
                                st.info(f"ℹ️ {product_line}: {acc_clase:.2%} - Buena predicción")
                                
                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                        st.exception(e)
        else:
            st.info("⚠️ La variable 'Product line' es necesaria para este modelo.")
    
    elif opcion == "🔍 Detección de Anomalías":
        st.subheader("🔍 Detección de Anomalías (Isolation Forest)")
        
        st.markdown("""
        **Objetivo:** Identificar observaciones atípicas o anómalas en los datos del supermercado.
        
        **Algoritmo:** Isolation Forest aísla anomalías mediante divisiones aleatorias. Las observaciones que requieren menos divisiones para ser aisladas se consideran anómalas.
        
        **Aplicaciones:** Detección de fraudes, errores de captura, comportamientos inusuales de compra.
        """)
        
        numeric_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        can_detect = len(variables) >= 1
        
        if can_detect:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"💡 Se usarán {len(variables)} variables, {len(numeric_vars)} de ellas numéricas")
            
            with col2:
                st.markdown("**⚙️ Configuración:**")
                contamination = st.slider(
                    "Proporción esperada de anomalías:", 
                    min_value=0.01, 
                    max_value=0.2, 
                    value=0.05, 
                    step=0.01,
                    help="Porcentaje esperado de datos anómalos"
                )
                
                n_estimators = st.slider(
                    "Número de árboles:",
                    min_value=50,
                    max_value=200,
                    value=100,
                    step=25
                )
            
            if st.button("🚀 Detectar Anomalías", type="primary", use_container_width=True):
                with st.spinner("🔄 Detectando anomalías..."):
                    try:
                        df_anom, modelo, preproc = modelo_4_anomalias.detectar_anomalias(
                            df[variables].dropna(), 
                            variables, 
                            contamination
                        )
                        
                        st.success("✅ Detección de anomalías completada!")
                        
                        # Métricas principales
                        total_anomalias = sum(df_anom['Anomalía'] == 'Sí')
                        total_normales = sum(df_anom['Anomalía'] == 'No')
                        
                        st.subheader("📊 Resultados de la Detección")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Registros", len(df_anom))
                        with col2:
                            st.metric("Anomalías Detectadas", total_anomalias)
                        with col3:
                            st.metric("Datos Normales", total_normales)
                        with col4:
                            percentage = (total_anomalias / len(df_anom)) * 100
                            st.metric("% Anomalías", f"{percentage:.2f}%")
                        
                        # Vista previa de resultados
                        st.subheader("👀 Vista Previa de Resultados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Datos Normales (muestra):**")
                            normales = df_anom[df_anom['Anomalía'] == 'No'].head()
                            if len(normales) > 0:
                                st.dataframe(normales, use_container_width=True)
                            else:
                                st.info("No hay datos normales para mostrar.")
                        
                        with col2:
                            st.markdown("**🚨 Anomalías Detectadas:**")
                            anomalias = df_anom[df_anom['Anomalía'] == 'Sí']
                            if len(anomalias) > 0:
                                st.dataframe(anomalias, use_container_width=True)
                            else:
                                st.info("No se detectaron anomalías con los parámetros actuales.")
                        
                        # Distribución de anomalías
                        st.subheader("📊 Distribución de Anomalías")
                        
                        anomaly_dist = df_anom['Anomalía'].value_counts()
                        
                        fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
                        colors = ['#2ecc71', '#e74c3c']  # Verde para normal, rojo para anomalías
                        bars = ax_dist.bar(anomaly_dist.index, anomaly_dist.values, color=colors, alpha=0.8)
                        
                        # Añadir etiquetas
                        for bar, value in zip(bars, anomaly_dist.values):
                            height = bar.get_height()
                            ax_dist.text(
                                bar.get_x() + bar.get_width()/2., 
                                height + height*0.01,
                                f'{value}\n({value/len(df_anom)*100:.1f}%)', 
                                ha='center', 
                                va='bottom', 
                                fontweight='bold'
                            )
                        
                        ax_dist.set_title('Distribución: Datos Normales vs Anomalías', fontsize=14, fontweight='bold')
                        ax_dist.set_xlabel('Tipo de Dato', fontsize=12)
                        ax_dist.set_ylabel('Cantidad', fontsize=12)
                        ax_dist.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_dist)
                        
                        # Análisis visual detallado
                        if numeric_vars:
                            st.subheader("🎯 Análisis Visual Detallado")
                            
                            numeric_cols_for_viz = [col for col in df_anom.columns 
                                                   if col != 'Anomalía' and pd.api.types.is_numeric_dtype(df_anom[col])]
                            
                            if numeric_cols_for_viz:
                                selected_viz_vars = st.multiselect(
                                    "Selecciona hasta 2 variables numéricas para análisis visual:",
                                    options=numeric_cols_for_viz,
                                    default=numeric_cols_for_viz[:min(len(numeric_cols_for_viz), 2)],
                                    max_selections=2,
                                    key="viz_anomaly_vars"
                                )
                                
                                if selected_viz_vars:
                                    if len(selected_viz_vars) == 1:
                                        var_name = selected_viz_vars[0]
                                        st.markdown(f"**Análisis de '{var_name}':**")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Histograma
                                            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                                            sns.histplot(
                                                data=df_anom, 
                                                x=var_name, 
                                                hue='Anomalía', 
                                                kde=True, 
                                                ax=ax_hist, 
                                                palette={'No':'#5dade2', 'Sí':'#e74c3c'},
                                                alpha=0.7
                                            )
                                            ax_hist.set_title(f"Distribución de {var_name}", fontsize=14, fontweight='bold')
                                            ax_hist.grid(True, alpha=0.3)
                                            plt.tight_layout()
                                            st.pyplot(fig_hist)
                                        
                                        with col2:
                                            # Boxplot
                                            fig_box, ax_box = plt.subplots(figsize=(10, 6))
                                            sns.boxplot(
                                                data=df_anom, 
                                                x='Anomalía', 
                                                y=var_name, 
                                                ax=ax_box, 
                                                palette={'No':'#5dade2', 'Sí':'#e74c3c'}
                                            )
                                            ax_box.set_title(f"Boxplot de {var_name}", fontsize=14, fontweight='bold')
                                            ax_box.grid(True, alpha=0.3)
                                            plt.tight_layout()
                                            st.pyplot(fig_box)
                                    
                                    elif len(selected_viz_vars) == 2:
                                        var1, var2 = selected_viz_vars[0], selected_viz_vars[1]
                                        st.markdown(f"**Relación entre '{var1}' y '{var2}':**")
                                        
                                        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                                        sns.scatterplot(
                                            data=df_anom, 
                                            x=var1, 
                                            y=var2, 
                                            hue='Anomalía', 
                                            style='Anomalía', 
                                            ax=ax_scatter, 
                                            palette={'No':'#5dade2', 'Sí':'#e74c3c'}, 
                                            markers={'No':'o', 'Sí':'X'}, 
                                            s=100,
                                            alpha=0.7
                                        )
                                        ax_scatter.set_title(f"Relación entre {var1} y {var2}", fontsize=14, fontweight='bold')
                                        ax_scatter.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig_scatter)
                                    
                                    # Estadísticas descriptivas
                                    st.subheader("📈 Estadísticas Descriptivas")
                                    
                                    stats_normales = df_anom[df_anom['Anomalía'] == 'No'][selected_viz_vars].describe()
                                    stats_anomalias = df_anom[df_anom['Anomalía'] == 'Sí'][selected_viz_vars].describe()
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**📊 Datos Normales:**")
                                        st.dataframe(stats_normales.round(3), use_container_width=True)
                                    
                                    with col2:
                                        st.markdown("**🚨 Anomalías:**")
                                        if len(stats_anomalias.columns) > 0:
                                            st.dataframe(stats_anomalias.round(3), use_container_width=True)
                                        else:
                                            st.info("No hay anomalías suficientes para estadísticas")
                        
                        # Opción de descarga
                        st.subheader("💾 Descargar Resultados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_all = df_anom.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Descargar todos los datos con etiquetas",
                                data=csv_all,
                                file_name="datos_con_anomalias.csv",
                                mime="text/csv",
                                help="Dataset completo con clasificación de anomalías"
                            )
                        
                        with col2:
                            if len(anomalias) > 0:
                                csv_anomalies = anomalias.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar solo anomalías",
                                    data=csv_anomalies,
                                    file_name="solo_anomalias.csv",
                                    mime="text/csv",
                                    help="Solo los registros identificados como anómalos"
                                )
                            else:
                                st.info("No hay anomalías para descargar")
                                
                    except Exception as e:
                        st.error(f"❌ Error durante la detección: {str(e)}")
                        st.exception(e)
        else:
            st.info("⚠️ Selecciona al menos una variable para detectar anomalías.")
    
    else:
        st.info("Selecciona un modelo para comenzar el análisis.")

else:
    # Mostrar información cuando no hay datos cargados
    st.markdown("---")
    st.subheader("🚀 ¿Cómo empezar?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📤 1. Carga tus datos
        - Usa el panel lateral
        - Formato: archivo Excel (.xlsx)
        - Datos de supermercado recomendados
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 2. Configura variables
        - Selecciona variables relevantes
        - Usa las guías de configuración rápida
        - Revisa las recomendaciones por modelo
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 3. Ejecuta modelos
        - Elige el modelo de tu interés
        - Ajusta parámetros si es necesario
        - Analiza los resultados
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em; margin-top: 2em;'>
    <p><strong>🛒 Modelos Conceptuales de Redes Neuronales para Supermercados</strong></p>
    <p>Maestría en Analítica de Datos - Universidad Central</p>
    <p>Desarrollado con ❤️ usando Streamlit, Scikit-learn, y conceptos de Deep Learning</p>
    <p><em>Versión 3.0 - Noviembre 2024 | Arquitectura completa optimizada</em></p>
</div>
""", unsafe_allow_html=True)
