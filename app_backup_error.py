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

st.set_page_config(page_title="Modelos Conceptuales Supermercado", layout="wide", initial_sidebar_state="expanded")

# Configurar estilo de matplotlib para compatibilidad
plt.style.use('default')
# Configurar seaborn 
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')  # Fallback para versiones anteriores

st.markdown("""
<style>
    body, .main, .stApp {
        background-color: #ffffff !important;
        color: #222 !important;
    }
    .main-title {
        font-size:2.5rem;
        font-weight:bold;
        color:#2c3e50;
        margin-bottom:0.5em;
    }
    .subtitle {
        font-size:1.3rem;
        color:#34495e;
        margin-bottom:0.5em;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        margin: 0.5em 0;
    }
    .stButton>button:hover {
        background-color: #217dbb;
        color: #ecf0f1;
    }
    .stDataFrame, .stTable {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Modelos Conceptuales de Redes Neuronales para Supermercados</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Carga de datos")
try:
    st.sidebar.image("data/logo_uc.png", width=120)
except Exception:
    st.sidebar.warning("No se pudo cargar el logo institucional. Verifica que 'data/logo_uc.png' exista.")

st.sidebar.markdown('<div class="subtitle">Maestría en Analítica de Datos<br>Universidad Central</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="subtitle">Carga de datos</div>', unsafe_allow_html=True)
archivo = st.sidebar.file_uploader("Sube tu archivo de datos (xlsx)", type=["xlsx"])

df = None
if archivo:
    df = data_loader.cargar_datos(archivo)
    st.success("Datos cargados correctamente.")
else:
    st.info("Por favor, sube un archivo de datos para comenzar.")

if df is not None:
    st.header("Análisis Descriptivo")
    st.markdown('<div class="subtitle">Explora los datos antes de aplicar los modelos</div>', unsafe_allow_html=True)
    eda.analisis_descriptivo(df)
    
    st.header("Modelos Propuestos")
    st.markdown('<div class="subtitle">Selecciona y ejecuta el modelo de tu interés</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:#eaf2fb; border-radius:8px; padding:1em; margin-bottom:1em;'>
    <b>Explicación de los modelos y conceptos de redes neuronales aplicados:</b><br><br>
    <ul>
    <li><b>Regresión (MLPRegressor):</b> Utiliza un perceptrón multicapa (red neuronal feedforward) para predecir la calificación del cliente. El modelo aprende relaciones no lineales entre variables transaccionales/demográficas y la calificación, usando capas ocultas y activación ReLU.</li>
    <li><b>Segmentación (PCA + KMeans):</b> Simula el uso de autoencoders (redes neuronales para reducción de dimensionalidad) mediante PCA para obtener una representación latente de los datos y luego agrupa clientes con KMeans. Los autoencoders permiten encontrar patrones complejos y PCA es una aproximación clásica.</li>
    <li><b>Clasificación (MLPClassifier):</b> Implementa una red neuronal multicapa para predecir la próxima línea de producto. Utiliza softmax en la salida para clasificación multiclase y aprende límites de decisión complejos mediante retropropagación.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Guías de selección de variables
    st.markdown('<div class="subtitle">Selecciona las variables de entrada para los modelos</div>', unsafe_allow_html=True)
    
    with st.expander("📋 Guías de Selección de Variables por Modelo", expanded=False):
        st.markdown("""
        ### 🎯 **Regresión - Predicción de Calificación (Rating)**
        **Variables recomendadas:**
        - ✅ **Variables de transacción**: `Unit price`, `Quantity`, `Total`, `Tax 5%`, `cogs`, `gross income`
        - ✅ **Variables demográficas**: `Gender`, `Customer type`
        - ✅ **Variables de producto**: `Product line`
        - ✅ **Variables de ubicación**: `Branch`, `City`
        - ❌ **Excluir**: `Rating` (variable objetivo), `Date`, `Time` (opcional)
        
        ### 👥 **Segmentación de Clientes (PCA + KMeans)**
        **Variables recomendadas:**
        - ✅ **Variables de comportamiento**: `Total`, `Quantity`, `Unit price`, `gross income`
        - ✅ **Variables demográficas**: `Gender`, `Customer type`
        - ⚠️ **Mínimo 2 variables numéricas** para análisis efectivo
        - ❌ **Evitar**: Variables con alta cardinalidad sin preprocesamiento
        
        ### 🛍️ **Clasificación de Productos (MLPClassifier)**
        **Variables recomendadas:**
        - ✅ **Variables de contexto**: `Total`, `Quantity`, `Unit price`, `Rating`
        - ✅ **Variables demográficas**: `Gender`, `Customer type`, `Branch`
        - ✅ **Variables temporales**: `Date`, `Time` (si disponibles)
        - ❌ **Excluir**: `Product line` (variable objetivo)
        
        ### 🔍 **Detección de Anomalías (Isolation Forest)**
        **Variables recomendadas:**
        - ✅ **Variables transaccionales**: `Total`, `Quantity`, `Unit price`, `Tax 5%`
        - ✅ **Variables de tiempo**: `Date`, `Time` (para patrones temporales)
        - ✅ **Cualquier variable numérica** que pueda tener valores atípicos        - ⚠️ **Mínimo 1 variable** requerida
        """)
    
    # Inicializar session state para selección de variables
    if 'var_select' not in st.session_state:
        st.session_state.var_select = list(df.columns)
    
    # Selector de variables
    col1, col2 = st.columns([2, 1])
    
    with col1:
        variables = st.multiselect(
            "Variables disponibles:", 
            options=list(df.columns), 
            default=st.session_state.var_select, 
            key="var_select_widget",
            help="Selecciona las variables que deseas usar en los modelos. Consulta las guías arriba para recomendaciones específicas."
        )
          # Actualizar session_state cuando cambie la selección
        if variables != st.session_state.var_select:
            st.session_state.var_select = variables
    
    with col2:
        st.markdown("**🔧 Acciones Rápidas:**")
        
        if st.button("🎯 Config. Regresión", help="Seleccionar variables óptimas para regresión"):
            regression_vars = [col for col in df.columns if col not in ['Rating']]
            st.session_state.var_select = regression_vars
            st.rerun()
        
        if st.button("👥 Config. Segmentación", help="Seleccionar variables óptimas para segmentación"):
            seg_vars = [col for col in df.columns if col in ['Total', 'Quantity', 'Unit price', 'gross income', 'Gender', 'Customer type', 'Branch', 'City']]
            if not seg_vars:
                seg_vars = df.select_dtypes(include=['number']).columns.tolist()
            st.session_state.var_select = seg_vars
            st.rerun()
        
        if st.button("🛍️ Config. Clasificación", help="Seleccionar variables óptimas para clasificación"):
            class_vars = [col for col in df.columns if col not in ['Product line']]
            st.session_state.var_select = class_vars
            st.rerun()
        
        if st.button("🔍 Config. Anomalías", help="Seleccionar variables óptimas para detección de anomalías"):
            anomaly_vars = df.select_dtypes(include=['number']).columns.tolist()
            if 'Date' in df.columns:
                anomaly_vars.append('Date')
            if 'Time' in df.columns:
                anomaly_vars.append('Time')
            st.session_state.var_select = anomaly_vars
            st.rerun()
      # === VALIDACIONES POR MODELO ===
    
    # Usar las variables del session state
    variables = st.session_state.var_select
    
    # Información sobre la selección actual
    if variables:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variables Seleccionadas", len(variables))
        with col2:
            numeric_vars = len([v for v in variables if pd.api.types.is_numeric_dtype(df[v])])
            st.metric("Variables Numéricas", numeric_vars)
        with col3:
            categorical_vars = len([v for v in variables if not pd.api.types.is_numeric_dtype(df[v])])
            st.metric("Variables Categóricas", categorical_vars)
    
    # Función para validar variables por modelo
    def validar_variables_modelo(modelo_tipo, variables_seleccionadas, df):
        """Valida si las variables seleccionadas son apropiadas para el modelo"""
        warnings = []
        recommendations = []
        
        if modelo_tipo == "regresion":
            if 'Rating' not in df.columns:
                warnings.append("❌ La variable 'Rating' es necesaria para el modelo de regresión.")
            
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 2:
                warnings.append("⚠️ Se recomienda al menos 2 variables numéricas para mejor rendimiento.")
            
            if 'Rating' in variables_seleccionadas:
                recommendations.append("💡 Se detectó 'Rating' en las variables. Se excluirá automáticamente ya que es la variable objetivo.")
        
        elif modelo_tipo == "segmentacion":
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 2:
                warnings.append("❌ Se necesitan al menos 2 variables numéricas para segmentación efectiva.")
            
            if len(variables_seleccionadas) > 10:
                recommendations.append("💡 Muchas variables seleccionadas. Considera usar PCA para reducir dimensionalidad.")
        
        elif modelo_tipo == "clasificacion":
            if 'Product line' not in df.columns:
                warnings.append("❌ La variable 'Product line' es necesaria para el modelo de clasificación.")
            
            if 'Product line' in variables_seleccionadas:
                recommendations.append("💡 Se detectó 'Product line' en las variables. Se excluirá automáticamente ya que es la variable objetivo.")
            
            categorical_vars = [v for v in variables_seleccionadas if not pd.api.types.is_numeric_dtype(df[v])]
            if len(categorical_vars) > 5:
                recommendations.append("💡 Muchas variables categóricas. El modelo puede tardar más en entrenar.")
        
        elif modelo_tipo == "anomalias":
            numeric_vars = [v for v in variables_seleccionadas if pd.api.types.is_numeric_dtype(df[v])]
            if len(numeric_vars) < 1:
                warnings.append("❌ Se necesita al menos 1 variable numérica para detección de anomalías.")
            
            if len(variables_seleccionadas) == 1:
                recommendations.append("💡 Con solo 1 variable, las anomalías serán unidimensionales. Considera añadir más variables.")
        
        return warnings, recommendations
    
    # Selector de modelo
    opcion = st.selectbox("Selecciona un modelo:", [
        "Regresión: Predicción de Calificación",
        "Segmentación de Clientes", 
        "Clasificación de Producto",
        "Detección de Anomalías (Isolation Forest)"
    ], key="modelo_select")
    
    # Validaciones específicas por modelo
    if opcion == "Regresión: Predicción de Calificación":
        warnings, recommendations = validar_variables_modelo("regresion", variables, df)
    elif opcion == "Segmentación de Clientes":
        warnings, recommendations = validar_variables_modelo("segmentacion", variables, df)
    elif opcion == "Clasificación de Producto":
        warnings, recommendations = validar_variables_modelo("clasificacion", variables, df)
    elif opcion == "Detección de Anomalías (Isolation Forest)":
        warnings, recommendations = validar_variables_modelo("anomalias", variables, df)
    else:
        warnings, recommendations = [], []
    
    # Mostrar validaciones
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    if recommendations:
        with st.expander("💡 Recomendaciones para optimizar el modelo", expanded=False):
            for rec in recommendations:
                st.info(rec)
    
    # === SECCIÓN DE MODELOS ===
    
    if opcion == "Regresión: Predicción de Calificación":
        st.subheader("🎯 Modelo de Regresión (MLPRegressor)")
        st.markdown("""
        **Objetivo:** Predecir la calificación del cliente basándose en variables transaccionales y demográficas.
        
        **Algoritmo:** Red neuronal multicapa que aprende relaciones no lineales complejas entre las variables de entrada y la calificación del cliente.
        """)
        
        can_train = 'Rating' in df.columns and len([v for v in variables if pd.api.types.is_numeric_dtype(df[v])]) >= 1
        
        if can_train:
            if st.button("🚀 Entrenar modelo de regresión", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        input_vars = [v for v in variables if v != 'Rating']
                        modelo, preproc, resultados = modelo_1_regresion.entrenar_regresion(df[input_vars + ['Rating']].dropna())
                        
                        st.success("✅ Entrenamiento finalizado exitosamente.")
                        
                        # Métricas en columnas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MSE (Error Cuadrático Medio)", f"{resultados['MSE']:.3f}")
                        with col2:
                            st.metric("MAE (Error Absoluto Medio)", f"{resultados['MAE']:.3f}")
                        with col3:
                            st.metric("R² (Coeficiente de Determinación)", f"{resultados['R2']:.3f}")
                        
                        # Gráfico de predicciones vs valores reales
                        st.subheader("📊 Predicciones vs Valores Reales")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            chart_data = pd.DataFrame({
                                "Real": resultados['y_test'].values,
                                "Predicción": resultados['y_pred']
                            })
                            st.line_chart(chart_data, height=400)
                        
                        with col2:
                            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                            plt.style.use('seaborn-v0_8-whitegrid')
                            
                            sns.scatterplot(x=resultados['y_test'], y=resultados['y_pred'], ax=ax_scatter, alpha=0.7, color='#3498db')
                            
                            min_val = min(min(resultados['y_test']), min(resultados['y_pred']))
                            max_val = max(max(resultados['y_test']), max(resultados['y_pred']))
                            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
                            
                            ax_scatter.set_xlabel("Valores Reales", fontsize=12)
                            ax_scatter.set_ylabel("Valores Predichos", fontsize=12)
                            ax_scatter.set_title("Predicciones vs Valores Reales", fontsize=14, fontweight='bold')
                            ax_scatter.legend()
                            st.pyplot(fig_scatter)

                        # Análisis de Residuos
                        st.subheader("🔍 Análisis de Residuos")
                        try:
                            residuos = resultados['y_test'].values - resultados['y_pred']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_res, ax_res = plt.subplots(figsize=(8, 6))
                                sns.scatterplot(x=resultados['y_pred'], y=residuos, ax=ax_res, alpha=0.7, color='#e74c3c')
                                ax_res.hlines(y=0, xmin=min(resultados['y_pred']), xmax=max(resultados['y_pred']), 
                                            color='black', linestyle='--', linewidth=2)
                                ax_res.set_xlabel("Valores Predichos", fontsize=12)
                                ax_res.set_ylabel("Residuos", fontsize=12)
                                ax_res.set_title("Residuos vs Predicciones", fontsize=14, fontweight='bold')
                                st.pyplot(fig_res)
                            
                            with col2:
                                fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                                sns.histplot(residuos, kde=True, ax=ax_hist, color='#9b59b6', alpha=0.7)
                                ax_hist.axvline(x=0, color='black', linestyle='--', linewidth=2)
                                ax_hist.set_xlabel("Residuos", fontsize=12)
                                ax_hist.set_ylabel("Frecuencia", fontsize=12)
                                ax_hist.set_title("Distribución de Residuos", fontsize=14, fontweight='bold')
                                st.pyplot(fig_hist)
                                
                        except Exception as e_res:
                            st.warning(f"No se pudo generar el análisis de residuos: {e_res}")

                        # Importancia de variables
                        st.subheader("📈 Importancia de Variables")
                        if hasattr(modelo, 'feature_importances_'):
                            importancias = modelo.feature_importances_
                        elif hasattr(modelo, 'coefs_'):
                            importancias = np.abs(modelo.coefs_[0]).sum(axis=1)
                        else:
                            importancias = None
                        
                        if importancias is not None:
                            try:
                                feature_names = preproc.get_feature_names_out()
                            except Exception:
                                feature_names = [f"var_{i}" for i in range(len(importancias))]
                            
                            imp_df = pd.DataFrame({"Variable": feature_names, "Importancia": importancias})
                            imp_df = imp_df.sort_values("Importancia", ascending=False).head(15)
                            
                            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                            sns.barplot(data=imp_df, x="Importancia", y="Variable", ax=ax_imp, palette="viridis")
                            ax_imp.set_title("Top 15 Variables Más Importantes", fontsize=14, fontweight='bold')
                            ax_imp.set_xlabel("Importancia", fontsize=12)
                            plt.tight_layout()
                            st.pyplot(fig_imp)
                            
                            with st.expander("📋 Ver tabla completa de importancias", expanded=False):
                                st.dataframe(imp_df, use_container_width=True)
                        else:
                            st.info("💡 El modelo MLPRegressor no permite calcular importancia de variables directamente.")
                            
                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento del modelo: {e}")
        else:
            st.info("⚠️ Verifica que las variables seleccionadas cumplan con los requisitos mostrados arriba.")
            
    elif opcion == "Segmentación de Clientes":
        st.subheader("👥 Segmentación de Clientes (PCA + KMeans)")
        st.markdown("""
        **Objetivo:** Agrupar clientes en segmentos homogéneos basándose en patrones de comportamiento.
        
        **Algoritmo:** Reducción de dimensionalidad con PCA seguida de clustering con KMeans para identificar grupos naturales en los datos.
        """)
        
        numeric_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        can_segment = len(numeric_vars) >= 2
        
        if can_segment:
            col1, col2 = st.columns([1, 3])
            with col1:
                n_clusters = st.slider("Número de segmentos", min_value=2, max_value=8, value=3)
            with col2:
                st.info(f"💡 Se usarán {len(numeric_vars)} variables numéricas para la segmentación")
            
            if st.button("🚀 Ejecutar segmentación", type="primary"):
                with st.spinner("Segmentando clientes..."):
                    try:
                        df_seg, kmeans, pca, preproc = modelo_2_segmentacion.segmentar_clientes(df[variables].dropna(), n_clusters=n_clusters)
                        caracteristicas = modelo_2_segmentacion.caracterizar_segmentos(df_seg)
                        
                        st.success("✅ Segmentación completada exitosamente.")
                        
                        # Métricas de segmentación
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Segmentos Creados", n_clusters)
                        with col2:
                            st.metric("Clientes Segmentados", len(df_seg))
                        with col3:
                            try:
                                from sklearn.metrics import silhouette_score
                                sil_score = silhouette_score(pca.transform(preproc.transform(df[variables].dropna())), df_seg['Segmento'])
                                st.metric("Silhouette Score", f"{sil_score:.3f}")
                            except:
                                st.metric("Variables Usadas", len(variables))
                        
                        # Visualización PCA
                        st.subheader("🎯 Visualización de Segmentos (PCA)")
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
                                ax.scatter(centroid_x, centroid_y, c='black', s=200, marker='x', linewidths=3)
                            
                            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)", fontsize=12)
                            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)", fontsize=12)
                            ax.set_title("Segmentación de Clientes (Espacio PCA)", fontsize=14, fontweight='bold')
                            ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("**📊 Información de Segmentos:**")
                            segment_counts = df_seg['Segmento'].value_counts().sort_index()
                            for segment, count in segment_counts.items():
                                percentage = (count / len(df_seg)) * 100
                                st.write(f"**Segmento {segment}:** {count} clientes ({percentage:.1f}%)")
                            
                            st.markdown("**🔍 Varianza Explicada:**")
                            total_variance = sum(pca.explained_variance_ratio_[:2]) * 100
                            st.write(f"PC1 + PC2: {total_variance:.1f}%")

                        # Características por segmento
                        st.subheader("📋 Características por Segmento")
                        st.dataframe(caracteristicas.round(3), use_container_width=True)

                # Análisis por variables
                        st.subheader("📈 Análisis de Variables por Segmento")
                        numeric_vars_for_boxplot = [col for col in variables if pd.api.types.is_numeric_dtype(df[col])]

                        if numeric_vars_for_boxplot:
                            selected_vars_for_boxplot = st.multiselect(
                                "Selecciona variables numéricas para analizar por segmento:",
                                options=numeric_vars_for_boxplot,
                                default=numeric_vars_for_boxplot[:min(len(numeric_vars_for_boxplot), 2)],
                                key="segment_boxplot_vars"
                            )

                            if selected_vars_for_boxplot:
                                for i, var_to_plot in enumerate(selected_vars_for_boxplot):
                                    if i % 2 == 0:
                                        col1, col2 = st.columns(2)
                                    
                                    with col1 if i % 2 == 0 else col2:
                                        fig_box_seg, ax_box_seg = plt.subplots(figsize=(8, 6))
                                        plt.style.use('seaborn-v0_8-whitegrid')
                                        
                                        sns.boxplot(data=df_seg, x='Segmento', y=var_to_plot, ax=ax_box_seg, palette='Set2')
                                        sns.stripplot(data=df_seg, x='Segmento', y=var_to_plot, ax=ax_box_seg, 
                                                    color='black', alpha=0.3, size=3)
                                        
                                        ax_box_seg.set_title(f"Distribución de {var_to_plot} por Segmento", 
                                                           fontsize=12, fontweight='bold')
                                        ax_box_seg.set_xlabel("Segmento", fontsize=10)
                                        ax_box_seg.set_ylabel(var_to_plot, fontsize=10)
                                        plt.tight_layout()
                                        st.pyplot(fig_box_seg)
                                
                                with st.expander("📊 Análisis Estadístico Detallado", expanded=False):
                                    for var in selected_vars_for_boxplot:
                                        st.markdown(f"**{var}:**")
                                        stats_by_segment = df_seg.groupby('Segmento')[var].agg(['mean', 'std', 'median']).round(3)
                                        st.dataframe(stats_by_segment, use_container_width=True)
                                        
                                # Descargar resultados de segmentación
                                st.subheader("💾 Descargar Resultados")
                                csv_data = df_seg.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar datos segmentados (CSV)",
                                    data=csv_data,
                                    file_name="clientes_segmentados.csv",
                                    mime="text/csv",
                                    help="Descarga el dataset con la asignación de segmentos para cada cliente"
                                )
                        else:
                            st.info("No hay variables numéricas disponibles para análisis por segmento.")
                            
                    except Exception as e:
                        st.error(f"❌ Error durante la segmentación: {e}")
        else:
            st.info("⚠️ Verifica que tengas al menos 2 variables numéricas seleccionadas para la segmentación.")
            
    elif opcion == "Clasificación de Producto":
        st.subheader("🛍️ Clasificación de Producto (MLPClassifier)")
        st.markdown("""
        **Objetivo:** Predecir la próxima línea de producto que comprará un cliente basándose en su historial y características.
        
        **Algoritmo:** Red neuronal multicapa con clasificación multiclase que aprende patrones de compra complejos.
        """)
        
        can_classify = 'Product line' in df.columns
        
        if can_classify:
            product_lines = df['Product line'].unique()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"💡 Se entrenará para clasificar entre {len(product_lines)} líneas de producto")
            with col2:
                st.metric("Clases Únicas", len(product_lines))
            
            with st.expander("📊 Distribución de Líneas de Producto", expanded=False):
                class_dist = df['Product line'].value_counts()
                st.bar_chart(class_dist)
                st.dataframe(class_dist.reset_index().rename(columns={'index': 'Línea de Producto', 'Product line': 'Cantidad'}))
            
            if st.button("🚀 Entrenar modelo de clasificación", type="primary"):
                with st.spinner("Entrenando modelo de clasificación..."):
                    try:
                        input_vars = [v for v in variables if v != 'Product line']
                        modelo, preproc, resultados = modelo_3_clasificacion.entrenar_clasificacion(df[input_vars + ['Product line']].dropna())
                        
                        st.success("✅ Entrenamiento finalizado exitosamente.")
                        
                        # Métricas principales
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Exactitud (Accuracy)", f"{resultados['accuracy']:.3f}")
                        with col2:
                            precision_avg = np.mean([v['precision'] for v in resultados['reporte'].values() if isinstance(v, dict) and 'precision' in v])
                            st.metric("Precisión Promedio", f"{precision_avg:.3f}")
                        with col3:
                            recall_avg = np.mean([v['recall'] for v in resultados['reporte'].values() if isinstance(v, dict) and 'recall' in v])
                            st.metric("Recall Promedio", f"{recall_avg:.3f}")

                        # Reporte de clasificación
                        st.subheader("📋 Reporte de Clasificación Detallado")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            plt.style.use('seaborn-v0_8-whitegrid')
                            
                            sns.heatmap(resultados['matriz_confusion'], 
                                      annot=True, 
                                      fmt='d', 
                                      cmap='Blues', 
                                      ax=ax,
                                      xticklabels=product_lines,
                                      yticklabels=product_lines)
                            ax.set_xlabel('Predicción', fontsize=12)
                            ax.set_ylabel('Real', fontsize=12)
                            ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
                            plt.xticks(rotation=45)
                            plt.yticks(rotation=0)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            reporte_df = pd.DataFrame(resultados['reporte']).transpose()
                            class_metrics = reporte_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
                            st.dataframe(class_metrics.round(3), use_container_width=True)

                        # Análisis de importancia de variables
                        st.subheader("📈 Análisis de Importancia de Variables")
                        if hasattr(modelo, 'coefs_') and preproc is not None:
                            try:
                                feature_names = preproc.get_feature_names_out()
                                
                                if len(modelo.coefs_) > 0:
                                    importancias = np.mean(np.abs(modelo.coefs_[0]), axis=1)

                                    if len(importancias) == len(feature_names):
                                        imp_df_cls = pd.DataFrame({"Variable": feature_names, "Importancia Estimada": importancias})
                                        imp_df_cls = imp_df_cls.sort_values("Importancia Estimada", ascending=False).head(15)
                                        
                                        fig_imp_cls, ax_imp_cls = plt.subplots(figsize=(12, 8))
                                        sns.barplot(data=imp_df_cls, x="Importancia Estimada", y="Variable", ax=ax_imp_cls, palette="viridis")
                                        ax_imp_cls.set_title("Top 15 Variables Más Importantes (MLP)", fontsize=14, fontweight='bold')
                                        plt.tight_layout()
                                        st.pyplot(fig_imp_cls)
                                        
                                        st.caption("Nota: La 'importancia' para MLPClassifier se estima a partir de la magnitud de los pesos de la primera capa y es experimental.")
                                    else:
                                        st.info("No se pudo emparejar la importancia con los nombres de las variables.")
                                else:
                                    st.info("El modelo no tiene coeficientes disponibles para estimar la importancia.")
                            except Exception as e_imp:
                                st.warning(f"No se pudo calcular la importancia de variables: {e_imp}")
                        else:
                            st.info("💡 El modelo MLPClassifier no permite extracción directa de importancia de variables.")
                            
                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento del modelo: {e}")
        else:
            st.info("⚠️ La variable 'Product line' es necesaria para este modelo.")
            
    elif opcion == "Detección de Anomalías (Isolation Forest)":
        st.subheader("🔍 Detección de Anomalías (Isolation Forest)")
        st.markdown("""
        **Objetivo:** Identificar observaciones atípicas o anómalas en los datos del supermercado.
        
        **Algoritmo:** Isolation Forest aísla anomalías mediante divisiones aleatorias en árboles. 
        Las observaciones que requieren menos divisiones para ser aisladas se consideran anómalas.
        
        **Aplicaciones:** Detección de fraudes, errores de captura, comportamientos inusuales de compra.
        """)
        
        numeric_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        can_detect = len(variables) >= 1
        
        if can_detect:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                contamination = st.slider(
                    "Proporción esperada de anomalías", 
                    min_value=0.01, 
                    max_value=0.2, 
                    value=0.05, 
                    step=0.01,
                    help="Porcentaje esperado de datos anómalos en el dataset"
                )
            
            with col2:
                st.info(f"💡 Se usarán {len(variables)} variables, {len(numeric_vars)} de ellas numéricas")
                
            if st.button("🚀 Detectar anomalías", type="primary"):
                with st.spinner("Detectando anomalías..."):
                    try:
                        df_anom, modelo, preproc = modelo_4_anomalias.detectar_anomalias(df[variables].dropna(), variables, contamination)
                        
                        st.success("✅ Detección de anomalías completada.")
                        
                        # Métricas principales
                        total_anomalias = sum(df_anom['Anomalía'] == 'Sí')
                        total_normales = sum(df_anom['Anomalía'] == 'No')
                        
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

                        # Vista previa de anomalías
                        st.subheader("👀 Vista Previa de Resultados")
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Datos Normales (muestra):**")
                            normales = df_anom[df_anom['Anomalía'] == 'No'].head()
                            if len(normales) > 0:
                                st.dataframe(normales.head(), use_container_width=True)
                            else:
                                st.info("No hay datos normales para mostrar.")
                        
                        with col2:
                            st.markdown("**Anomalías Detectadas:**")
                            anomalias = df_anom[df_anom['Anomalía'] == 'Sí']
                            if len(anomalias) > 0:
                                st.dataframe(anomalias.head(), use_container_width=True)
                            else:
                                st.info("No se detectaron anomalías con los parámetros actuales.")

                        # Distribución de anomalías
                        st.subheader("📊 Distribución de Anomalías")
                        anomaly_dist = df_anom['Anomalía'].value_counts()
                        
                        fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
                        colors = ['#2ecc71', '#e74c3c']  # Verde para normal, rojo para anomalías
                        bars = ax_dist.bar(anomaly_dist.index, anomaly_dist.values, color=colors, alpha=0.8)
                        
                        # Añadir etiquetas en las barras
                        for bar, value in zip(bars, anomaly_dist.values):
                            height = bar.get_height()
                            ax_dist.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                       f'{value}\n({value/len(df_anom)*100:.1f}%)', 
                                       ha='center', va='bottom', fontweight='bold')
                        
                        ax_dist.set_title('Distribución de Anomalías vs Datos Normales', fontsize=14, fontweight='bold')
                        ax_dist.set_xlabel('Tipo de Dato', fontsize=12)
                        ax_dist.set_ylabel('Cantidad', fontsize=12)
                        plt.style.use('seaborn-v0_8-whitegrid')
                        plt.tight_layout()
                        st.pyplot(fig_dist)

                        # Visualización detallada de anomalías
                        st.subheader("🎯 Visualización Detallada de Anomalías")
                        
                        numeric_cols_for_viz = [col for col in df_anom.columns if col != 'Anomalía' and pd.api.types.is_numeric_dtype(df_anom[col])]
                        
                        if numeric_cols_for_viz:
                            selected_viz_vars = st.multiselect(
                                "Selecciona hasta 2 variables numéricas para análisis visual:",
                                options=numeric_cols_for_viz,
                                default=numeric_cols_for_viz[:min(len(numeric_cols_for_viz), 2)],
                                max_selections=2,
                                key="viz_anom_vars_modern"
                            )

                            if selected_viz_vars:
                                plt.style.use('seaborn-v0_8-whitegrid')

                                if len(selected_viz_vars) == 1:
                                    var_to_plot = selected_viz_vars[0]
                                    st.write(f"Distribución de '{var_to_plot}' para datos normales vs. anomalías:")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.histplot(data=df_anom, x=var_to_plot, hue='Anomalía', kde=True, ax=ax, 
                                                   palette={'No':'#5dade2', 'Sí':'#e74c3c'})
                                        ax.set_title(f"Distribución de {var_to_plot}", fontsize=14, fontweight='bold')
                                        ax.set_xlabel(var_to_plot, fontsize=12)
                                        ax.set_ylabel("Frecuencia", fontsize=12)
                                        ax.legend(title='Tipo de Dato')
                                        st.pyplot(fig)

                                    with col2:
                                        fig_box, ax_box = plt.subplots(figsize=(10, 6))
                                        sns.boxplot(data=df_anom, x='Anomalía', y=var_to_plot, ax=ax_box, 
                                                  palette={'No':'#5dade2', 'Sí':'#e74c3c'})
                                        ax_box.set_title(f"Boxplot de {var_to_plot} por Tipo de Dato", fontsize=14, fontweight='bold')
                                        ax_box.set_xlabel("Tipo de Dato", fontsize=12)
                                        ax_box.set_ylabel(var_to_plot, fontsize=12)
                                        st.pyplot(fig_box)

                                elif len(selected_viz_vars) == 2:
                                    var1, var2 = selected_viz_vars[0], selected_viz_vars[1]
                                    st.write(f"Relación entre '{var1}' y '{var2}' para datos normales vs. anomalías:")
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.scatterplot(data=df_anom, x=var1, y=var2, hue='Anomalía', style='Anomalía', ax=ax, 
                                                  palette={'No':'#5dade2', 'Sí':'#e74c3c'}, markers={'No':'o', 'Sí':'X'}, s=100)
                                    ax.set_title(f"Relación entre {var1} y {var2}", fontsize=14, fontweight='bold')
                                    ax.set_xlabel(var1, fontsize=12)
                                    ax.set_ylabel(var2, fontsize=12)
                                    ax.legend(title='Tipo de Dato')
                                    st.pyplot(fig)
                                  # Estadísticas descriptivas
                                st.subheader("📈 Estadísticas Descriptivas por Tipo de Dato")
                                try:
                                    desc_stats = df_anom.groupby('Anomalía')[selected_viz_vars].agg(['mean', 'std', 'min', 'max']).reset_index()
                                    desc_stats.columns = [' - '.join(col).strip() if isinstance(col, tuple) and col[1] else col[0] for col in desc_stats.columns.values]
                                    st.dataframe(desc_stats, use_container_width=True)
                                except Exception:
                                    st.dataframe(df_anom.groupby('Anomalía')[selected_viz_vars].describe(), use_container_width=True)
                            else:
                                st.info("Selecciona al menos una variable para visualizar anomalías.")
                        else:
                            st.info("No hay variables numéricas disponibles para visualización detallada de anomalías.")
                        
                        # Descargar resultados de anomalías
                        st.subheader("💾 Descargar Resultados")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_all = df_anom.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Descargar todos los datos con etiquetas",
                                data=csv_all,
                                file_name="datos_con_anomalias.csv",
                                mime="text/csv",
                                help="Descarga el dataset completo con la clasificación de anomalías"
                            )
                        
                        with col2:
                            if len(anomalias) > 0:
                                csv_anomalies = anomalias.to_csv(index=False).encode('utf-8')                                st.download_button(
                                    label="📥 Descargar solo anomalías",
                                    data=csv_anomalies,
                                    file_name="solo_anomalias.csv",
                                    mime="text/csv",
                                    help="Descarga únicamente los registros identificados como anómalos"
                                )
                            else:
                                st.info("No hay anomalías para descargar")
                            
                    except Exception as e:
                        st.error(f"❌ Error durante la detección de anomalías: {e}")
        else:
            st.info("⚠️ Selecciona al menos una variable para detectar anomalías.")
    
    else:
        st.info("Selecciona un modelo para ver su descripción y cómo se aplica a los datos del supermercado.")
        
else:
    st.info("Por favor, carga un archivo de datos para comenzar el análisis y la modelización.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em; margin-top: 2em;'>
    <p><strong>Modelos Conceptuales de Redes Neuronales para Supermercados</strong></p>
    <p>Maestría en Analítica de Datos - Universidad Central</p>
    <p>Desarrollado con ❤️ usando Streamlit y Scikit-learn</p>
    <p><em>Versión 2.0 - Noviembre 2024</em></p>
</div>
""", unsafe_allow_html=True)
