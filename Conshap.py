import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import base64
import os
import shap
import matplotlib.pyplot as plt
import numpy as np  # Importar numpy para manejar arrays

# Rutas
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RedNeuronal.h5")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_encoders.pkl")

# Columnas esperadas (excluyendo 'NOCOBRO')
expected_columns = ['COMUNA', 'REGION', 'URBANIDAD', 'FPAGO', 'TIPOBENEFICIO',
                    'COBROMARZO', 'SEXO', 'ECIVIL', 'NACIONALIDAD']

# Cargar el modelo y los encoders
@st.cache_resource
def load_resources():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"El archivo del modelo no se encontró en la ruta: {MODEL_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"El archivo de encoders no se encontró en la ruta: {ENCODER_PATH}")
        
        model = load_model(MODEL_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        return model, label_encoders
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo o encoders: {str(e)}")
        return None, None

model, label_encoders = load_resources()

# Función para obtener la imagen codificada en base64
def get_base64_encoded_image(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded
    except Exception:
        st.sidebar.error("No se pudo cargar la imagen del logo.")
        return None

# Obtener la imagen codificada
logo_base64 = get_base64_encoded_image('logo.png')

# Mostrar la imagen alineada a la izquierda arriba de los títulos
if logo_base64:
    st.markdown(
        f"""
        <div style="text-align: left">
            <img src="data:image/png;base64,{logo_base64}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

# Línea horizontal gruesa
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

# Título y descripción de la aplicación
st.title("Probabilidad para el Cobro de Beneficios: Modelo Predictivo")
st.markdown("""
Esta aplicación predice si un beneficiario cobrará sus beneficios basándose en sus características demográficas, geográficas e información del beneficio.

Por favor, seleccione la información solicitada en la barra lateral y presione **Predecir**.
""")

# Línea horizontal
st.markdown("---")

# Input de datos
st.sidebar.header("Seleccione las Características:")
input_data = {}

# Diccionarios de mapeo para las opciones
sex_display_options = ['FEMENINO', 'MASCULINO', 'INDETERMINADO']
sex_actual_values = ['F', 'M', 'E']
sex_option_map = dict(zip(sex_display_options, sex_actual_values))

nationality_display_options = ['CHILENO', 'EXTRANJERO']
nationality_actual_values = ['C', 'E']
nationality_option_map = dict(zip(nationality_display_options, nationality_actual_values))

# Organizar los campos de entrada en secciones
with st.sidebar.expander("Información Demográfica", expanded=True):
    # Usar columnas para los selectores
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de SEXO
        input_selection = st.selectbox("SEXO:", sex_display_options)
        input_data['SEXO'] = sex_option_map[input_selection]
    with col2:
        # Selector de NACIONALIDAD
        input_selection = st.selectbox("NACIONALIDAD:", nationality_display_options)
        input_data['NACIONALIDAD'] = nationality_option_map[input_selection]
    
    # Selector de ESTADO CIVIL
    display_label = 'ESTADO CIVIL'
    if label_encoders and 'ECIVIL' in label_encoders:
        options = label_encoders['ECIVIL'].classes_
        input_data['ECIVIL'] = st.selectbox(f"{display_label}:", options)
    else:
        st.warning(f"No se encontraron opciones para ECIVIL. Verifica los encoders.")

with st.sidebar.expander("Información Geográfica", expanded=True):
    # Usar columnas para los selectores
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(['REGION', 'COMUNA']):
        with [col1, col2][i%2]:
            if label_encoders and col in label_encoders:
                options = label_encoders[col].classes_
                input_data[col] = st.selectbox(f"{col}:", options)
            else:
                st.warning(f"No se encontraron opciones para {col}. Verifica los encoders.")
    
    # URBANIDAD
    if label_encoders and 'URBANIDAD' in label_encoders:
        options = label_encoders['URBANIDAD'].classes_
        input_data['URBANIDAD'] = st.selectbox("URBANIDAD:", options)
    else:
        st.warning("No se encontraron opciones para URBANIDAD. Verifica los encoders.")

with st.sidebar.expander("Información del Beneficio", expanded=True):
    # Usar columnas para los selectores
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(['FPAGO', 'TIPOBENEFICIO']):
        with [col1, col2][i%2]:
            if label_encoders and col in label_encoders:
                options = label_encoders[col].classes_
                input_data[col] = st.selectbox(f"{col}:", options)
            else:
                st.warning(f"No se encontraron opciones para {col}. Verifica los encoders.")
    
    # COBROMARZO
    if label_encoders and 'COBROMARZO' in label_encoders:
        options = label_encoders['COBROMARZO'].classes_
        input_data['COBROMARZO'] = st.selectbox("COBROMARZO:", options)
    else:
        st.warning("No se encontraron opciones para COBROMARZO. Verifica los encoders.")

# Crear un placeholder para los resultados
result_placeholder = st.empty()

# Botón de predicción
if st.sidebar.button("Predecir"):
    if model and label_encoders:
        try:
            with st.spinner('Realizando predicción...'):
                # Crear DataFrame
                input_df = pd.DataFrame([input_data])

                # Transformar datos con los encoders
                for col in expected_columns:
                    if col in label_encoders:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                    else:
                        # Convertir a numérico si hubiera columnas adicionales
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                # Verificar si hay valores nulos después de la conversión
                if input_df.isnull().any().any():
                    st.error("Por favor, asegúrese de que todos los campos estén correctamente llenos.")
                    st.write("Datos ingresados:")
                    st.write(input_df)
                else:
                    # Asegurar que el orden de las columnas coincide con el esperado por el modelo
                    input_df = input_df[expected_columns]
                    # Convertir los tipos de datos a float32
                    input_df = input_df.astype('float32')
                    # Hacer predicción
                    prediction_prob = model.predict(input_df)
                    prediction = (prediction_prob > 0.5).astype(int)
                    resultado = "NO COBRARÁ" if prediction[0][0] == 1 else "SÍ COBRARÁ"

                    # === Añadir código para SHAP ===

                    # Crear un conjunto de datos de fondo
                    background_data = input_df.copy()
                    for _ in range(50):
                        background_data = pd.concat([background_data, input_df], ignore_index=True)

                    # Inicializar el DeepExplainer
                    explainer = shap.DeepExplainer(model, background_data.values)

                    # Calcular los valores SHAP para la predicción actual
                    shap_values = explainer.shap_values(input_df.values)

                    # Manejar el caso en que shap_values es una lista
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # Exprimir la última dimensión si es de tamaño 1
                    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
                        shap_values = shap_values[:, :, 0]

                    # Verificar que las formas de shap_values y input_df coinciden
                    st.write("Forma de shap_values después de ajustar:", shap_values.shape)
                    st.write("Forma de input_df:", input_df.shape)

                    if shap_values.shape != input_df.shape:
                        st.error(f"La forma de los shap_values {shap_values.shape} no coincide con la forma de los datos de entrada {input_df.shape}.")
                    else:
                        # Visualizar los valores SHAP
                        st.markdown("### Interpretación de la Predicción:")
                        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.close(fig)

            # Mostrar el resultado en el placeholder
            with result_placeholder:
                st.success('Predicción completada con éxito.')
                st.markdown(f"## Resultado de la Predicción:")
                st.markdown(f"### El beneficiario **{resultado}** su beneficio.")
                st.markdown(f"**Probabilidad estimada de no cobro:** {prediction_prob[0][0]:.2%}")

        except Exception as e:
            st.error(f"Error durante la predicción: {str(e)}")
    else:
        st.error("El modelo no está cargado correctamente.")
else:
    # Mostrar mensaje inicial en el placeholder
    with result_placeholder:
        st.markdown("## Resultado de la Predicción:")
        st.info("Por favor, ingrese los datos y presione **Predecir** para ver el resultado.")

# Línea horizontal gruesa
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

# Texto al final de la aplicación
st.markdown("Tarea grupal desarrollada por: **Rubén Galaz y Francisco Becker**")
