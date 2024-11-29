# cd "C:\FBECKER\OneDrive\Bases\Tarea DL"
# .\venv\Scripts\activate
# streamlit run Tarea-Streamlit.py

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import base64

# Rutas
MODEL_PATH = r"/workspaces/DeepLearning/RedNeuronal.h5"
ENCODER_PATH = r"/workspaces/DeepLearning/label_encoders.pkl"

# Columnas esperadas (excluyendo 'NOCOBRO')
expected_columns = ['COMUNA', 'REGION', 'URBANIDAD', 'FPAGO', 'TIPOBENEFICIO',
                    'COBROMARZO', 'SEXO', 'ECIVIL', 'NACIONALIDAD']

# Cargar el modelo y los encoders
@st.cache_resource
def load_resources():
    try:
        model = load_model(MODEL_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        return model, label_encoders
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo o encoders: {str(e)}")
        return None, None

model, label_encoders = load_resources()

# Función para obtener la imagen codificada en base64
def get_base64_encoded_image(image_path):
    with open(image_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Obtener la imagen codificada
logo_base64 = get_base64_encoded_image('logo.png')

# Mostrar la imagen alineada a la derecha arriba de los títulos
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
st.title("Predicción de No Cobro de Beneficios")
st.markdown("""
Esta aplicación predice si un beneficiario cobrará sus beneficios basándose en sus características demográficas y otros factores.

Por favor, ingrese la información solicitada en la barra lateral y presione **Predecir**.
""")

# Línea horizontal
st.markdown("---")

# Input de datos
st.sidebar.header("Ingrese Características de beneficiario:")
input_data = {}

# Organizar los campos de entrada en secciones
st.sidebar.subheader("Información Personal")

# Información Personal
personal_info_cols = ['SEXO', 'ECIVIL', 'NACIONALIDAD']
for col in personal_info_cols:
    options = label_encoders[col].classes_
    input_data[col] = st.sidebar.selectbox(f"{col}:", options)

st.sidebar.subheader("Información Geográfica")

# Información Geográfica
geo_info_cols = ['REGION', 'COMUNA', 'URBANIDAD']
for col in geo_info_cols:
    options = label_encoders[col].classes_
    input_data[col] = st.sidebar.selectbox(f"{col}:", options)

st.sidebar.subheader("Información del Beneficio")

# Información del Beneficio
benefit_info_cols = ['FPAGO', 'TIPOBENEFICIO', 'COBROMARZO']
for col in benefit_info_cols:
    options = label_encoders[col].classes_
    input_data[col] = st.sidebar.selectbox(f"{col}:", options)

# Botón de predicción
if st.sidebar.button("Predecir"):
    if model is not None and label_encoders is not None:
        try:
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

                # Mostrar el resultado con estilo
                st.markdown(f"## Resultado de la Predicción:")
                st.markdown(f"### El beneficiario **{resultado}** su beneficio.")
                st.markdown(f"**Probabilidad estimada de no cobro:** {prediction_prob[0][0]:.2%}")

        except Exception as e:
            st.error(f"Error durante la predicción: {str(e)}")
    else:
        st.error("El modelo no está cargado correctamente.")

# Línea horizontal gruesa
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

# Texto al final de la aplicación
st.markdown("Tarea grupal desarrollada por: **Rubén Galz y Francisco Becker**")
