# cd "C:\FBECKER\OneDrive\Bases\Tarea DL"
# .\venv\Scripts\activate
# streamlit run Tarea-Streamlit.py

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import base64
import os

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

# Cargar el dataframe
# Reemplaza esto con tu propio código para cargar tu dataframe
# Por ejemplo: df = pd.read_csv('tu_archivo.csv')
# Para este ejemplo, crearé un dataframe de muestra
data = {
    'COMUNA': ['Santiago', 'Valparaíso', 'Concepción', 'Santiago', 'Valparaíso', 'Concepción', 'Santiago'],
    'REGION': ['Metropolitana', 'Valparaíso', 'Biobío', 'Metropolitana', 'Valparaíso', 'Biobío', 'Metropolitana'],
    'URBANIDAD': ['Urbano', 'Urbano', 'Urbano', 'Rural', 'Rural', 'Urbano', 'Urbano'],
    'FPAGO': ['Efectivo', 'Tarjeta', 'Efectivo', 'Cheque', 'Efectivo', 'Tarjeta', 'Efectivo'],
    'TIPOBENEFICIO': ['Tipo A', 'Tipo B', 'Tipo A', 'Tipo C', 'Tipo B', 'Tipo A', 'Tipo C'],
    'COBROMARZO': [1, 0, 1, 1, 0, 1, 1],
    'SEXO': ['Femenino', 'Masculino', 'Femenino', 'Femenino', 'Masculino', 'Femenino', 'Masculino'],
    'ECIVIL': ['Soltero', 'Casado', 'Soltero', 'Casado', 'Soltero', 'Casado', 'Soltero'],
    'NACIONALIDAD': ['Chilena', 'Chilena', 'Extranjera', 'Chilena', 'Extranjera', 'Chilena', 'Chilena'],
    'NOCOBRO': [0, 1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

# Título de la aplicación
st.title('Estadísticas del Dataframe')

# Mostrar el dataframe
st.write("Vista previa del dataframe:")
st.dataframe(df)

# Lista de columnas disponibles para selección
columnas = df.columns.tolist()

# Selector de campos
st.sidebar.header('Seleccione los campos para analizar')
campo_uno = st.sidebar.selectbox('Seleccione el primer campo:', [''] + columnas)
campo_dos = st.sidebar.selectbox('Seleccione el segundo campo (opcional):', [''] + columnas)

# Función para mostrar estadísticas
def mostrar_estadisticas(campo1, campo2=None):
    if campo1 and not campo2:
        st.subheader(f'Estadísticas para {campo1}')
        conteo = df[campo1].value_counts()
        st.bar_chart(conteo)
        st.write(conteo)
    elif campo1 and campo2:
        st.subheader(f'Estadísticas para {campo1} y {campo2}')
        conteo_doble = df.groupby([campo1, campo2]).size().unstack()
        st.write(conteo_doble)
        st.bar_chart(conteo_doble)
    else:
        st.write('Por favor, seleccione al menos un campo para mostrar las estadísticas.')

# Llamar a la función con los campos seleccionados
mostrar_estadisticas(campo_uno, campo_dos)

# Ejemplos específicos para mostrar estadísticas adicionales
if 'SEXO' in [campo_uno, campo_dos]:
    st.subheader('Cantidad por Sexo')
    sexo_conteo = df['SEXO'].value_counts()
    st.write(sexo_conteo)
    st.bar_chart(sexo_conteo)

if 'REGION' in [campo_uno, campo_dos]:
    st.subheader('Cantidad por Región')
    region_conteo = df['REGION'].value_counts()
    st.write(region_conteo)
    st.bar_chart(region_conteo)

if 'FPAGO' in [campo_uno, campo_dos]:
    st.subheader('Cantidad por Forma de Pago')
    fpago_conteo = df['FPAGO'].value_counts()
    st.write(fpago_conteo)
    st.bar_chart(fpago_conteo)