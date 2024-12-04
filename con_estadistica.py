import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import base64
import os
import h5py  # Importar h5py para manejar archivos H5
import numpy as np

import streamlit as st
import pandas as pd

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