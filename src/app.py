import streamlit as st
import joblib

clf = joblib.load('../models/modelo_de_clasificacion.pkl')
vectorizer = joblib.load('../models/vectorizador.pkl')

st.title("Clasificador de productos")

user_input = st.text_input("Ingresa el nombre de producto del Supermercado para clasificarlo. (Ej: Papas fritas marco polo 230 gramos)")

if user_input:
    user_input_vectorized = vectorizer.transform([user_input])

    prediction = clf.predict(user_input_vectorized)

    st.write("El producto pertenece a la categoría:", prediction[0])