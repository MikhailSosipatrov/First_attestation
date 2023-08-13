import streamlit as st
import pandas as pd



st.image("logo-3.png")

st.header("Анализ данных")
st.markdown("---")

st.subheader("Загрузите набор данных  в формате csv ниже:")
file = st.file_uploader('', type="csv", accept_multiple_files=False)


if file is not None:
    df = pd.read_csv(file)
    df = df.drop(df.columns[0], axis = 1)
    st.dataframe(df)
    options = st.multiselect("Выберите переменные(не более двух)", [col for col in df])








