import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



st.image("logo-3.png")

st.header("Анализ данных")
st.markdown("---")

st.subheader("Загрузите набор данных  в формате csv ниже:")
file = st.file_uploader('', type="csv", accept_multiple_files=False)

figure = plt.figure()
if file is not None:
    df = pd.read_csv(file)
    df = df.drop(df.columns[0], axis = 1)
    st.dataframe(df)
    options = st.multiselect("Выберите переменные(не более двух)", [col for col in df])
    if (len(options) == 2):
        slctd_opt_1, slctd_opt_2 = options
        if (isinstance(df[slctd_opt_1][0], str) and (not isinstance(df[slctd_opt_2][0], str))):

            fig = plt.figure(figsize=(10, 10))
            labels = df[slctd_opt_1].unique()
            sizes = [len(df[df[slctd_opt_1] == i]) for i in labels]
            plt.pie(sizes, labels = labels, autopct='%1.1f%%')
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[slctd_opt_2], kde=True, bins = 100)
            st.pyplot(fig)

        elif (isinstance(df[slctd_opt_2][0], str) and (not isinstance(df[slctd_opt_1][0], str))):

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[slctd_opt_1], kde=True, bins = 100)
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 10))
            labels = df[slctd_opt_2].unique()
            sizes = [len(df[df[slctd_opt_2] == i]) for i in labels]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            st.pyplot(fig)

        else:
            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[slctd_opt_1], kde=True, bins = 100)
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[slctd_opt_2], kde=True, bins = 100)
            st.pyplot(fig)












