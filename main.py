import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
import seaborn as sns
from scipy.stats import norm
import numpy as np






#Функция для бутстрапирования



# plt.style.use('ggplot')

def get_bootstrap(
        data_column_1,
        data_column_2,
        boot_it=1000,  # boot iterations
        statistic=np.mean,  # what we try to find
        bootstrap_conf_level=0.95  # confidence level
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)):
        samples_1 = data_column_1.sample(
            boot_len,
            replace=True
        ).values

        samples_2 = data_column_2.sample(
            boot_len,
            replace=True
        ).values

        boot_data.append(statistic(samples_1 - samples_2))
    pd_boot_data = pd.DataFrame(boot_data)

    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])

    p_1 = norm.cdf(
        x=0,
        loc=np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_2 = norm.cdf(
        x=0,
        loc=-np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2

    # Visuals
    _, _, bars = plt.hist(pd_boot_data[0], bins=50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else:
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')

    plt.vlines(quants, ymin=0, ymax=50, linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()

    return {"boot_data": boot_data,
            "quants": quants,
            "p_value": p_value}






#Заголовок страницы
st.image("logo-3.png")
st.header("Анализ данных")
st.markdown("---")
st.subheader("Загрузите набор данных  в формате csv ниже:")
file = st.file_uploader('', type="csv", accept_multiple_files=False)





#Визуализация данных
figure = plt.figure()
if file is not None:
    df = pd.read_csv(file)
    df = df.drop(df.columns[0], axis = 1)

    st.subheader("Ваш набор данных: ")
    st.dataframe(df)
    st.markdown("---")

    st.subheader("Выберите переменные для визуализации")
    viz_options = st.multiselect("(2 переменные)", [col for col in df])
    if (len(viz_options) == 2):
        if (isinstance(df[viz_options[0]][0], str) and (not isinstance(df[viz_options[1]][0], str))):


            fig = plt.figure(figsize=(10, 10))
            labels = df[viz_options[0]].unique()
            sizes = [len(df[df[viz_options[0]] == i]) for i in labels]
            plt.pie(sizes, labels = labels, autopct='%1.1f%%')
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[viz_options[1]], kde=True, bins = 100)
            st.pyplot(fig)

        elif (isinstance(df[viz_options[1]][0], str) and (not isinstance(df[viz_options[0]][0], str))):

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[viz_options[0]], kde=True, bins = 100)
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 10))
            labels = df[viz_options[1]].unique()
            sizes = [len(df[df[viz_options[1]] == i]) for i in labels]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            st.pyplot(fig)

        else:
            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[viz_options[0]], kde=True, bins = 100)
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 5))
            sns.histplot(df[viz_options[1]], kde=True, bins = 100)
            st.pyplot(fig)
    st.markdown("---")






    #Выбор переменных(столбцов) и алгоритмов проверки гипотез
    st.subheader("Выберите переменные для проверки гипотез")
    gip_options = st.multiselect("(2 переменные). Переменные должны быть категориальными.",
                                 [col for col in df if isinstance(df[col][0], str)])
    st.subheader("Выберите алгоритмы проверки гипотез")
    gip_tests = st.multiselect('2 алгоритма', ['A/B', 't-test', 'mann-whitney U-test', 'bootstraping'])
    st.markdown("---")




    #Проверка гипотез для первой переменной(столбца)
    if len(gip_options) == 2 and len(gip_tests) == 2:
        st.subheader(f"Выберите 2 значения переменной {gip_options[0]}")
        gip_slctd_opt_1_values = st.multiselect('', df[gip_options[0]].unique())
        if len(gip_slctd_opt_1_values) == 2:
            st.subheader(f"Выберите значение, по которому будут сравниваться {gip_slctd_opt_1_values[0]} и {gip_slctd_opt_1_values[1]}")
            value_1 = st.selectbox('', [col for col in df if isinstance(df[col][0], int) or isinstance(df[col][0], float)], key = '1')
            data_1_1, data_1_2 = df[df[gip_options[0]]==gip_slctd_opt_1_values[0]][value_1], df[df[gip_options[0]]==gip_slctd_opt_1_values[1]][value_1]

            match gip_tests[0]:
                case 'A/B':
                    st.subheader(f'Результат A/B: {data_1_2.mean()-data_1_1.mean()}')
                case 't-test':
                    res = stats.ttest_ind(data_1_1,
                                          data_1_2,
                                          equal_var=False)
                    st.subheader(f'Результат t-test (p-value): {res.pvalue}')
                case 'mann-whitney U-test':
                    res = stats.mannwhitneyu(data_1_1, data_1_2)
                    st.subheader(f'Результат mann-whitney U-test: {res}')
                case 'bootstraping':
                    res = get_bootstrap(data_1_1, data_1_2)
                    st.subheader(f"Результат bootstraping(p-value): {res['p_value']}")

            match gip_tests[1]:
                case 'A/B':
                    st.subheader(f'Результат A/B: {data_1_2.mean() - data_1_1.mean()}')
                case 't-test':
                    res = stats.ttest_ind(data_1_1,
                                          data_1_2,
                                          equal_var=False)
                    st.subheader(f'Результат t-test (p-value): {res.pvalue}')
                case 'mann-whitney U-test':
                    res = stats.mannwhitneyu(data_1_1, data_1_2)
                    st.subheader(f'Результат mann-whitney U-test: {res}')
                case 'bootstraping':
                    res = get_bootstrap(data_1_1, data_1_2)
                    st.subheader(f"Результат bootstraping(p-value): {res['p_value']}")
            st.markdown("---")





        #Проверка гипотез для второй переменной(столбца)
        st.subheader(f"Выберите 2 значения переменной {gip_options[1]}")
        gip_slctd_opt_2_values = st.multiselect('', df[gip_options[1]].unique())
        if len(gip_slctd_opt_2_values) == 2:
            st.subheader(
                f"Выберите значение, по которому будут сравниваться {gip_slctd_opt_2_values[0]} и {gip_slctd_opt_2_values[1]}")
            value_2 = st.selectbox('', [col for col in df if isinstance(df[col][0], int) or isinstance(df[col][0], float)], key = '2')
            data_2_1, data_2_2 = df[df[gip_options[1]] == gip_slctd_opt_2_values[0]][value_2], \
            df[df[gip_options[1]] == gip_slctd_opt_2_values[1]][value_2]

            match gip_tests[0]:
                case 'A/B':
                    st.subheader(f'Результат A/B: {data_2_2.mean() - data_2_1.mean()}')
                case 't-test':
                    res = stats.ttest_ind(data_2_1,
                                          data_2_2,
                                          equal_var=False)
                    st.subheader(f'Результат t-test (p-value): {res.pvalue}')
                case 'mann-whitney U-test':
                    res = stats.mannwhitneyu(data_2_1, data_2_2)
                    st.subheader(f'Результат mann-whitney U-test: {res}')
                case 'bootstraping':
                    res = get_bootstrap(data_2_1, data_2_2)
                    st.subheader(f"Результат bootstraping(p-value): {res['p_value']}")

            match gip_tests[1]:
                case 'A/B':
                    st.subheader(f'Результат A/B: {data_2_2.mean() - data_2_1.mean()}')
                case 't-test':
                    res = stats.ttest_ind(data_2_1,
                                          data_2_2,
                                          equal_var=False)
                    st.subheader(f'Результат t-test (p-value): {res.pvalue}')
                case 'mann-whitney U-test':
                    res = stats.mannwhitneyu(data_2_1, data_2_2)
                    st.subheader(f'Результат mann-whitney U-test: {res}')
                case 'bootstraping':
                    res = get_bootstrap(data_2_1, data_2_2)
                    st.subheader(f"Результат bootstraping(p-value): {res['p_value']}")
            st.markdown("---")






