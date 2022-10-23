import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sts

HEADERS = 'ECDCCases.csv'
EXITHEADER = 'Введите номер задания или введите \'0\' для выхода:'


def ex1(HEADERS=HEADERS):
    DATA = pd.read_csv(HEADERS)
    return DATA


def ex2(v):
    def heatmap_f(data, a):
        colors = ['green', 'red']
        sns.heatmap(data.isna(), cmap=sns.color_palette(colors)).set(title=f'Тепловая карта пропущенных значений {a}')
        plt.show()

    def percentage(data):
        for column in data.columns:
            missing = np.mean(data[column].isna() * 100)
            print(f'{column} : {round(missing, 1)}%')

    def delete_bads(DataFrame):
        d = dict()
        for column in DataFrame.columns:
            missing = DataFrame[column].isna().sum()
            without_missing = len(DataFrame[column]) - missing
            d[column] = missing
        sorted_dict = sorted([(value, key) for (key, value) in d.items()])
        n = 2
        sort = dict((sorted_dict[:-n - 1:-1]))

        # plt.bar(sort.values(), sort.keys(), edgecolor='black')
        # plt.xticks(rotation=90)
        # plt.show()

        badlist = (sort.values())
        data = DataFrame.drop(labels=badlist, axis=1)
        return data

    def recode_empty_cells(dataframe):
        for column in dataframe.columns:
            if (dataframe[column]).dtype == 'float64':
                dataframe[column] = dataframe[column].fillna(dataframe[column].median())
            if (dataframe[column]).dtype == 'object':
                dataframe[column] = dataframe[column].fillna('Other')
        return dataframe

    if v == 1:
        heatmap_f(ex1(), '')

        print('Процентное содержание пропусков:')
        percentage(ex1())

        heatmap_f(delete_bads(ex1()), 'после удаления \nстоблцов с наибольшим процентным содержанием пропусков')

        print('Процентное содержание пропусков после удаления стоблцов с наибольшим процентным содержанием пропусков:')
        percentage(delete_bads(ex1()))

        print('Процентное содержание пропусков после заполнения пропусков:')
        percentage(recode_empty_cells(delete_bads(ex1())))
    else:
        return recode_empty_cells(delete_bads(ex1()))


def ex3():
    print(ex2(0).columns)
    print('info:\n', ex2(0).info())
    print('describe:\n', ex2(0).describe())
    list_columns = []
    count = 0
    for column in ex2(0).columns:
        if ex2(0)[column].dtype != 'object':
            list_columns.append(column)
            count += 1

    fig, ax = plt.subplots(1, 2, figsize=(20, 16))
    sns.boxplot(data=ex2(0)[list_columns[0]], ax=ax[0])
    ax[0].set_title(list_columns[0], fontsize=16)
    sns.boxplot(data=ex2(0)[list_columns[1]], ax=ax[1])
    ax[1].set_title(list_columns[1], fontsize=16)
    plt.show()

    for i in range(2, 6):
        boxplot = sns.boxplot(ex2(0)[list_columns[i]])
        boxplot.axes.set_title(list_columns[i], fontsize=16)
        plt.show()

    print((ex2(0)[ex2(0)['deaths'] > 3000])['countryterritoryCode'].unique())
    print(len(ex2(0)[ex2(0)['deaths'] > 3000]))

    print('Дубликаты', ex2(0)[ex2(0).duplicated()])
    ex2(0).drop_duplicates(keep=False)


def ex5():
    DATA = pd.read_csv('bmi.csv')
    DATA_northwest = DATA[DATA['region'] == 'northwest']
    DATA_southwest = DATA[DATA['region'] == 'southwest']

    def Shopiro():
        res1 = sts.shapiro(DATA_northwest['bmi'])
        res2 = sts.shapiro(DATA_southwest['bmi'])
        print('Критерий Шопиро-Уилка\n', res1, '\n', res2)

    def Bartlett():
        res = sts.bartlett(DATA_northwest['bmi'], DATA_southwest['bmi'])
        print('Критерий Бартлетта\n', res)

    def t_Student():
        t_res = sts.ttest_ind(DATA_northwest['bmi'], DATA_southwest['bmi'])
        print('t-критерий Стьюдента\n', t_res)

    Shopiro()
    Bartlett()
    t_Student()


def ex6():
    data = pd.DataFrame([[1, 97, 100],
                         [2, 98, 100],
                         [3, 109, 100],
                         [4, 95, 100],
                         [5, 97, 100],
                         [6, 104, 100]],
                        columns=['Point', 'Observed', 'Expected'])
    print(sts.chisquare(data['Observed'], data['Expected']))


def ex7():
    data = pd.DataFrame({'Женат': [89, 17, 11, 43, 22, 1],
                         'Гражданский брак': [80, 22, 20, 35, 6, 4],
                         'Не состоит в отношениях': [35, 44, 35, 6, 8, 22]})
    data.index = ['Полный рабочий день', 'Частичная занятость', 'Временно не работает', 'На домохозяйстве', 'На пенсии',
                  'Учёба']
    print(data)
    print(sts.chi2_contingency(data))


print('Введите номер задания:')
example = int(input())
while True:
    match example:
        case 1:
            ex1()
            print(EXITHEADER)
            example = int(input())
        case 2:
            ex2(1)
            print(EXITHEADER)
            example = int(input())
        case 3:
            ex3()
            print(EXITHEADER)
            example = int(input())
        case 5:
            ex5()
            print(EXITHEADER)
            example = int(input())
        case 6:
            ex6()
            print(EXITHEADER)
            example = int(input())
        case 7:
            ex7()
            print(EXITHEADER)
            example = int(input())
        case 0:
            print("Выход")
            break
