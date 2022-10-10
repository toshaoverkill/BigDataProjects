import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as sts
import statistics
import seaborn as sns

HEADERS = 'insurance.csv'
EXITHEADER = 'Введите номер задания или введите \'0\' для выхода:'


def ex1(HEADERS=HEADERS):
    DATA = pd.read_csv(HEADERS)
    return DATA


def ex2():
    print('describe:\n', ex1().describe())


def ex3():
    print('Выбор варианта вывода графика (1 или 2):')
    a = int(input())
    match a:
        case 1:
            ex1().hist(color='green', edgecolor='black', bins=10)
            plt.show()
        case 2:
            fig, ax = plt.subplots(1, 4, figsize=(15, 4))
            ax[0].hist(ex1()['age'], label='age', edgecolor='black', color='green', bins=15)
            ax[0].legend()
            ax[1].hist(ex1()['bmi'], label='bmi', edgecolor='black', color='orange', bins=15)
            ax[1].legend()
            ax[2].hist(ex1()['children'], label='children', edgecolor='black', color='pink', bins=15)
            ax[2].legend()
            ax[3].hist(ex1()['charges'], label='charges', edgecolor='black', color='blue', bins=15)
            ax[3].legend()
            plt.show()


def ex4():
    mean_bmi = np.mean(ex1()['bmi'])
    med_bmi = np.median(ex1()['bmi'])
    moda_bmi = statistics.mode(ex1()['bmi'])
    data1 = [mean_bmi, med_bmi, moda_bmi]
    print('Медиана для bmi: ', med_bmi, ',мода для bmi: ', moda_bmi, ',среднее для bmi: ', mean_bmi)
    mean_ch = np.mean(ex1()['charges'])  # Среднее
    med_ch = np.median(ex1()['charges'])  # Медиана
    moda_ch = statistics.mode(ex1()['charges'])  # Мода
    print('Медиана для charges: ', med_ch, ',мода для charges: ', moda_ch, ',среднее для charges: ', mean_ch)
    data2 = [mean_ch, med_ch, moda_ch]
    std_bmi = ex1()['bmi'].std()
    raz_bmi = ex1()['bmi'].max() - ex1()['bmi'].min()
    q1_bmi = np.percentile(ex1()['bmi'], 25, method='midpoint')
    q3_bmi = np.percentile(ex1()['bmi'], 75, method='midpoint')
    iqr1_bmi = q3_bmi - q1_bmi
    iqr2_bmi = sts.iqr(ex1()['bmi'], interpolation='midpoint')
    print('Стандартное отклонение для bmi: ', std_bmi, ',размах для bmi: ', raz_bmi,
          ',межквартильный размах через numpy для bmi: ', iqr1_bmi, ',межквартильный размах через scipy: ', iqr2_bmi)
    data3 = [std_bmi, raz_bmi, iqr1_bmi, iqr2_bmi]
    std_charges = ex1()['charges'].std()  # Стандартное отклонение
    raz_charges = ex1()['charges'].max() - ex1()['charges'].min()  # Размах
    q1_charges = np.percentile(ex1()['charges'], 25, method='midpoint')
    q3_charges = np.percentile(ex1()['charges'], 75, method='midpoint')
    iqr1_charges = q3_charges - q1_charges  # Межквартильный размах через numpy
    iqr2_charges = sts.iqr(ex1()['charges'], interpolation='midpoint')  # Межквартильный размах через scipy
    data4 = [std_charges, raz_charges, iqr1_charges, iqr2_charges]
    print('Стандартное отклонение для charges: ', std_charges, ',размах для charges: ', raz_charges,
          ',межквартильный размах через numpy для charges: ', iqr1_charges, ',межквартильный размах через scipy: ',
          iqr2_charges)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(ex1()['bmi'], label='bmi', edgecolor='black', color='green', bins=15)
    ax[0].vlines(med_bmi, ymin=0, ymax=200, color='blue', label='Мера центральной тенденции(медиана)')
    ax[0].vlines(moda_bmi, ymin=0, ymax=200, color='orange', label='Мера центральной тенденции(мода)')
    ax[0].vlines(mean_bmi, ymin=0, ymax=200, color='red', label='Мера центральной тенденции(среднее)')
    ax[0].vlines(std_bmi, ymin=0, ymax=200, color='teal', label='Мера центральной тенденции(стандартное отклонение)')
    ax[0].vlines(raz_bmi, ymin=0, ymax=200, color='green', label='Мера центральной тенденции(размах)')
    ax[0].vlines(iqr1_bmi, ymin=0, ymax=200, color='pink',
                 label='Мера центральной тенденции(межквартильный размах через numpy)')
    ax[0].vlines(iqr2_bmi, ymin=0, ymax=200, color='crimson',
                 label='Мера центральной тенденции(межквартильный размах через scipy)')
    ax[0].legend()
    ax[1].hist(ex1()['charges'], label='charges', edgecolor='black', color='green', bins=15)
    ax[1].vlines(med_ch, ymin=0, ymax=400, color='blue', label='Мера центральной тенденции(медиана)')
    ax[1].vlines(moda_ch, ymin=0, ymax=400, color='orange', label='Мера центральной тенденции(мода)')
    ax[1].vlines(mean_ch, ymin=0, ymax=400, color='red', label='Мера центральной тенденции(среднее)')
    ax[1].vlines(std_charges, ymin=0, ymax=400, color='teal',
                 label='Мера центральной тенденции(стандартное отклонение)')
    ax[1].vlines(raz_charges, ymin=0, ymax=400, color='green', label='Мера центральной тенденции(размах)')
    ax[1].vlines(iqr1_charges, ymin=0, ymax=400, color='pink',
                 label='Мера центральной тенденции(межквартильный размах через numpy)')
    ax[1].vlines(iqr2_charges, ymin=0, ymax=400, color='crimson',
                 label='Мера центральной тенденции(межквартильный размах через scipy)')
    ax[1].legend()
    plt.show()


def ex5():
    fig, ax = plt.subplots(1, 4, figsize=(20, 16))
    sns.boxplot(data=ex1()['age'], ax=ax[0])
    ax[0].set_title('age', fontsize=16)
    sns.boxplot(data=ex1()['bmi'], ax=ax[1])
    ax[1].set_title('bmi', fontsize=16)
    sns.boxplot(data=ex1()['charges'], ax=ax[2])
    ax[2].set_title('charges', fontsize=16)
    sns.boxplot(data=ex1()['children'], ax=ax[3])
    ax[3].set_title('children', fontsize=16)
    plt.show()


def ex6():
    print('Введите 1 или 2 для проверки центральной предельной теоремы для bmi и changes соответственно:')
    v = int(input())
    match v:
        case 1:
            print('Введите от 1 до 3 в зависимости от длины выборки:')
            b = int(input())
            bmi_arr = []
            match b:
                case 1:
                    for i in range(300):
                        data_bmi = ex1()['bmi'].sample(n=5)
                        mean = np.mean(data_bmi)
                        bmi_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(bmi_arr)
                    print('Среднее отклонение: ', np.mean(bmi_arr), ' стандартные отклонения: ',
                          np.mean(bmi_arr) + np.std(bmi_arr), ' и ', np.mean(bmi_arr) - np.std(bmi_arr))
                    plt.legend()
                    plt.title('BMI')
                    plt.show()
                case 2:
                    for i in range(300):
                        data_bmi = ex1()['bmi'].sample(n=50)
                        mean = np.mean(data_bmi)
                        bmi_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(bmi_arr)
                    print('Среднее отклонение: ', np.mean(bmi_arr), ' стандартные отклонения: ',
                          np.mean(bmi_arr) + np.std(bmi_arr), ' и ', np.mean(bmi_arr) - np.std(bmi_arr))
                    plt.legend()
                    plt.title('BMI')
                    plt.show()
                case 3:
                    for i in range(300):
                        data_bmi = ex1()['bmi'].sample(n=500)
                        mean = np.mean(data_bmi)
                        bmi_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(bmi_arr)
                    print('Среднее отклонение: ', np.mean(bmi_arr), ' стандартные отклонения: ',
                          np.mean(bmi_arr) + np.std(bmi_arr), ' и ', np.mean(bmi_arr) - np.std(bmi_arr))
                    plt.legend()
                    plt.title('BMI')
                    plt.show()
        case 2:
            print('Введите от 1 до 3 в зависимости от длины выборки:')
            b = int(input())
            ch_arr = []
            match b:
                case 1:
                    for i in range(300):
                        data_ch = ex1()['charges'].sample(n=5)
                        mean = np.mean(data_ch)
                        ch_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(ch_arr)
                    print('Среднее отклонение: ', np.mean(ch_arr), ' стандартные отклонения: ',
                          np.mean(ch_arr) + np.std(ch_arr), ' и ', np.mean(ch_arr) - np.std(ch_arr))
                    plt.legend()
                    plt.title('CHARGES')
                    plt.show()
                case 2:
                    for i in range(300):
                        data_ch = ex1()['charges'].sample(n=50)
                        mean = np.mean(data_ch)
                        ch_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(ch_arr)
                    print('Среднее отклонение: ', np.mean(ch_arr), ' стандартные отклонения: ',
                          np.mean(ch_arr) + np.std(ch_arr), ' и ', np.mean(ch_arr) - np.std(ch_arr))
                    plt.legend()
                    plt.title('CHARGES')
                    plt.show()
                case 3:
                    for i in range(300):
                        data_ch = ex1()['charges'].sample(n=500)
                        mean = np.mean(data_ch)
                        ch_arr.append(mean)
                    fig, ax = plt.subplots()
                    sns.distplot(ch_arr)
                    print('Среднее отклонение: ', np.mean(ch_arr), ' стандартные отклонения: ',
                          np.mean(ch_arr) + np.std(ch_arr), ' и ', np.mean(ch_arr) - np.std(ch_arr))
                    plt.legend()
                    plt.title('CHARGES')
                    plt.show()


def ex7():
    def confidence_interval_95(data):
        mn = data.mean()
        se = data.std() / np.sqrt(data.size)
        return mn - 1.96 * se, mn + 1.96 * se

    def confidence_interval_99(data):
        mn = data.mean()
        se = data.std() / np.sqrt(data.size)
        return mn - 2.58 * se, mn + 2.58 * se

    print('Доверительный интервал 95% для среднего значения индекса массы тела: ', confidence_interval_95(ex1()['bmi']))
    print('Доверительный интервал 95% для среднего значения индекса массы тела: ',
          confidence_interval_95(ex1()['charges']))
    print('Доверительный интервал 99% для среднего значения расходов', confidence_interval_99(ex1()['bmi']))
    print('Доверительный интервал 99% для среднего значения расходов', confidence_interval_99(ex1()['charges']))


print('Введите номер задания:')
example = int(input())
while True:
    match example:
        case 1:
            ex1()
            print(EXITHEADER)
            example = int(input())
        case 2:
            ex2()
            print(EXITHEADER)
            example = int(input())
        case 3:
            ex3()
            print(EXITHEADER)
            example = int(input())
        case 4:
            ex4()
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
