import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DATA_insurance = pd.read_csv('insurance.csv')  # Выгрузка данных
print(f'Датафрейм: \n{DATA_insurance}')


# def check_duplicateRows(DATA):
#     Dup_Rows = DATA[DATA.duplicated()]
#     print("\n\nПовторяющиеся строки : \n {}".format(Dup_Rows))
#     DATA.drop_duplicates(keep=False)
#     return DATA


def preprocessing_function(DATA):  # Предобработка данных
    print('DATA info():')
    DATA_insurance.info()

    def percentage(DATA):  # Процентное содержание пропусков
        print('Процентное содержание пропусков:')
        for column in DATA.columns:
            missing = np.mean(DATA[column].isna() * 100)
            print(f' {column} : {round(missing, 1)}%')

    def unique_column(DATA, column):
        print(f'Уникальные значения {column}:\n {DATA[column].unique()}')

    percentage(DATA_insurance)
    unique_column(DATA_insurance, 'region')


def anova_test(DATA, a, b):
    explore_df = DATA[[a, b]]
    print(f'Исследуемые параметры: \n{explore_df}')
    group = explore_df.groupby(a).groups
    southwest = explore_df[b][group['southwest']]
    southeast = explore_df[b][group['southeast']]
    northwest = explore_df[b][group['northwest']]
    northeast = explore_df[b][group['northeast']]
    print(f'Результаты ANOVA теста:\n {sts.f_oneway(northwest, southeast, northeast, southwest)}')

    region = ['southwest', 'southeast', 'northwest', 'northeast']
    region_pairs = []
    for region1 in range(3):
        for region2 in range(region1 + 1, 4):
            region_pairs.append((region[region1], region[region2]))
    print(region_pairs)

    # t-test
    def hypothesis(a, l=len(region_pairs)):
        q = a * l
        if q > 0.05:
            return 'Гипотеза принимается'
        else:
            return 'Гипотеза отклоняется'

    for region1, region2 in region_pairs:
        test_p = sts.ttest_ind(explore_df[b][group[region1]], explore_df[b][group[region2]])[1]
        print(f'Пара: {region1} и {region2},p-value: {test_p} соответственно {hypothesis(test_p)}')

    # post-hoc
    tukey = pairwise_tukeyhsd(endog=explore_df[b].to_numpy(), groups=explore_df[a].to_numpy(), alpha=0.05)
    tukey.plot_simultaneous()
    print(tukey)
    plt.vlines(x=31.25, ymin=-0.5, ymax=4.5, color='red')
    tukey.summary()
    plt.show()


def anova_test_2(DATA, a, b):
    explore_df = DATA[[a, b]]
    model = ols('bmi ~ region', data=explore_df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    print(f'Результаты ANOVA теста через библиотеку statsmodel: \n{anova_result}')


def two_factor_anova(DATA, a, b, c):
    explore_df = DATA[[a, b, c]]
    print(f'Исследуемые параметры: \n{explore_df}')
    model = ols('bmi ~ C(region)+C(sex)+C(region):C(sex)', data=explore_df).fit()
    print(sm.stats.anova_lm(model, typ=2))
    explore_df['combination'] = explore_df[a] + ' / ' + explore_df[c]
    tukey = pairwise_tukeyhsd(endog=explore_df[b], groups=explore_df['combination'], alpha=0.05)
    tukey.plot_simultaneous()
    tukey.summary()
    plt.show()
    print(tukey)


def main():
    preprocessing_function(DATA_insurance)
    anova_test(DATA_insurance, 'region', 'bmi')
    anova_test_2(DATA_insurance, 'region', 'bmi')
    two_factor_anova(DATA_insurance, 'region', 'bmi', 'sex')


if __name__ == '__main__':
    main()
