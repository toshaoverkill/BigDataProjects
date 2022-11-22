import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import time

DATA_wine = pd.read_csv('wine.csv')
DATA_class = DATA_wine['quality'].value_counts()


def class_gistogram(DATA):
    fig, ax = plt.subplots()
    ax.bar(DATA.index, DATA.values, edgecolor='black')
    ax.set_title('Гистограмма баланса классов')
    ax.set_ylabel('Количество элементов')
    ax.set_xlabel('Классы')
    plt.show()


def preprocessing_function(DATA):  # Предобработка данных
    print('DATA info():')
    DATA.info()

    def percentage(DATA):  # Процентное содержание пропусков
        print('Процентное содержание пропусков:')
        for column in DATA.columns:
            missing = np.mean(DATA[column].isna() * 100)
            print(f' {column} : {round(missing, 1)}%')

    percentage(DATA)


def razbienie(DATA, class_column):
    predictors = DATA[DATA_wine.columns.difference([class_column])]
    print('Предикторы: \n', predictors)
    target = DATA[class_column]
    print('Таргет: \n', target)
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size=0.8, shuffle=True,
                                                        random_state=271)
    print('Размер для признаков обучающей выборки: ', x_train.shape, '\n',
          'Размер для признаков тестовой выборки: ', x_test.shape, '\n',
          'Размер целевого показателя обучающей выборки: ', y_train.shape, '\n',
          'Размер показателя тестовой выборки: ', y_test.shape)

    def logic_f():
        l = 'Логистическая регрессия'
        start_logic_f_time = time.time()
        model = LogisticRegression(random_state=271)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        print('Предсказанные значения: \n', y_predict)
        print('Исходные значения :\n', np.array(y_test))
        print('Совпадение массивов: \n', np.array_equal(y_predict, np.array(y_test)))
        fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto=True)
        fig.update_layout(title=f'Матрица ошибок.{l}', xaxis_title='Target', yaxis_title='Prediction')
        fig.show()
        print('Логистическая регрессия: \n', classification_report(y_test, y_predict))
        print('Время работы алгоритма: ', time.time() - start_logic_f_time)

    def svm_f():
        l = 'Support Vector Machine(SVM)'
        start_svm_f_time = time.time()
        param_kernel = ('kinear', 'rbf', 'poly', 'sigmoid')
        parametrs = {'kernel': param_kernel}
        model = SVC()
        grid_search_svm = GridSearchCV(estimator=model, param_grid=parametrs, cv=6)
        grid_search_svm.fit(x_train, y_train)
        best_model = grid_search_svm.best_estimator_
        best_model.kernel
        svm_preds = best_model.predict(x_test)
        print('SVM: \n', classification_report(svm_preds, y_test))
        fig = px.imshow(confusion_matrix(y_test, svm_preds), text_auto=True)
        fig.update_layout(title=f'Матрица ошибок.{l}', xaxis_title='Target', yaxis_title='Prediction')
        fig.show()
        print('Время работы алгоритма: ', time.time() - start_svm_f_time)

    def knn_f():
        l = 'Nearest Neighbor (KNN)'
        start_knn_time = time.time()
        number_of_neighbors = np.arange(3, 10, 25)
        model_KNN = KNeighborsClassifier()
        params = {'n_neighbors': number_of_neighbors}
        grid_search = GridSearchCV(estimator=model_KNN, param_grid=params, cv=6)
        grid_search.fit(x_train, y_train)
        print('Лучшее значение macro-average: ', grid_search.best_score_)
        print('Лучшая модель получается при ', grid_search.best_estimator_)
        knn_preds = grid_search.predict(x_test)
        print('KNN: \n', classification_report(knn_preds, y_test))
        fig = px.imshow(confusion_matrix(y_test, knn_preds), text_auto=True)
        fig.update_layout(title=f'Матрица ошибок.{l}', xaxis_title='Target', yaxis_title='Prediction')
        fig.show()
        print('Время работы алгоритма: ', time.time() - start_knn_time)

    logic_f()
    svm_f()
    knn_f()


def main():
    print(f'Датафрейм: \n', DATA_wine)
    preprocessing_function(DATA_wine)
    class_gistogram(DATA_class)
    razbienie(DATA_wine, 'quality')


if __name__ == '__main__':
    main()
