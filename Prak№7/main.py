import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({"Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], "Street": [80, 98, 75, 91, 78],
                   "Garage": [100, 82, 105, 89, 102]})

DATA_bitcoin = pd.read_csv("bitcoin.csv")
DATA_housePrice = pd.read_csv("housePrice.csv")

projection = 14


def correlation(DATA, *args):
    if len(args) == 2:
        two_vectors = np.corrcoef(DATA[args[0]], DATA[args[1]])
        print(f"Два вектора:\n {two_vectors}")
    df = pd.DataFrame(DATA, columns=args)
    corr = df.corr()
    return (f"Матрица корреляции:\n {corr}")


def sc(DATA, x, y):
    plt.grid(True)
    plt.title("Диаграмма рассеяния", fontsize=20)
    plt.xlabel(f"{x} axis")
    plt.ylabel(f"{y} axis")
    plt.scatter(DATA[x], DATA[y], marker="o", color="crimson")
    plt.show()


def hide(DATA=DATA_bitcoin, projection=projection):
    DATA['predict'] = DATA['close'].shift(-projection)
    return DATA


# def normalize(DATA):
#     numerics = ["int64", "float64"]
#     newDATA = DATA.select_dtypes(include=numerics)
#     scaler = preprocessing.MinMaxScaler()
#     scaler.fit(newDATA)
#     scaled_features = scaler.transform(newDATA)
#     df_MinMax = pd.DataFrame(data=scaled_features,
#                              columns=newDATA.columns)

    # return df_MinMax


def slicing(DATA, x_column, y_column, projection=projection):
    x = pd.DataFrame(DATA, columns=[x_column])
    y = pd.DataFrame(DATA, columns=[y_column])
    x = np.array(x, type(float))
    y = np.array(y, type(float))
    x = x[:-projection]
    y = y[:-projection]
    print("x: \n", x, "\ny: \n", y)
    return [x, y]


def line_regres(x, y, df=hide()):
    regression = LinearRegression()
    regression.fit(x, y)
    print(f"Наклон линии регерессии: {regression.coef_}")
    print(f"Y-перехват: {regression.intercept_}")
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x, y, alpha=0.3, color='purple')
    # plt.plot(x, regression.predict(x), color='yellow', linewidth=3)
    # plt.xlabel('predict')
    # plt.ylabel('close')
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.show()
    print(f"Regression score: {regression.score(x, y)}")
    print("Predict: ", regression.predict(df[["close"]][-projection:]))


def check_missing_and_fill(DATA):
    for column in DATA.columns:
        missing = np.mean(DATA[column].isna() * 100)
        if missing > 0:
            DATA[column] = DATA[column].fillna('other')
    DATA['Area'] = pd.to_numeric(DATA['Area'], errors='coerce')
    DATA = DATA.dropna()
    DATA = DATA.reset_index(drop=True)
    return DATA


def line_regres_manually(DATA):
    x = pd.DataFrame(DATA, columns=['Area'])
    y = pd.DataFrame(DATA, columns=['Price(USD)'])
    x = np.array(x, type(float))
    y = np.array(y, type(float))
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    print('b_1 = ', b_1, 'b_0 = ', b_0)
    plt.scatter(x, y, color='pink', marker='o', alpha=0.4)
    y_pred = b_0 + b_1 * x
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.xlabel('area')
    plt.ylabel('price(usd)')
    plt.show()


def main():
    print(correlation(df, "Street", "Garage"))
    sc(df, "Street", "Garage")
    print("DATA_bitcoin \n", DATA_bitcoin)
    print("hide: \n", hide())
    line_regres(slicing(hide(), 'close', 'predict')[0], slicing(hide(), 'close', 'predict')[1])
    print(DATA_housePrice)
    line_regres_manually(check_missing_and_fill(DATA_housePrice))


if __name__ == '__main__':
    main()
