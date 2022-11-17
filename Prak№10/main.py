import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import time
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import preprocessing

DATA_beers = pd.read_csv('beers.csv')
print(DATA_beers)
print((DATA_beers['style'].unique()).tolist())


def dropcolumns():
    DATA_beers.drop(columns=[DATA_beers.keys()[0], DATA_beers.keys()[3], DATA_beers.keys()[4], DATA_beers.keys()[6]],
                    axis=1, inplace=True)
    return (DATA_beers)


def replacevalue():
    d = {}
    keys = range(len(((DATA_beers['style'].unique()).tolist())))
    values = ((DATA_beers['style'].unique()).tolist())
    for i in keys:
        d[values[i]] = keys[i]

    def repl():
        DATA_beers['style'].replace(d, inplace=True)
        return (DATA_beers)
    return repl()


def preprocessing_function(DATA=DATA_beers):  # Предобработка данных
    print('DATA info():')
    DATA.info()
    DATA_beers['style'] = DATA_beers['style'].astype(float)
    def median_fill(column):
        median_value = DATA[column].median()
        DATA[column].fillna(median_value, inplace=True)

    def percentage(DATA):  # Процентное содержание пропусков
        print('Процентное содержание пропусков до заполнения:')
        for column in DATA.columns:
            missing = np.mean(DATA[column].isna() * 100)
            print(f' {column} : {round(missing, 1)}%')
            if missing > 0:
                median_fill(column)
        print('Процентное содержание пропусков после заполнения:')
        for column in DATA.columns:
            missing = np.mean(DATA[column].isna() * 100)
            print(f' {column} : {round(missing, 1)}%')

    percentage(DATA)


def kmeans():
    start_kmeans = time.time()
    models = []
    score1 = []
    score2 = []
    for i in range(2, 10):
        model = KMeans(n_clusters=i, random_state=123, init='k-means++').fit(DATA_beers)
        models.append(model)
        score1.append(model.inertia_)
        score2.append(silhouette_score(DATA_beers, model.labels_))

    def showscore(score):
        plt.grid()
        plt.plot(np.arange(2, 10), score, marker='o')
        plt.show()

    showscore(score1)
    showscore(score2)

    model1 = KMeans(n_clusters=3, random_state=123, init='k-means++')
    model1.fit(DATA_beers)
    print(model1.cluster_centers_)
    labels = model1.labels_
    DATA_beers['Claster'] = labels
    print(DATA_beers['Claster'].value_counts())
    fig = go.Figure(
        data=[go.Scatter3d(x=DATA_beers['abv'], y=DATA_beers['ibu'], z=DATA_beers['style'], mode='markers',
                           marker_color=DATA_beers['Claster'], marker_size=4)])
    fig.show()
    print('Время работы k-means: ', time.time() - start_kmeans)


def hierarchical_agglomerative_clustering():
    start_hie = time.time()
    model2 = AgglomerativeClustering(3, compute_distances=True)
    clustering = model2.fit(DATA_beers)
    DATA_beers['Cluster'] = clustering.labels_

    fig = go.Figure(data=[go.Scatter3d(x=DATA_beers['abv'], y=DATA_beers['ibu'], z=DATA_beers['style'], mode='markers',
                                       marker_color=DATA_beers['Claster'], marker_size=4)])
    fig.show()
    print('Время работы hierarchical_agglomerative_clustering: ', time.time() - start_hie)


def dbscan():
    start_dbscan = time.time()
    model3 = DBSCAN(eps=11, min_samples=2).fit(DATA_beers)
    DATA_beers['Claster'] = model3.labels_
    fig = go.Figure(data=[go.Scatter3d(x=DATA_beers['abv'], y=DATA_beers['ibu'], z=DATA_beers['style'], mode='markers',
                                       marker_color=DATA_beers['Claster'], marker_size=4)])
    fig.show()
    print('Время работы DBSCAN: ', time.time() - start_dbscan)


def t_sne_f():
    scaler = preprocessing.MinMaxScaler()
    D = pd.DataFrame(scaler.fit_transform(DATA_beers), columns=DATA_beers.columns)

    T = TSNE(n_components=2, perplexity=50, random_state=123)
    TSNE_features = T.fit_transform(D)
    TSNE_features[1:4, :]
    DATA = D.copy()
    DATA['x'] = TSNE_features[:, 0]
    DATA['y'] = TSNE_features[:, 1]
    fig = plt.figure()
    sns.scatterplot(x='x', y='y', hue=DATA_beers['style'], data=DATA, palette='bright')
    plt.show()


def visualize():
    x = DATA_beers['abv']
    y = DATA_beers['ibu']
    z = DATA_beers['style']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def main():
    print('Датафрейм с удаленными столбцами: \n', dropcolumns())
    print(replacevalue())
    preprocessing_function()
    kmeans()
    hierarchical_agglomerative_clustering()
    dbscan()
    # t_sne_f()
    visualize()


if __name__ == '__main__':
    main()
