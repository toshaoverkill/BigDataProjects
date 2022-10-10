from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import umap
from sklearn.datasets import load_digits
from datetime import datetime


digits = load_digits()
X, y = load_digits(return_X_y=True)
fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')
n = 0
plt.gray()
for i in range(0, 2):
    for j in range(0, 5):
        axs[i, j].matshow(digits.images[n])
        axs[i, j].set(title=y[n])
        n = n + 1
plt.show()


def t_sne_f():
    perplexity = [5, 25, 50]
    for i in range(len(perplexity)):
        P = perplexity[i]
        T = TSNE(n_components=2, perplexity=P, random_state=123)
        X_embedded = T.fit_transform(X)
        fig = plt.figure()
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y.astype(str), data=X_embedded,
                        palette='bright').set(
            xlabel="X", ylabel="Y", title=P)
        plt.show()
    return datetime.now()


def umap_f():
    D = pd.DataFrame(digits.data, columns=digits.feature_names)
    scaler = preprocessing.MinMaxScaler()
    D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
    n_n = (5, 25, 50)
    m_d = (0.1, 0.6)
    um = dict()
    DATA = D.copy()
    for i in range(len(n_n)):
        for j in range(len(m_d)):
            um[(n_n[i], m_d[j])] = (
                umap.UMAP(n_neighbors=n_n[i], min_dist=m_d[j], random_state=123).fit_transform(DATA))

    for index, value in enumerate(um.values()):
        DATA['x'] = value[:, 0]
        DATA['y'] = value[:, 1]
        fig = plt.figure()
        sns.scatterplot(x='x', y='y', hue=y.astype(str), data=DATA, palette='bright').set(
            title=list(um.keys())[index])
        plt.show()
    return datetime.now()

start_time = datetime.now()
print("Время работы t-sne: ", t_sne_f() - start_time, ', а UMAP: ', umap_f() - start_time)
