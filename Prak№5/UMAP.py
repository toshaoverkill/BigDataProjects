import umap
# from sklearn.manifold import TSNE
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

data = pd.read_csv('zoo.csv')
D = data.drop(['class_type', 'animal_name'], axis=1)

scaler = preprocessing.MinMaxScaler()
D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
n_n = (5, 25, 50)
m_d = (0.1, 0.6)
um = dict()
DATA = D.copy()
for i in range(len(n_n)):
    for j in range(len(m_d)):
        um[(n_n[i], m_d[j])] = (umap.UMAP(n_neighbors=n_n[i], min_dist=m_d[j], random_state=123).fit_transform(DATA))

for index, value in enumerate(um.values()):
    DATA['x'] = value[:, 0]
    DATA['y'] = value[:, 1]
    fig = plt.figure()
    sns.scatterplot(x='x', y='y', hue=data['class_type'], data=DATA, palette='bright').set(
        title=list(um.keys())[index])
    plt.show()

