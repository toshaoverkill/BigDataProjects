from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

data = pd.read_csv('zoo.csv')
D = data.drop(['class_type', 'animal_name'], axis=1)
scaler = preprocessing.MinMaxScaler()
D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)

T = TSNE(n_components=2, perplexity=50, random_state=123)
TSNE_features = T.fit_transform(D)
TSNE_features[1:4, :]
DATA = D.copy()
DATA['x'] = TSNE_features[:, 0]
DATA['y'] = TSNE_features[:, 1]
fig = plt.figure()
sns.scatterplot(x='x', y='y', hue=data['class_type'], data=DATA, palette='bright')
plt.show()
