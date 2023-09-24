import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from conf import dataPath
from translate import Translator
import re

plt.rcParams['font.sans-serif'] = ['SimHei']

translator = Translator(to_lang="en", from_lang="zh")

data = pd.read_csv(os.path.join(dataPath, 'table1.csv'))

X = data.iloc[:100, 4:14]
X['性别'] = X['性别'].map({'男': 1, '女': 0})
pressure = data.iloc[:100, 15].str.split('/', expand=True)
X[['血压_高', '血压_低']] = pressure
columns = X.columns
X = X.astype(np.float32)
X_normalized = (X - X.min()) / (X.max() - X.min())
X_normalized.to_csv('./dataset.csv', index=False)

kmeans = KMeans(n_clusters=5)

# 拟合模型并进行聚类
kmeans.fit(X_normalized)

# 获取簇中心和簇标签
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# pca = PCA(n_components=3)
#
# # 拟合PCA模型并进行降维
# X_2D = pca.fit_transform(X)
#
# # X_2D现在包含了降维后的数据，形状为 (n_samples, 2)
# plt.figure(figsize=(8, 6))
# ax = plt.axes(projection='3d')
# for i in range(100):
#     ax.scatter(X_2D[i, 0], X_2D[i, 1], X_2D[i, 2], marker='o', c='b', s=30)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
# plt.title('PCA Dimension Reduction')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()

fig, axes = plt.subplots(4, 3)
for i in range(4):
    for j in range(3):
        idx = i * 3 + j
        grouped_data = {}
        for k in range(5):
            grouped_data[k] = []
        for k in range(100):
            grouped_data[cluster_labels[k]].append(X.iloc[k, idx])
        tags = list(grouped_data.keys())
        values = list(grouped_data.values())
        ax = axes[i, j]
        ax.boxplot(values, labels=tags)
        ax.set_title(columns[idx])
        ax.set_xlabel('类别')
        ax.set_ylabel(columns[idx])
plt.tight_layout()
plt.show()
