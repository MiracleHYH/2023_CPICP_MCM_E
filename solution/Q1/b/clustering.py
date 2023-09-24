import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset.csv', header=None)

X = data.iloc[:, :-1].values  # 取特征
Y = data.iloc[:, -1].values  # 取标签

# 创建K均值聚类模型
kmeans = KMeans(n_clusters=2)

# 拟合模型并进行聚类
kmeans.fit(X)

# 获取簇中心和簇标签
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

pca = PCA(n_components=2)

# 拟合PCA模型并进行降维
X_2D = pca.fit_transform(X)

# X_2D现在包含了降维后的数据，形状为 (n_samples, 2)
plt.figure(figsize=(8, 6))
for i in range(100):
    color = 'b' if cluster_labels[i] == 1 else 'r'
    marker = 'o' if int(Y[i]) == 1 else 'x'
    plt.scatter(X_2D[i, 0], X_2D[i, 1], marker=marker, c=color, s=30)
plt.title('PCA Dimension Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
