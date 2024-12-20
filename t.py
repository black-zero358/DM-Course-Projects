import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# 1. 加载数据集
iris = datasets.load_iris()
X = iris.data  # 使用全部四个特征进行聚类

# 2. 设置聚类数量
n_clusters = 3

# 3. 创建KMeans模型，并进行拟合和预测
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 4. 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)
print("The average silhouette_score is :", silhouette_avg)

# 5. 可视化聚类结果
# 创建一个 subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# 第一个 subplot 是轮廓图
# 轮廓系数范围在 -1, 1 之间，但这里将在 -0.1 和 1 之间显示
ax1.set_xlim([-0.1, 1])
# (n_clusters+1)*10 是为了在每个聚类之间插入一点空白空间
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# 计算每个样本的轮廓系数
sample_silhouette_values = silhouette_samples(X, labels)

y_lower = 10
for i in range(n_clusters):
    # 聚集该聚类的轮廓分数，并排序
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # 在每个聚类层之间标记聚类的编号
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10  # 为下一个聚类计算新的 y_lower

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# 红线为轮廓系数的平均值
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # 清除 y 轴的刻度
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 第二个 subplot 展示了实际的聚类
colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=300, lw=0, alpha=0.7, c=colors, edgecolor='k')

# 标记聚类中心
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on Iris dataset with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

plt.show()
