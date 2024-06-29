import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示的字体，例如宋体、黑体等
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 1. 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None, na_values='?')

# 添加列名
column_name = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
               'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
               'wheel-base', 'length', 'width', 'height', 'curb-weight',
               'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
               'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
df.columns = column_name

# 2. 探索性数据分析 (EDA)
display(df.head())  # 显示前几行数据

# 检查缺失值
print("\n每个特征缺失的数量:\n", df.isnull().sum())

# 删除缺失值所在的行
df.dropna(inplace=True)

# 显示统计信息
display(df.describe())

# 计算每个特征和价格之间的相关系数，只包含数值特征
numerical_features = ['symboling', 'normalized-losses', 'wheel-base', 'length',
                      'width', 'height', 'curb-weight', 'bore', 'stroke',
                      'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
                      'highway-mpg', 'price']
corr = df[numerical_features].corr(method='pearson')['price']
print("\n每个特征与价格的相关系数:\n", corr.sort_values(ascending=False))

# 显示相关性矩阵的热力图
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='RdYlBu')
plt.title('相关性矩阵')
plt.show()

# 3. 数据预处理: 编码分类特征和标准化数值特征
categorical_features = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
                        'body-style', 'drive-wheels', 'engine-location',
                        'engine-type', 'num-of-cylinders', 'fuel-system']
for feature in categorical_features:
    df[feature] = pd.factorize(df[feature])[0]

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 4. 探索性数据可视化
sns.pairplot(df[['horsepower', 'curb-weight', 'engine-size', 'price']])
plt.suptitle('数值特征成对关系图', y=1.02)
plt.show()

# 5. K-Means 聚类分析
selected_features = ['horsepower', 'curb-weight', 'engine-size', 'price']
X = df[selected_features].values

# 通过肘部法则确定最佳聚类数
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('聚类数')
plt.ylabel('惯性')
plt.title('肘部法则确定最佳聚类数')
plt.show()

# 根据肘部法则选择聚类数 k=3
k = 3  # 确定最佳聚类数
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# 可视化聚类结果
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'black']
for i in range(k):
    plt.scatter(X[df['cluster'] == i, 0], X[df['cluster'] == i, 1], s=50, c=colors[i], label=f'聚类 {i+1}')

# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=800, marker='*', c='yellow', label='聚类中心', edgecolors='red', linewidths=2)

plt.title('K-Means 聚类结果')
plt.xlabel('马力')
plt.ylabel('整备质量')
plt.legend()
plt.show()

# 6. 模型评估
print(f"SSE (误差平方和): {kmeans.inertia_}")
print(f"轮廓系数: {silhouette_score(X, kmeans.labels_):.3f}")

# 可视化轮廓系数
plt.figure(figsize=(10, 6))
visualizer = SilhouetteVisualizer(kmeans)
visualizer.fit(X)
visualizer.show()
