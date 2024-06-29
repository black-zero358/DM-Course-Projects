import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import mode
import numpy as np

# 加载数据的函数
def load_data(url):
    column_names = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
    ]
    df = pd.read_csv(url, header=None, na_values='?')
    df.columns = column_names
    return df

# 执行探索性数据分析（EDA）的函数
def perform_eda(df):
    display(df.head())
    print("\n每个特征的缺失值数量:\n", df.isnull().sum())
    df.dropna(inplace=True)
    display(df.describe())
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    corr = df[numerical_features].corr(method='pearson')['price'].sort_values(ascending=False)
    print("\n特征与价格的相关性:\n", corr)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='RdYlBu')
    plt.title('相关矩阵')
    plt.show()

# 数据预处理函数
def preprocess_data(df):
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    for feature in categorical_features:
        df[feature] = pd.factorize(df[feature])[0]
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

# 聚类融合和分析结果函数
def clustering_fusion_analysis(df):
    selected_features = ['horsepower', 'curb-weight', 'engine-size', 'price']
    X = df[selected_features].values

    # KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Agglomerative聚类
    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative_labels = agglomerative.fit_predict(X)

    # 聚类融合: 采用简单的多数投票法
    combined_labels = np.vstack((kmeans_labels, agglomerative_labels)).T
    fused_labels = mode(combined_labels, axis=1)[0].flatten()

    # 结果分析
    df['kmeans_cluster'] = kmeans_labels
    df['agglomerative_cluster'] = agglomerative_labels
    df['fused_cluster'] = fused_labels

    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'black']
    for i in range(3):
        plt.scatter(X[df['fused_cluster'] == i, 0], X[df['fused_cluster'] == i, 1], s=50, c=colors[i], label=f'簇 {i + 1}')
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=800, marker='*', c='yellow', label='簇中心', edgecolors='red', linewidths=2)
    plt.title('融合聚类结果')
    plt.xlabel('马力')
    plt.ylabel('整备重量')
    plt.legend()
    plt.show()

    # 计算和打印性能指标
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    agglomerative_silhouette = silhouette_score(X, agglomerative_labels)
    fused_silhouette = silhouette_score(X, fused_labels)

    print(f"KMeans 轮廓系数: {kmeans_silhouette:.3f}")
    print(f"Agglomerative 轮廓系数: {agglomerative_silhouette:.3f}")
    print(f"融合聚类 轮廓系数: {fused_silhouette:.3f}")

    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    visualizer.show()

    # 绘制 Agglomerative 聚类的轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.hist(silhouette_score(X, agglomerative_labels), bins=30, color='blue', edgecolor='black')
    plt.title('Agglomerative 聚类轮廓系数分布')
    plt.xlabel('轮廓系数')
    plt.ylabel('频数')
    plt.show()

    # 绘制 融合聚类的轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.hist(silhouette_score(X, fused_labels), bins=30, color='green', edgecolor='black')
    plt.title('融合聚类轮廓系数分布')
    plt.xlabel('轮廓系数')
    plt.ylabel('频数')
    plt.show()

# 主函数执行
def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    df = load_data(url)
    perform_eda(df)
    df = preprocess_data(df)
    sns.pairplot(df[['horsepower', 'curb-weight', 'engine-size', 'price']])
    plt.suptitle('选择特征的成对关系图', y=1.02)
    plt.show()
    clustering_fusion_analysis(df)

if __name__ == "__main__":
    main()
