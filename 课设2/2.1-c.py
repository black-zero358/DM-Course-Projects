import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 加载数据的函数
def load_data(url):
    # 定义列名
    column_names = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
    ]
    # 读取CSV文件，处理缺失值
    df = pd.read_csv(url, header=None, na_values='?')
    # 设置列名
    df.columns = column_names
    return df


# 执行探索性数据分析（EDA）的函数
def perform_eda(df):
    # 显示数据框前5行
    display(df.head())
    # 打印每个特征的缺失值数量
    print("\n每个特征的缺失值数量:\n", df.isnull().sum())
    # 删除缺失值
    df.dropna(inplace=True)
    # 显示描述性统计
    display(df.describe())
    # 选择数值型特征
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 计算与价格的相关性，并按相关性排序
    corr = df[numerical_features].corr(method='pearson')['price'].sort_values(ascending=False)
    print("\n特征与价格的相关性:\n", corr)
    # 绘制相关矩阵热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='RdYlBu')
    plt.title('相关矩阵')
    plt.show()


# 数据预处理函数
def preprocess_data(df):
    # 选择分类特征
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    # 将分类特征因子化
    for feature in categorical_features:
        df[feature] = pd.factorize(df[feature])[0]
    # 标准化数值特征
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


# 执行聚类和可视化的函数
def perform_clustering(df):
    # 选择进行聚类的特征
    selected_features = ['horsepower', 'curb-weight', 'engine-size', 'price']
    X = df[selected_features].values
    inertia = []
    # 使用肘部法则确定最优簇数
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    # 绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('簇的数量')
    plt.ylabel('惯量')
    plt.title('肘部法则确定最佳簇数')
    plt.show()
    # 选择最优簇数（根据肘部法则）
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_
    # 绘制聚类结果
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'black']
    for i in range(optimal_k):
        plt.scatter(X[df['cluster'] == i, 0], X[df['cluster'] == i, 1], s=50, c=colors[i], label=f'簇 {i + 1}')
    # 绘制簇中心
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=800, marker='*', c='yellow', label='簇中心', edgecolors='red',
                linewidths=2)
    plt.title('K-Means 聚类结果')
    plt.xlabel('马力')
    plt.ylabel('整备重量')
    plt.legend()
    plt.show()
    # 打印聚类性能指标
    print(f"误差平方和 (SSE): {kmeans.inertia_}")
    print(f"轮廓系数: {silhouette_score(X, kmeans.labels_):.3f}")
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    visualizer.show()


# 主函数执行
def main():
    # 设置Matplotlib的字体和负号显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    # 加载数据
    df = load_data(url)
    # 执行探索性数据分析
    perform_eda(df)
    # 数据预处理
    df = preprocess_data(df)
    # 绘制选择特征的成对关系图
    sns.pairplot(df[['horsepower', 'curb-weight', 'engine-size', 'price']])
    plt.suptitle('选择特征的成对关系图', y=1.02)
    plt.show()
    # 执行聚类分析
    perform_clustering(df)


# 检查是否作为脚本执行
if __name__ == "__main__":
    main()
