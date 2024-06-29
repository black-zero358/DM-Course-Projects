#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示的字体，例如宋体、黑体等
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 加载并预处理数据
def load_and_preprocess_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    # 创建一个DataFrame以便更好地查看数据
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data

# 数据探索
def explore_data(df, data):
    # 显示数据的前5行
    print(df.head())
    # 打印数据集的关键词信息
    print("数据关键词:", data.keys())
    # 打印特征名称
    print("特征名称:", list(data.feature_names))
    # 打印数据集的描述统计信息
    print(df.describe())
    # 打印目标分类的名称
    print(data.target_names)
    # 打印目标分类的数量分布
    print(pd.Series(data.target).value_counts())
    # 检查数据集中是否有缺失值
    print(df.isnull().sum())
    return df, data

# 数据可视化
def visualize_data(df):
    # 绘制直方图
    df.hist(bins=10, figsize=(20, 15))
    plt.show()
    # 绘制箱线图
    plt.figure(figsize=(16, 9))
    sns.boxplot(data=df, orient='h')
    plt.show()
    # 绘制小提琴图
    plt.figure(figsize=(16, 9))
    sns.violinplot(data=df, orient='h')
    plt.show()
    # 计算特征之间的相关系数并绘制热力图
    corr_df = df.corr().abs()
    sns.heatmap(corr_df, annot=False, cmap='coolwarm')
    plt.show()

# 模型评估函数
def evaluate_model(model, X, y, name=""):
    # 预测数据
    pred = model.predict(X)
    # 打印模型评估指标
    print(f"{name} - 准确率: {accuracy_score(y, pred):.4f}")
    print(f"{name} - 精确率: {precision_score(y, pred, average='macro'):.4f}")
    print(f"{name} - 召回率: {recall_score(y, pred, average='macro'):.4f}")
    print(f"{name} - F1 分数: {f1_score(y, pred, average='macro'):.4f}")

# 学习曲线绘制函数
def plot_learning_curve(model, X, y):
    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10)
    # 绘制学习曲线
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='训练得分')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='交叉验证得分')
    plt.title('学习曲线_' + str(model))
    plt.xlabel('训练集大小')
    plt.ylabel('得分')
    plt.legend()
    plt.show()

# 混淆矩阵绘制函数
def plot_confusion_matrix(model, X, y):
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    disp.ax_.set_title('混淆矩阵_' + str(model))
    plt.show()

# ROC 曲线绘制函数
def plot_roc_curve(models, X_test, y_test):
    ax = plt.gca()
    for model in models:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=str(model))
    plt.title("ROC 曲线")
    plt.show()

# 主函数
def main():
    # 加载和预处理数据
    df, data = load_and_preprocess_data()
    # 数据探索
    explore_data(df, data)
    # 数据可视化
    visualize_data(df)

    # 数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 主成分分析（PCA）
    pca = PCA(n_components=5)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.transform(X_test)

    # 定义不同的模型
    models = [
        (LogisticRegression(random_state=42), "逻辑回归"),
        (SVC(kernel='linear', random_state=42), "支持向量机"),
        (KNeighborsClassifier(n_neighbors=4), "K-近邻算法"),
        (DecisionTreeClassifier(random_state=42), "决策树"),
        (GaussianNB(), "朴素贝叶斯")
    ]

    # 对每个模型进行训练和评估
    for model, name in models:
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)
        plot_learning_curve(model, X_train, y_train)
        plot_confusion_matrix(model, X_test, y_test)

        # 使用PCA数据训练和评估模型
        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_new, y_train)
        evaluate_model(model_pca, X_test_new, y_test, f"{name} (PCA)")

    # 投票分类器（软投票）
    voting = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                          ('svc', SVC(kernel='linear', random_state=42, probability=True)),
                                          ('knn', KNeighborsClassifier(n_neighbors=4))], voting='soft')
    voting.fit(X_train, y_train)
    evaluate_model(voting, X_test, y_test, "投票分类器")
    plot_learning_curve(voting, X_train, y_train)
    plot_confusion_matrix(voting, X_test, y_test)
    plot_roc_curve([voting], X_test, y_test)

    # AdaBoost 分类器（基于逻辑回归）
    ada_lr = AdaBoostClassifier(LogisticRegression(random_state=42), learning_rate=0.55, algorithm="SAMME",
                                n_estimators=49)
    ada_lr.fit(X_train, y_train)
    evaluate_model(ada_lr, X_test, y_test, "AdaBoost (逻辑回归)")
    plot_learning_curve(ada_lr, X_train, y_train)
    plot_confusion_matrix(ada_lr, X_test, y_test)

    # AdaBoost 分类器（基于决策树）
    ada_dt = AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth=3), learning_rate=0.85,
                                n_estimators=150, algorithm="SAMME", random_state=42)
    ada_dt.fit(X_train, y_train)
    evaluate_model(ada_dt, X_test, y_test, "AdaBoost (决策树)")
    plot_learning_curve(ada_dt, X_train, y_train)
    plot_confusion_matrix(ada_dt, X_test, y_test)

    # 特征重要性分析
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    importances = dt.feature_importances_
    index = np.argsort(importances)[::-1]

    # 打印前10个重要特征
    print("特征重要性排名:")
    for i in range(10):
        print(f"({index[i]}) {df.columns[index[i]]} {importances[index[i]]:.6f}")

    # 绘制前10个重要特征的柱状图
    plt.bar(range(10), importances[index][:10])
    plt.xticks(range(10), df.columns[index][:10], rotation=90)
    plt.show()

    # 绘制特征重要性的饼图
    labels, values, explode = ["others"], [0], [0]
    for i in range(30):
        if i < 11:
            labels.append(df.columns[index[i]])
            values.append(importances[index[i]])
            explode.append(0.2 if i < 5 else 0)
        else:
            values[0] += importances[index[i]]

    plt.pie(values, labels=labels, explode=explode, autopct='%1.1f%%')
    plt.show()

# 主程序入口
if __name__ == "__main__":
    main()
