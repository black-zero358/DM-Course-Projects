#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    RocCurveDisplay
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

# 加载数据集
data = load_breast_cancer()
# 转换为 DataFrame 格式
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())

# 查看数据关键词
print("数据关键词:", data.keys())
# 查看特征名称
print("特征名称:", list(data.feature_names))

# 查看各个数值型特征的基本统计量
print(df.describe())

# 查看标签分布情况：恶性，良性
print(data.target_names)
print(pd.Series(data.target).value_counts())

# 检查是否存在缺失值
print(df.isnull().sum())

# 绘制特征的直方图
df.hist(bins=10, figsize=(20, 15))
plt.show()

# 绘制箱线图，水平
plt.figure(figsize=(16, 9))
sns.boxplot(data=df, orient='h')
plt.show()

# 绘制小提琴图
plt.figure(figsize=(16, 9))
sns.violinplot(data=df, orient='h')
plt.show()

# 计算各个特征间的相关性
corr_df = df.corr().abs()
corr_values = corr_df.values[np.triu_indices_from(corr_df, k=1)]
print('特征相关性系数最大值：', corr_values.max())
print('特征相关性系数最小值：', corr_values.min())
print('平均特征相关性系数：', corr_values.mean())

# 绘制相关系数矩阵的热力图,颜色越深越相关
sns.heatmap(corr_df, annot=False, cmap='coolwarm')
plt.show()

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 对数据进行归一化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 用PCA模型进行降维
pca = PCA(n_components=5)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

# 建立模型
lr = LogisticRegression(random_state=42)
svc = SVC(kernel='linear', random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
dt = DecisionTreeClassifier(random_state=42)
nb = GaussianNB()

# 训练模型：所有特征数据
lr.fit(X_train, y_train)
svc.fit(X_train, y_train)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
nb.fit(X_train, y_train)

# 训练模型：PCA后的数据
lr_pca = LogisticRegression(random_state=42)
svc_pca = SVC(kernel='linear', random_state=42)
knn_pca = KNeighborsClassifier(n_neighbors=4)
dt_pca = DecisionTreeClassifier(random_state=42)
nb_pca = GaussianNB()
lr_pca.fit(X_train_new, y_train)
svc_pca.fit(X_train_new, y_train)
knn_pca.fit(X_train_new, y_train)
dt_pca.fit(X_train_new, y_train)
nb_pca.fit(X_train_new, y_train)

# 绘制决策树
plt.figure(figsize=(12, 8), dpi=500)
plot_tree(dt, filled=True, rounded=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title('DT')
plt.show()

# 绘制PCA后的决策树
plt.figure(figsize=(12, 8), dpi=500)
plot_tree(dt_pca, filled=True, rounded=True, feature_names=[f'PC{i + 1}' for i in range(pca.n_components_)],
          class_names=data.target_names)
plt.title('DT_PCA')
plt.show()


# 绘制学习曲线
def learningCurve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
    plt.title('Learning Curve_' + str(model))
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


# 绘制混淆矩阵
def confusionMatrix(model, X, y):
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    disp.ax_.set_title('Confusion Matrix_' + str(model))
    plt.show()


# 绘制ROC曲线
def ROC(model, model_pca):
    ax = plt.gca()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=str(model))
    RocCurveDisplay.from_estimator(model_pca, X_test_new, y_test, ax=ax, name=str(model_pca) + "_PCA")
    plt.title("ROC Curve")
    plt.show()


# 评估模型
def evaluate_model(model, X, y, name=""):
    pred = model.predict(X)
    print(f"{name} - Accuracy: {accuracy_score(y, pred):.4f}")
    print(f"{name} - Precision: {precision_score(y, pred, average='macro'):.4f}")
    print(f"{name} - Recall: {recall_score(y, pred, average='macro'):.4f}")
    print(f"{name} - F1 Score: {f1_score(y, pred, average='macro'):.4f}")


# 对所有特征数据的模型进行评估
evaluate_model(lr, X_test, y_test, "Logistic Regression")
evaluate_model(svc, X_test, y_test, "Support Vector Machine")
evaluate_model(knn, X_test, y_test, "K-Nearest Neighbors")
evaluate_model(dt, X_test, y_test, "Decision Tree")
evaluate_model(nb, X_test, y_test, "Naive Bayes")

# 对PCA后的数据的模型进行评估
evaluate_model(lr_pca, X_test_new, y_test, "Logistic Regression (PCA)")
evaluate_model(svc_pca, X_test_new, y_test, "Support Vector Machine (PCA)")
evaluate_model(knn_pca, X_test_new, y_test, "K-Nearest Neighbors (PCA)")
evaluate_model(dt_pca, X_test_new, y_test, "Decision Tree (PCA)")
evaluate_model(nb_pca, X_test_new, y_test, "Naive Bayes (PCA)")

# 绘制混淆矩阵和ROC曲线
learningCurve(nb, X_train, y_train)
confusionMatrix(nb, X_test, y_test)
ROC(nb, nb_pca)

# Voting Classifier
clf1 = LogisticRegression(random_state=42)
clf2 = SVC(kernel='linear', random_state=42, probability=True)
clf3 = KNeighborsClassifier(n_neighbors=4)
voting = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('knn', clf3)], voting='soft')
voting.fit(X_train, y_train)

learningCurve(voting, X_train, y_train)
confusionMatrix(voting, X_test, y_test)

voting_pred = voting.predict(X_test)
print(f"Voting Classifier - Accuracy: {accuracy_score(y_test, voting_pred):.4f}")
print(f"Voting Classifier - Precision: {precision_score(y_test, voting_pred, average='macro'):.4f}")
print(f"Voting Classifier - Recall: {recall_score(y_test, voting_pred, average='macro'):.4f}")
print(f"Voting Classifier - F1 Score: {f1_score(y_test, voting_pred, average='macro'):.4f}")


def ROC_voting():
    ax = plt.gca()
    RocCurveDisplay.from_estimator(voting, X_test, y_test, ax=ax)
    plt.title("ROC Curve - Voting Classifier")
    plt.show()


ROC_voting()

# AdaBoost with Logistic Regression
lr_ada = LogisticRegression(random_state=42)
ada_lr = AdaBoostClassifier(learning_rate=0.55, algorithm="SAMME", n_estimators=49)
ada_lr.fit(X_train, y_train)

# AdaBoost with Decision Tree
dt_ada = DecisionTreeClassifier(random_state=42, max_depth=3)
ada_dt = AdaBoostClassifier(learning_rate=0.85, n_estimators=150, algorithm="SAMME",
                            random_state=42)
ada_dt.fit(X_train, y_train)

learningCurve(ada_lr, X_train, y_train)
confusionMatrix(ada_lr, X_test, y_test)

ada_lr_pred = ada_lr.predict(X_test)
print(f"AdaBoostClassifier (LogisticRegression) - Accuracy: {accuracy_score(y_test, ada_lr_pred):.4f}")
print(
    f"AdaBoostClassifier (LogisticRegression) - Precision: {precision_score(y_test, ada_lr_pred, average='macro'):.4f}")
print(f"AdaBoostClassifier (LogisticRegression) - Recall: {recall_score(y_test, ada_lr_pred, average='macro'):.4f}")
print(f"AdaBoostClassifier (LogisticRegression) - F1 Score: {f1_score(y_test, ada_lr_pred, average='macro'):.4f}")

# AdaBoost with Decision Tree
dt_ada = DecisionTreeClassifier(random_state=42, max_depth=3)
# ada_dt = AdaBoostClassifier(base_estimator=dt_ada, learning_rate=0.85, n_estimators=150, random_state=42)
ada_dt = AdaBoostClassifier(learning_rate=0.85, n_estimators=150, random_state=42)
ada_dt.fit(X_train, y_train)

learningCurve(ada_dt, X_train, y_train)
confusionMatrix(ada_dt, X_test, y_test)

ada_dt_pred = ada_dt.predict(X_test)
print(f"AdaBoostClassifier (DecisionTree) - Accuracy: {accuracy_score(y_test, ada_dt_pred):.4f}")
print(f"AdaBoostClassifier (DecisionTree) - Precision: {precision_score(y_test, ada_dt_pred, average='macro'):.4f}")
print(f"AdaBoostClassifier (DecisionTree) - Recall: {recall_score(y_test, ada_dt_pred, average='macro'):.4f}")
print(f"AdaBoostClassifier (DecisionTree) - F1 Score: {f1_score(y_test, ada_dt_pred, average='macro'):.4f}")

# 决策树特征重要性
dt_1 = DecisionTreeClassifier(random_state=42)
dt_1.fit(X_train, y_train)
importances = dt_1.feature_importances_
index = np.argsort(importances)[::-1]

print("Feature Ranking:")
for i in range(10):
    print(f"({index[i]}) {df.columns[index[i]]} {importances[index[i]]:.6f}")

plt.bar(range(10), importances[index][:10])
plt.xticks(range(10), df.columns[index][:10], rotation=90)
plt.show()

# 饼图显示特征重要性
labels = ["others"]
values = [0]
explode = [0]
for i in range(30):
    if i < 11:
        labels.append(df.columns[index[i]])
        values.append(importances[index[i]])
        if i < 5:
            explode.append(0.2)
        else:
            explode.append(0)
    else:
        values[0] += importances[index[i]]

plt.pie(values, labels=labels, explode=explode, autopct='%1.1f%%')
plt.show()
