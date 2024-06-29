#!/usr/bin/env python
# coding: utf-8
# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix,plot_roc_curve
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

# In[2]:
# 加载数据集
data = load_breast_cancer()
# 转换为 DataFrame 格式
df = pd.DataFrame(data.data, columns=data.feature_names)
# 查看前5条数据


print(df.head())
# In[3]:
# 查看数据关键词
print("数据关键词:", data.keys())
# 查看特征名称
print("特征名称:", list(data.feature_names))
# In[4]:
# 查看各个数值型特征的基本统计量
print(df.describe())
# In[5]:
# 查看标签分布情况：恶性，良性
print(data.target_names)
print(pd.Series(data.target).value_counts())
# In[6]:
# 检查是否存在缺失值
print(df.isnull().sum())
# In[7]:
# 绘制特征的直方图
df.hist(bins=10, figsize=(20, 15))
plt.show()
# In[8]:
# 绘制箱线图，水平
plt.figure(figsize=(16, 9))
sns.boxplot(data=df, orient='h')
plt.show()
# In[9]:


# 绘制小提琴图
plt.figure(figsize=(16, 9))
sns.violinplot(data=df, orient='h')
plt.show()
# In[10]:
# 计算各个特征间的相关性
corr_df = df.corr().abs()
corr_values = corr_df.values[np.triu_indices_from(corr_df, k=1)]
print('特征相关性系数最大值：', corr_values.max())
print('特征相关性系数最小值：', corr_values.min())
print('平均特征相关性系数：', corr_values.mean())
# In[11]:
# 绘制相关系数矩阵的热力图,颜色越深越相关
sns.heatmap(corr_df, annot=False, cmap='coolwarm')
plt.show()
# In[12]:
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2, random_state=42)
# In[13]:
# 对数据进行归一化处理, 最小最大归一
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 训练集用fit_transform
X_test = scaler.transform(X_test)  # 测试集用transform
# In[14]:
# 用PCA模型进行降维
pca = PCA(n_components=5)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

# In[15]:
# 建立模型
lr = LogisticRegression(random_state=42)  # 逻辑回归
svc = SVC(kernel='linear', random_state=42)  # 支持向量机
knn = KNeighborsClassifier(n_neighbors=4)  # KNN
dt = DecisionTreeClassifier(random_state=42)  # 决策树
nb = GaussianNB()  # 朴素贝叶斯（高斯模型）
# 训练模型：所有特征数据
lr.fit(X_train, y_train)
# In[16]:
svc.fit(X_train, y_train)
# In[17]:
knn.fit(X_train, y_train)
# In[18]:
dt.fit(X_train, y_train)
# In[19]:
nb.fit(X_train, y_train)
# In[20]:
lr_pca = LogisticRegression(random_state=42)  # 逻辑回归
svc_pca = SVC(kernel='linear', random_state=42)  # 支持向量机
knn_pca = KNeighborsClassifier(n_neighbors=4)  # KNN
dt_pca = DecisionTreeClassifier(random_state=42)  # 决策树
nb_pca = GaussianNB()  # 朴素贝叶斯（高斯模型）
# 训练模型：所有特征数据
lr_pca.fit(X_train_new, y_train)
svc_pca.fit(X_train_new, y_train)
knn_pca.fit(X_train_new, y_train)
dt_pca.fit(X_train_new, y_train)
nb_pca.fit(X_train_new, y_train)

# In[21]:
plt.figure(figsize=(12, 8), dpi=500)
plot_tree(dt, filled=True, rounded=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title('DT')
plt.show()
plt.figure(figsize=(12, 8), dpi=500)
plot_tree(dt_pca, filled=True, rounded=True, feature_names=data.feature_names,
          class_names=data.target_names)
plt.title('DT_PCA')
plt.show()


# In[22]:
# 绘制学习曲线
def learningCurve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=10)  # 支持向量机

    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
    plt.title('Learning Curve_' + str(model))
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()


# 绘制混淆矩阵
def confusionMatrix(model):
    plot_confusion_matrix(model, X_test, y_test)
    plt.title('Confusion Matrix_' + str(model))


# In[23]:
learningCurve(nb)
confusionMatrix(nb)
# In[24]:
# 预测测试集结果
lr_pred = lr.predict(X_test)
svc_pred = svc.predict(X_test)
knn_pred = knn.predict(X_test)

dt_pred = dt.predict(X_test)
nb_pred = nb.predict(X_test)
# In[25]:
# 主成分分析预测测试集结果
lr_pred_pca = lr_pca.predict(X_test_new)
svc_pred_pca = svc_pca.predict(X_test_new)
knn_pred_pca = knn_pca.predict(X_test_new)
dt_pred_pca = dt_pca.predict(X_test_new)
nb_pred_pca = nb_pca.predict(X_test_new)
# In[26]:
# 打印模型评估指标
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_pred))  # 分类准确率分数
print("Precision:", precision_score(y_test, lr_pred, average='macro'))  # 计算精度
print("Recall:", recall_score(y_test, lr_pred, average='macro'))  # 召回率
print("F1 Score:", f1_score(y_test, lr_pred, average='macro'))
print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, svc_pred))
print("Precision:", precision_score(y_test, svc_pred, average='macro'))
print("Recall:", recall_score(y_test, svc_pred, average='macro'))
print("F1 Score:", f1_score(y_test, svc_pred, average='macro'))
print("K-Nearest Neighbors:")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Precision:", precision_score(y_test, knn_pred, average='macro'))
print("Recall:", recall_score(y_test, knn_pred, average='macro'))
print("F1 Score:", f1_score(y_test, knn_pred, average='macro'))
print("Decision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, average='macro'))
print("Recall:", recall_score(y_test, dt_pred, average='macro'))
print("F1 Score:", f1_score(y_test, dt_pred, average='macro'))
print("Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, nb_pred))

print("Precision:", precision_score(y_test, nb_pred, average='macro'))
print("Recall:", recall_score(y_test, nb_pred, average='macro'))
print("F1 Score:", f1_score(y_test, nb_pred, average='macro'))
# In[27]:
# 打印模型评估指标
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_pred_pca))  # 分类准确率分数
print("Precision:", precision_score(y_test, lr_pred_pca, average='macro'))  # 计算精度
print("Recall:", recall_score(y_test, lr_pred_pca, average='macro'))  # 召回率
print("F1 Score:", f1_score(y_test, lr_pred_pca, average='macro'))
print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, svc_pred_pca))
print("Precision:", precision_score(y_test, svc_pred_pca, average='macro'))
print("Recall:", recall_score(y_test, svc_pred_pca, average='macro'))
print("F1 Score:", f1_score(y_test, svc_pred_pca, average='macro'))
print("K-Nearest Neighbors:")
print("Accuracy:", accuracy_score(y_test, knn_pred_pca))
print("Precision:", precision_score(y_test, knn_pred_pca, average='macro'))
print("Recall:", recall_score(y_test, knn_pred_pca, average='macro'))
print("F1 Score:", f1_score(y_test, knn_pred_pca, average='macro'))
print("Decision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_pred_pca))
print("Precision:", precision_score(y_test, dt_pred_pca, average='macro'))
print("Recall:", recall_score(y_test, dt_pred_pca, average='macro'))
print("F1 Score:", f1_score(y_test, dt_pred_pca, average='macro'))
print("Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, nb_pred_pca))
print("Precision:", precision_score(y_test, nb_pred_pca, average='macro'))
print("Recall:", recall_score(y_test, nb_pred_pca, average='macro'))
print("F1 Score:", f1_score(y_test, nb_pred_pca, average='macro'))


# In[28]:
def ROC(model, model_pca):  # 绘制ROC曲线
    ax = plt.gca()

    model_disp = plot_roc_curve(model, X_test, y_test, ax=ax, label=str(model))
    # PCA
    model_pca_disp = plot_roc_curve(model_pca, X_test_new, y_test, ax=ax, label=str(model) + "_PCA")
    plt.title("ROC Curve")
    plt.show()


ROC(nb, nb_pca)  # 绘制主成分分析ROC曲线
# In[29]:
clf1 = LogisticRegression(random_state=42)  # 逻辑回归
clf2 = SVC(kernel='linear', random_state=42, probability=True)  # 支持向量机
clf3 = KNeighborsClassifier(n_neighbors=4)  # KNN
clf5 = LogisticRegression(random_state=42)  # 逻辑回归
voting = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('knn', clf3)], voting='soft')
voting.fit(X_train, y_train)
# In[30]:
learningCurve(voting)
confusionMatrix(voting)
voting_pred = voting.predict(X_test)
print(voting.score(X_test, y_test))
print("voting:")
print("Accuracy:", accuracy_score(y_test, voting_pred))
print("Precision:", precision_score(y_test, voting_pred, average='macro'))
print("Recall:", recall_score(y_test, voting_pred, average='macro'))
print("F1 Score:", f1_score(y_test, voting_pred, average='macro'))
svc_pca = SVC(kernel='linear', random_state=42)  # 支持向量机


def ROC_voting():
    ax = plt.gca()
    voting_disp = plot_roc_curve(voting, X_test, y_test, ax=ax)
    knn_disp = plot_roc_curve(lr, X_test, y_test, ax=ax)
    dt_disp = plot_roc_curve(svc, X_test, y_test, ax=ax)

    svc_disp = plot_roc_curve(knn, X_test, y_test, ax=ax)
    plt.title("ROC Curve")
    plt.show()


# In[31]:
ROC_voting()
# In[32]:
lr_ada = LogisticRegression(random_state=42)  # 逻辑回归
ada_lr = AdaBoostClassifier(base_estimator=lr_ada, learning_rate=0.55, algorithm="SAMME.R", n_estimators=49)  # ADA
ada_lr.fit(X_train, y_train)
# learningCurve(ada_lr)
# confusionMatrix(ada_lr)
# In[33]:
learningCurve(ada_lr)
confusionMatrix(ada_lr)
ada__lr_pred = ada_lr.predict(X_test)
print("AdaBoostClassifier_LogisticRegression:")
print("Accuracy:", accuracy_score(y_test, ada__lr_pred))  # 分类准确率分数
print("Precision:", precision_score(y_test, ada__lr_pred, average='macro'))  # 计算精度
print("Recall:", recall_score(y_test, ada__lr_pred, average='macro'))  # 召回率
print("F1 Score:", f1_score(y_test, ada__lr_pred, average='macro'))
# In[34]:
# In[35]:
dt_ada = DecisionTreeClassifier(random_state=42, max_depth=3)  # 决策树
ada_dt = AdaBoostClassifier(base_estimator=dt_ada, learning_rate=0.85, n_estimators=150, random_state=42)  # ADA
ada_dt.fit(X_train, y_train)
# In[36]:
learningCurve(ada_dt)
confusionMatrix(ada_dt)

ada_dt_pred = ada_dt.predict(X_test)
print("AdaBoostClassifier_ DecisionTreeClassifier:")
print("Accuracy:", accuracy_score(y_test, ada_dt_pred))  # 分类准确率分数
print("Precision:", precision_score(y_test, ada_dt_pred, average='macro'))  # 计算精度
print("Recall:", recall_score(y_test, ada_dt_pred, average='macro'))  # 召回率
print("F1 Score:", f1_score(y_test, ada_dt_pred, average='macro'))
# In[37]:
dt_1 = DecisionTreeClassifier(random_state=42)  # 决策树
dt_1.fit(X_train, y_train)
importances = dt.feature_importances_
# sort
index = np.argsort(importances)[::-1]
print("Feature Ranking")
for i in range(0, 10):
    print("(%d) %s %f" % (index[i], df.columns[index[i] + 2], importances[index[i]]))
plt.bar(range(10), importances[index][0:10])
plt.xticks(range(30), index)
# In[38]:
labels = ["others"]
values = [0]
explode = [0]
for i in range(0, 30):
    # print("(%d) %s %f" % (index[i], dataset.columns[index[i]+2], importances[index[i]]))
    if i < 11:
        labels.append(df.columns[index[i] + 2])
        values.append(importances[index[i]])
        if i < 5:
            explode.append(0.2)
        else:
            explode.append(0)
    else:
        values[0] = values[0] + importances[index[i]]
for i in range(12):
    print("%s %f" % (labels[i], values[i]))
# plt.bar(range(11), values)
plt.pie(values, labels=labels, explode=explode)
