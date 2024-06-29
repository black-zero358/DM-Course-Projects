import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 将numpy数组转换为DataFrame，以便进行后续的数据处理
X = pd.DataFrame(data)
y = pd.DataFrame(target, columns=["PRICE"])

# 1.2 查看数据
print("First five rows of data:\n", X.head())
print("Target (Price):\n", y.head())

# 2. 特征工程
# 2.1 计算每个特征与房价之间的相关系数
correlations = X.corrwith(y['PRICE']).abs().sort_values(ascending=False)
print("Correlations with PRICE:\n", correlations)

# 2.2 选择相关性高的五个特征
top_features = correlations.head(5).index
print("Top 5 features:", top_features)

# 2.3 展示特征之间的相关性
sns.pairplot(X[top_features])
plt.show()

# 3. 构建线性回归模型
# 3.1 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X[top_features], y, test_size=0.3, random_state=42)

# 3.2 创建模型
model = LinearRegression()

# 3.3 训练模型
model.fit(X_train, y_train)
print("Model intercept:", model.intercept_)
print("Model coefficients:", model.coef_)
print("Training score (R^2):", model.score(X_train, y_train))

# 3.4 8折交叉验证
cv_scores = cross_val_score(model, X[top_features], y, cv=8)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# 3.5 预测测试集
y_pred = model.predict(X_test)

# 3.6 计算MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

def train_and_evaluate(X, y, features, model):
    """
    训练并评估模型。
    """
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    print("Model intercept:", model.intercept_)
    print("Model coefficients:", model.coef_)
    print("Training score (R^2):", model.score(X_train, y_train))
    cv_scores = cross_val_score(model, X[features], y, cv=8)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", np.mean(cv_scores))
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

# 使用所有的特征
print("Using all features:")
all_features = list(range(X.shape[1]))
train_and_evaluate(X, y, all_features, LinearRegression())

# 使用前5个最相关的特征
print("\nUsing top 5 features:")
top_5_features = correlations.head(5).index.tolist()
train_and_evaluate(X, y, top_5_features, LinearRegression())

# 使用前10个最相关的特征
print("\nUsing top 10 features:")
top_10_features = correlations.head(10).index.tolist()
train_and_evaluate(X, y, top_10_features, LinearRegression())
