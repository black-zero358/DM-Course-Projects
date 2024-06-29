import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器（使用ID3算法）
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Classification Report:')
print(class_report)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('Decision Tree')
plt.savefig('decision_tree.png')
plt.show()

# 打印决策树的文本表示
tree_rules = export_text(clf, feature_names=iris.feature_names)
print('Decision Tree Rules:')
print(tree_rules)

# 交叉验证模型的稳定性
cv_scores = cross_val_score(clf, X, y, cv=10)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean():.2f}')

# 超参数调优
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 最佳参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(f'Best parameters: {best_params}')

# 使用最优模型进行预测和评估
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Best model accuracy: {accuracy_best:.2f}')

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print('Confusion Matrix (Best Model):')
print(conf_matrix_best)

class_report_best = classification_report(y_test, y_pred_best, target_names=iris.target_names)
print('Classification Report (Best Model):')
print(class_report_best)

# 可视化最优决策树
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('Optimal Decision Tree')
plt.savefig('optimal_decision_tree.png')
plt.show()
