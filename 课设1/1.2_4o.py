#!/usr/bin/env python
# coding: utf-8

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


def load_and_preprocess_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data


def explore_data(df, data):
    print(df.head())
    print("数据关键词:", data.keys())
    print("特征名称:", list(data.feature_names))
    print(df.describe())
    print(data.target_names)
    print(pd.Series(data.target).value_counts())
    print(df.isnull().sum())
    return df, data


def visualize_data(df):
    df.hist(bins=10, figsize=(20, 15))
    plt.show()
    plt.figure(figsize=(16, 9))
    sns.boxplot(data=df, orient='h')
    plt.show()
    plt.figure(figsize=(16, 9))
    sns.violinplot(data=df, orient='h')
    plt.show()
    corr_df = df.corr().abs()
    sns.heatmap(corr_df, annot=False, cmap='coolwarm')
    plt.show()


def evaluate_model(model, X, y, name=""):
    pred = model.predict(X)
    print(f"{name} - Accuracy: {accuracy_score(y, pred):.4f}")
    print(f"{name} - Precision: {precision_score(y, pred, average='macro'):.4f}")
    print(f"{name} - Recall: {recall_score(y, pred, average='macro'):.4f}")
    print(f"{name} - F1 Score: {f1_score(y, pred, average='macro'):.4f}")


def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
    plt.title('Learning Curve_' + str(model))
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, X, y):
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    disp.ax_.set_title('Confusion Matrix_' + str(model))
    plt.show()


def plot_roc_curve(models, X_test, y_test):
    ax = plt.gca()
    for model in models:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=str(model))
    plt.title("ROC Curve")
    plt.show()


def main():
    df, data = load_and_preprocess_data()
    explore_data(df, data)
    visualize_data(df)

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=5)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.transform(X_test)

    models = [
        (LogisticRegression(random_state=42), "Logistic Regression"),
        (SVC(kernel='linear', random_state=42), "Support Vector Machine"),
        (KNeighborsClassifier(n_neighbors=4), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (GaussianNB(), "Naive Bayes")
    ]

    for model, name in models:
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)
        plot_learning_curve(model, X_train, y_train)
        plot_confusion_matrix(model, X_test, y_test)

        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_new, y_train)
        evaluate_model(model_pca, X_test_new, y_test, f"{name} (PCA)")

    voting = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                          ('svc', SVC(kernel='linear', random_state=42, probability=True)),
                                          ('knn', KNeighborsClassifier(n_neighbors=4))], voting='soft')
    voting.fit(X_train, y_train)
    evaluate_model(voting, X_test, y_test, "Voting Classifier")
    plot_learning_curve(voting, X_train, y_train)
    plot_confusion_matrix(voting, X_test, y_test)
    plot_roc_curve([voting], X_test, y_test)

    ada_lr = AdaBoostClassifier(LogisticRegression(random_state=42), learning_rate=0.55, algorithm="SAMME",
                                n_estimators=49)
    ada_lr.fit(X_train, y_train)
    evaluate_model(ada_lr, X_test, y_test, "AdaBoost (Logistic Regression)")
    plot_learning_curve(ada_lr, X_train, y_train)
    plot_confusion_matrix(ada_lr, X_test, y_test)

    ada_dt = AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth=3), learning_rate=0.85,
                                n_estimators=150, algorithm="SAMME", random_state=42)
    ada_dt.fit(X_train, y_train)
    evaluate_model(ada_dt, X_test, y_test, "AdaBoost (Decision Tree)")
    plot_learning_curve(ada_dt, X_train, y_train)
    plot_confusion_matrix(ada_dt, X_test, y_test)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    importances = dt.feature_importances_
    index = np.argsort(importances)[::-1]

    print("Feature Ranking:")
    for i in range(10):
        print(f"({index[i]}) {df.columns[index[i]]} {importances[index[i]]:.6f}")

    plt.bar(range(10), importances[index][:10])
    plt.xticks(range(10), df.columns[index][:10], rotation=90)
    plt.show()

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


if __name__ == "__main__":
    main()

