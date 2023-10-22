#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fruit Classifier using Multiple ML Algorithms

This script classifies fruits based on their attributes using different machine learning algorithms and evaluates
their performance to find the most accurate model.

Author: Fadi Helal
Date: Nov 28, 2019
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Load and preview the data
def load_data(filepath):
    data = pd.read_table(filepath)
    print("\nFirst 10 rows of the dataset:\n", data.head(10))
    print("\nDataset shape:", data.shape)
    print("\nUnique fruits:", data['fruit_name'].unique())
    print("\nDataset description:\n", data.describe())
    print("\nNumber of each fruit type:\n", data.groupby('fruit_name').size())
    return data


# Plot data
def plot_data(data):
    sns.countplot(data['fruit_name'], label="Count")
    plt.show()


# Data preprocessing
def preprocess_data(data, feature_names):
    X = data[feature_names]
    y = data['fruit_label']
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# Evaluate multiple classifiers
def evaluate_classifiers(X_train, y_train, X_test, y_test):
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=5000)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('K-NN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('SVM', SVC())
    ]

    best_accuracy = 0
    best_classifier = None

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        print(f'\nAccuracy of {name} on training set: {train_accuracy:.2f}')
        print(f'Accuracy of {name} on test set: {test_accuracy:.2f}')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_classifier = (name, clf)

    return best_classifier


# Main execution
if __name__ == "__main__":
    DATA_PATH = 'Downloads/Machine-Learning-with-Python-master/fruit_data_with_colors.txt'
    FEATURES = ['mass', 'width', 'height', 'color_score']

    fruits_data = load_data(DATA_PATH)
    plot_data(fruits_data)
    X_train, X_test, y_train, y_test = preprocess_data(fruits_data, FEATURES)
    best_clf_name, best_clf = evaluate_classifiers(X_train, y_train, X_test, y_test)

    # Evaluate best classifier
    print(f"\n{best_clf_name} had the highest accuracy.")
    predictions = best_clf.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
