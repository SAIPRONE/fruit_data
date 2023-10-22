# Fruit Classifier using Multiple ML Algorithms

## Overview
This repository contains a Python script that classifies fruits based on their attributes using multiple machine learning algorithms. It evaluates the performance of each model to determine the most accurate classifier. 

## 🖋 Author
**Fadi Helal**

## 📅 Date
**Nov 28, 2019**

## 🍎 Features Used
- Mass
- Width
- Height
- Color Score

## 🤖 Algorithms Evaluated
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (K-NN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

## Getting Started

### Prerequisites
- Python 3.x
- Required Libraries:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

You can install the required libraries using the following command:
pip install pandas matplotlib seaborn scikit-learn

### 🏃 Running the Script
1. Clone the repository to your local machine.
2. Navigate to the directory containing the script.
3. Execute the script using the following command:

python fruit_data_classification.py

Replace `<script_name>` with the name you saved the script as.

### 📊 Dataset
The script uses a dataset named `fruit_data.csv`. Ensure this dataset is present in the mentioned path or update the `DATA_PATH` variable in the script with the appropriate location.

## 📈 Outputs
1. The first 10 rows of the dataset are displayed.
2. Shape, unique fruits, dataset description, and number of each fruit type are printed.
3. A bar graph visualizing the count of each fruit type.
4. The training and testing accuracy of each algorithm is printed.
5. The best classifier, based on test accuracy, is determined.
6. A confusion matrix and classification report for the best classifier are displayed.

## 📄 License
This project is licensed under the BSD 3-Clause License.
