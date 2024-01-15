# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:49:41 2023

@author: Jianwei Peng
"""

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/credit risk analytics_ratings_data1.csv')
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# separate the target variable from the features
X = df.drop(['rating'], axis=1)
y = df['rating']

# Encoding the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the dataset into training and test sets with the ratio 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Defining the classifiers
classifiers = [
    LogisticRegression(max_iter=100000),
    RandomForestClassifier(),
    XGBClassifier(),
    MLPClassifier(max_iter=10000)
]

# Creating a dictionary to store the results
results = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'F1': [], 'Confusion Matrix': []}

# Running the experiments
for classifier in classifiers:
    print(f"Classifier: {classifier}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Evaluating accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.5f}")
    # Evaluating precision score
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"Precision Score: {precision:.5f}")
    # Evaluating f1 score
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"F1 Score: {f1:.5f}")
    # Displaying the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    # Adding the results to the dictionary
    results['Classifier'].append(str(classifier))
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['F1'].append(f1)
    results['Confusion Matrix'].append(cm)
    # Plotting the confusion matrix
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print("-----------------------")

# Creating a dataframe from the results dictionary
df_results = pd.DataFrame(results)

# Saving the dataframe to an excel file
df_results.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/新建 Microsoft Excel 工作表.xlsx', index=False)
