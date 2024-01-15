# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:13:56 2023

@author: Jianwei Peng
"""

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/Lending Club accepted_2007_to_2018Q4_data4.csv')
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Separating the target variable and the features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

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
results = {'Classifier': [], 'ROC AUC (OvR)': []}

# Running the experiments
for classifier in classifiers:
    print(f"Classifier: {classifier}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)
    # Evaluating ROC AUC score (multi_class='ovr')
    roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print(f"ROC AUC (OvR): {roc_auc_ovr:.5f}")
    # Adding the results to the dictionary
    results['Classifier'].append(str(classifier))
    results['ROC AUC (OvR)'].append(roc_auc_ovr)
print("-----------------------")

# Creating a dataframe from the results dictionary
df_results = pd.DataFrame(results)
# Saving the dataframe to an excel file
df_results.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/新建 Microsoft Excel 工作表.xlsx', index=False)
