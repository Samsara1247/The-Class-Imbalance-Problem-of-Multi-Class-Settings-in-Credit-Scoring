# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:32:38 2023

@author: Jianwei Peng
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from logitboost import LogitBoost
import pandas as pd

dataset1 = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/credit risk analytics_ratings_data1.csv')
dataset2 = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/prosperLoanData_data2.csv')
dataset3 = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/Credit Score Classification Clean Data_train_data3.csv')
dataset4 = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/Lending Club accepted_2007_to_2018Q4_data4.csv')
dataset5 = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/credit card_0-1-2-3_data5.csv')

# Define a dictionary with the datasets and their names
datasets = {
    'rating': dataset1,
    'prosperloan': dataset2,
    'clean_data': dataset3,
    'Lending Club': dataset4,
    'credit_card': dataset5
}

# Define a dictionary to store the results
results = {}

# Loop over the datasets
for dataset_name, dataset in datasets.items():

    # Split the dataset into training and testing sets
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Define the classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False, loss_function='MultiClass'),
        'GradientBoost': GradientBoostingClassifier(random_state=42, verbose=False),
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
        'LogitBoost': LogitBoost(random_state=42)
    }

    # Fit and evaluate each classifier
    results[dataset_name] = {}
    for clf_name, classifier in classifiers.items():

        classifier.fit(X_train, y_train)
        # Predict the labels
        y_pred = classifier.predict(X_test)
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        # Store the results for the classifier
        results[dataset_name][clf_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'F1 Score': f1,
        }

# Convert the results dictionary into a dataframe and save it into an excel file
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                    for i in results.keys() 
                                    for j in results[i].keys()}, orient='index')
results_df.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/Boosting results/Boosting.xlsx')
