# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:07:23 2023

@author: Jianwei Peng
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from logitboost import LogitBoost
from catboost import CatBoostClassifier
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=0)

    # Define the classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'GradientBoost': GradientBoostingClassifier(random_state=42, verbose=False),
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
        'LogitBoost': LogitBoost(random_state=42)
    }

    # Fit and evaluate each classifier
    results[dataset_name] = {}
    for clf_name, classifier in classifiers.items():
        # Fit the classifier
        classifier.fit(X_train, y_train)

        # Predict the probabilities
        y_prob = classifier.predict_proba(X_test)

        # Calculate the AUC scores
        auc_ovo = round(roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo'), 5)
        auc_ovr = round(roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr'), 5)

        # Store the results for the classifier
        results[dataset_name][clf_name] = {
            'AUC (OneVsOne)': auc_ovo,
            'AUC (OneVsRest)': auc_ovr
        }

# Convert the results dictionary into a dataframe and save it into an excel file
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                    for i in results.keys() 
                                    for j in results[i].keys()}, orient='index')
results_df.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/Boosting results/Boosting auc.xlsx')