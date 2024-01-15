# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:38:52 2023

@author: Jianwei Peng
"""

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTENC
from imblearn.under_sampling import NeighbourhoodCleaningRule, OneSidedSelection
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from logitboost import LogitBoost
from catboost import CatBoostClassifier

df = pd.read_csv('C:/Users/Jianwei Peng/Desktop/新建文件夹/Handled datasets/Credit Score Classification Clean Data_train_data3.csv')

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
    CatBoostClassifier(random_state=42, verbose=False),
    GradientBoostingClassifier(random_state=42, verbose=False),
    LGBMClassifier(class_weight='balanced', random_state=42),
    LogitBoost(random_state=42)
]

# Defining the sampling techniques
samplers = [
    SMOTETomek(),
    SMOTENC(categorical_features=[11,15,18]),
    ADASYN(),
    NeighbourhoodCleaningRule(),
    OneSidedSelection()
]

# Creating a dictionary to store the results
results = {'Sampler': [], 'Classifier': [], 'ROC AUC (OvO)': [], 'ROC AUC (OvR)': []}

# Running the experiments
for sampler in samplers:
    print(f"Sampler: {sampler}")
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    for classifier in classifiers:
        print(f"Classifier: {classifier}")
        classifier.fit(X_resampled, y_resampled)
        y_pred = classifier.predict_proba(X_test)
        # Evaluating ROC AUC score (multi_class='ovo')
        roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class='ovo')
        print(f"ROC AUC (OvO): {roc_auc_ovo:.5f}")
        # Evaluating ROC AUC score (multi_class='ovr')
        roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print(f"ROC AUC (OvR): {roc_auc_ovr:.5f}")
        # Adding the results to the dictionary
        results['Sampler'].append(str(sampler))
        results['Classifier'].append(str(classifier))
        results['ROC AUC (OvO)'].append(roc_auc_ovo)
        results['ROC AUC (OvR)'].append(roc_auc_ovr)
    print("-----------------------")

# Creating a dataframe from the results dictionary
df_results = pd.DataFrame(results)
# Saving the dataframe to an excel file
df_results.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/Boosting results/auc_data3_Boost.xlsx', index=False)
