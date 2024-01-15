# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:36:27 2023

@author: Jianwei Peng
"""

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import NeighbourhoodCleaningRule, OneSidedSelection

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
    OneVsOneClassifier(LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000000)),
    OneVsOneClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
    OneVsOneClassifier(XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)),
    OneVsOneClassifier(MLPClassifier(random_state=42, max_iter=10000,hidden_layer_sizes=(2000,))),
    OneVsRestClassifier(LogisticRegression(class_weight='balanced', random_state=42, max_iter=100000000)),
    OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
    OneVsRestClassifier(XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)),
    OneVsRestClassifier(MLPClassifier(random_state=42, max_iter=10000,hidden_layer_sizes=(2000,)))
]

# Defining the sampling techniques
samplers = [
    SMOTETomek(),
    SMOTENC(categorical_features=[5,6,7,8,10,11,22,36]),
    ADASYN(),
    NeighbourhoodCleaningRule(),
    OneSidedSelection()
]

# Creating a dictionary to store the results
results = {'Sampler': [], 'Classifier': [], 'Accuracy': [], 'Precision': [], 'F1': [], 'Confusion Matrix': []}

# Running the experiments
for sampler in samplers:
    print(f"Sampler: {sampler}")
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    for classifier in classifiers:
        print(f"Classifier: {classifier}")
        classifier.fit(X_resampled, y_resampled)
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
        results['Sampler'].append(str(sampler))
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
df_results.to_excel('C:/Users/Jianwei Peng/Desktop/新建文件夹/Results/Results of data4.xlsx', index=False)