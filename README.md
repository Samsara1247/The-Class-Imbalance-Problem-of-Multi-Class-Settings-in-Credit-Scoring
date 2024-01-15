# The-Class-Imbalance-Problem-of-Multi-Class-Settings-in-Credit-Scoring
The Class Imbalance Problem Revisited: Comparison of the Effectiveness of Different Models for Multi-Class Settings in Credit Scoring

The rise of financial defaults has forced financial institutions to invest a lot of costs in studying more comprehensive and complete credit prediction systems. The class imbalance problem is one of the biggest obstacles they encounter. In the past, most research focuses on binary imbalanced datasets. However, currently a number of real-world problems are characterized by more than two classes, while still being affected by skewed distributions. The literature on multi-class imbalance is scarce. This paper contributes on the benchmark of various solutions to multi-class imbalanced problems and the comparison of performance between different techniques. The empirical results show that various sampling techniques and cost-sensitive learning do not remarkably enhance the effectiveness, while CatBoost performs significantly better than other algorithms and models. We find that powerful ensemble algorithms gain higher performance than data-level and algorithm-level solutions when dealing with multi-class imbalance problems.

This repo contains datasets, codes and results relevant to the paper, providing further experimental conditions and parameters for interested readers. Moreover, please read the short description before viewing each file.

The experiments are performed on five public credit scoring datasets obtained from Kaggle and other institutions. Here are their links:

Dataset 1 Ratings: http://www.creditriskanalytics.net/datasets-private2.html

Dataset 2 ProsperLoan: https://www.kaggle.com/datasets/yousuf28/prosper-loan

Dataset 3 Credit Score: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

Dataset 4 Lending Club: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1?select=Loan_status_2007-2020Q3.gzip

Dataset 5 Credit Card: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
