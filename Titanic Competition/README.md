# Kaggle Competition: [Titanic - Machine Learning from Disaster]

[![Kaggle Competition Badge](https://img.shields.io/badge/Titanic-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)
[![License](https://img.shields.io/github/license/1AyaNabil1/Kaggle-Competition.svg)](https://github.com/1AyaNabil1/Kaggle-Competition/blob/main/LICENSE)

This repository contains my solution for the Kaggle competition "[Titanic - Machine Learning from Disaster]" hosted at [Kaggle](https://www.kaggle.com/competitions/titanic).

In this competition, In this project, we explore the famous Titanic dataset and build predictive models to determine the survival outcome of passengers.

## Competition Description

The Titanic dataset is divided into two parts: training and test datasets. I load the data, combine them into a single DataFrame, and perform data analysis and preprocessing to prepare it for machine learning.
_____________________________________________________________________________
# Solution Overview

## Exploring the Data

We begin by loading the training and test datasets, and merging them into a single DataFrame for analysis.

```python
import numpy as np
import pandas as pd 

training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
df = pd.concat([training, test])
```

### Data Exploration
I explore the data to understand its structure and characteristics. Key observations include:

* PassengerId, Age, SibSp, Parch, Fare are <code>numerical features</code>.
* Name, Sex, Ticket, Pclass, Cabin, Survived, and Embarked are <code>categorical features</code>.

## Data Preprocessing

### Handling Missing Values
I address missing values in the dataset by imputing or dropping them as necessary. For example:

```python
df.Age = df.Age.fillna(training.Age.median())
df = df dropna(subset=["Embarked"])
```

### Feature Engineering
We engineer new features, such as:

* Extracting the title from the passenger names.
* Creating a "cabin_multiple" feature to count the number of cabins.
* Normalizing the "Fare" feature.

## Model Building

We build and evaluate several machine learning models using cross-validation:
1. Gaussian Naive Bayes
2. Logistic Regression
3. Decision Tree
4. K-Nearest Neighbors (KNN)
5. Random Forest
6. Support Vector Classifier (SVC)
7. Xtreme Gradient Boosting (XGBoost) --> "the best model"

> The models are evaluated based on their cross-validation scores, and the best performing model is selected for predictions.

## Model Optimization
We further optimize the best model using hyperparameter tuning with GridSearchCV or RandomizedSearchCV.

## Model Evaluation
The final selected model's performance is assessed on a validation set or through cross-validation. The evaluation metrics used may include accuracy, precision, recall, F1-score, and ROC-AUC.

## Model Deployment
Once the best model is selected and trained on the entire training dataset, it can be used for making predictions on new, unseen data. The predictions can be saved to a CSV file for submission.

## Results
The code with 49.4s run code with 0.76555 score
________________________________________________________________________
This repository provides a comprehensive overview of the Titanic survival prediction project, from data exploration to model building and evaluation. Feel free to explore the code and use it as a reference for your own machine learning projects.
