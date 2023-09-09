# Kaggle Competition: [Predict CO2 Emissions in Rwanda]

[![Kaggle Competition Badge](https://img.shields.io/badge/CO2_Prediction-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s3e20?rvi=1)
[![License](https://img.shields.io/github/license/1AyaNabil1/Kaggle-Competition.svg)](https://github.com/1AyaNabil1/Kaggle-Competition/blob/main/LICENSE)

# Predicting CO2 Emissions in Rwanda Kaggle Competition
This directory  contains the code and steps to predict CO2 emissions in Rwanda for the Kaggle competition. The code achieves a score of 28.09473, which places it in the top 24% of submissions.

## Introduction

This directory contains the code and steps to predict CO2 emissions in Rwanda for the Kaggle competition. The goal of this competition is to predict CO2 emissions based on various environmental and geographical features. This README file provides an overview of the code, data analysis, feature engineering, and the machine learning model used for prediction.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Machine Learning](#machine-learning)
7. [Modeling](#modeling)
8. [Evaluation](#evaluation)
9. [Feature Importance](#feature-importance)
10. [Submission](#submission)

## 1. Introduction <a name="introduction"></a>

This Kaggle competition involves predicting CO2 emissions in Rwanda based on various environmental and meteorological features. The goal is to develop a model that accurately predicts CO2 emissions, which can have significant environmental and health implications.

## 2. Installation <a name="installation"></a>

Before running the code, ensure you have the required libraries installed:

```python
pip install numpy pandas matplotlib seaborn xgboost scikit-learn plotly
```

## 3. Data Preparation <a name="data-preparation"></a>
Load the datasets:
```python
sample = pd.read_csv("sample_submission.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```
Understand the nature of the data, check for missing values, and determine the data types:

```python
df.isnull().sum()
df.info()
```
## 4. Exploratory Data Analysis <a name="exploratory-data-analysis"></a>
Explore the data to gain insights into its distribution and trends:

* Visualize the distribution of emissions.
* Analyze the impact of COVID-19 on emissions.
* Explore geographical and temporal trends.

## 5. Feature Engineering <a name="feature-engineering"></a>

* Create new features such as cyclic features for months and weeks, holiday indicators, and rotational coordinates.
* Handle missing values using imputation techniques.
* Adjust emissions data for the COVID-19 effect in 2020.


## 6. Machine Learning <a name="machine-learning"></a>

* Select relevant features for modeling.
* Perform dimensionality reduction using PCA.
* Train a machine learning model, such as XGBoost, to predict CO2 emissions.
* Evaluate the model's performance using RMSE.

## 7. Modeling <a name="modeling"></a>

Define the features and target variable, split the data into training and testing sets, and train a model using XGBoost:

```python
X, y = train[features], train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(
        # Model hyperparameters
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

## 8. Evaluation <a name="evaluation"></a>
Evaluate the model's performance using RMSE and visualize the actual vs. predicted values.

## 9. Feature Importance <a name="feature-importance"></a>
Analyze feature importance to understand which features have the most impact on the model's predictions.

## 10. Submission <a name="submission"></a>
Prepare the submission file by predicting emissions for the test dataset and saving the results to a CSV file:
```python
sample['emission'] = pipeline.predict(test[features])
sample.to_csv('submission.csv', index=False)
```


