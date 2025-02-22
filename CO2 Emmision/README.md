# 🌍 Kaggle Competition: [Predict CO2 Emissions in Rwanda]  

[![Kaggle Competition Badge](https://img.shields.io/badge/CO2_Prediction-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s3e20?rvi=1)  
[![License](https://img.shields.io/github/license/1AyaNabil1/Kaggle-Competition.svg)](https://github.com/1AyaNabil1/Kaggle-Competition/blob/main/LICENSE)  

## 🔥 Predicting CO2 Emissions in Rwanda Kaggle Competition  
This directory contains the code and steps to predict CO2 emissions in Rwanda for the Kaggle competition. The code achieves a score of **28.09473**, placing it in the **top 24%** of submissions. 🚀

---

## 📌 Introduction  
This directory provides the complete workflow for predicting CO2 emissions in Rwanda as part of a Kaggle competition. 🌿 The goal is to leverage environmental and geographical features to develop an accurate model for CO2 emissions prediction. This README outlines data analysis, feature engineering, and the machine learning approach used.

## 📖 Table of Contents  
- [🌍 Kaggle Competition: \[Predict CO2 Emissions in Rwanda\]](#-kaggle-competition-predict-co2-emissions-in-rwanda)
  - [🔥 Predicting CO2 Emissions in Rwanda Kaggle Competition](#-predicting-co2-emissions-in-rwanda-kaggle-competition)
  - [📌 Introduction](#-introduction)
  - [📖 Table of Contents](#-table-of-contents)
  - [1️⃣ 📌 Introduction ](#1️⃣--introduction-)
  - [2️⃣ ⚙️ Installation ](#2️⃣-️-installation-)
  - [3️⃣ 📊 Data Preparation ](#3️⃣--data-preparation-)
  - [4️⃣ 📈 Exploratory Data Analysis ](#4️⃣--exploratory-data-analysis-)
  - [5️⃣ 🛠 Feature Engineering ](#5️⃣--feature-engineering-)
  - [6️⃣ 🤖 Machine Learning ](#6️⃣--machine-learning-)
  - [7️⃣ 📡 Modeling ](#7️⃣--modeling-)
  - [8️⃣ 📉 Evaluation ](#8️⃣--evaluation-)
  - [9️⃣ 🔍 Feature Importance ](#9️⃣--feature-importance-)
  - [🔟 📤 Submission ](#--submission-)

---

## 1️⃣ 📌 Introduction <a name="introduction"></a>  
This Kaggle competition involves predicting **CO2 emissions** in Rwanda using environmental and meteorological features. The objective is to develop a precise model that aids in understanding emissions trends and their potential environmental impact. 🌱

---

## 2️⃣ ⚙️ Installation <a name="installation"></a>  
Before running the code, install the required libraries:  
```bash
pip install numpy pandas matplotlib seaborn xgboost scikit-learn plotly
```

---

## 3️⃣ 📊 Data Preparation <a name="data-preparation"></a>  
Load the datasets:  
```python
sample = pd.read_csv("sample_submission.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```
Check for missing values and data types:  
```python
df.isnull().sum()
df.info()
```

---

## 4️⃣ 📈 Exploratory Data Analysis <a name="exploratory-data-analysis"></a>  
Gain insights into data distribution and trends:
✅ Visualize **CO2 emissions** distribution 📊  
✅ Analyze the impact of **COVID-19** on emissions 🦠  
✅ Explore **geographical & temporal trends** 🌍  

---

## 5️⃣ 🛠 Feature Engineering <a name="feature-engineering"></a>  
🔹 Create new features (e.g., **cyclic time features, holiday indicators, rotational coordinates**).  
🔹 Handle missing values with **imputation techniques**.  
🔹 Adjust emissions data to account for **COVID-19 effects in 2020**.  

---

## 6️⃣ 🤖 Machine Learning <a name="machine-learning"></a>  
📌 **Key steps in model development:**  
🔹 Feature selection 🎯  
🔹 **Dimensionality reduction** using PCA 📉  
🔹 Training **XGBoost** model 🚀  
🔹 **Evaluating** performance using RMSE 📊  

---

## 7️⃣ 📡 Modeling <a name="modeling"></a>  
Define **features & target variable**, split the data, and train the model using XGBoost:  
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

---

## 8️⃣ 📉 Evaluation <a name="evaluation"></a>  
Evaluate model performance using **RMSE** and visualize actual vs. predicted values 📈.  

---

## 9️⃣ 🔍 Feature Importance <a name="feature-importance"></a>  
Identify the most influential features in predicting **CO2 emissions** 🧐.  

---

## 🔟 📤 Submission <a name="submission"></a>  
Generate predictions for the test dataset and prepare the submission file:  
```python
sample['emission'] = pipeline.predict(test[features])
sample.to_csv('submission.csv', index=False)
```

---

🚀 **Let's achieve top ranks on the Kaggle leaderboard!** 🏆
