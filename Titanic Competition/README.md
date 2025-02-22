# 🚢 Kaggle Competition: [Titanic - Machine Learning from Disaster]  

[![Kaggle Competition Badge](https://img.shields.io/badge/Titanic-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)  
[![License](https://img.shields.io/github/license/1AyaNabil1/Kaggle-Competition.svg)](https://github.com/1AyaNabil1/Kaggle-Competition/blob/main/LICENSE)  

## 🏆 Predicting Survival on the Titanic Kaggle Competition  
This directory contains the code and steps to predict passenger survival in the Titanic dataset for the Kaggle competition. The code achieves a **0.76555 score**. 🚢

---

## 📌 Introduction  
This directory provides the complete workflow for predicting passenger survival in the Titanic disaster using machine learning. The goal is to leverage various passenger features to build an accurate model. This README outlines data analysis, feature engineering, and the machine learning approach used.

## 📖 Table of Contents  
- [🚢 Kaggle Competition: \[Titanic - Machine Learning from Disaster\]](#-kaggle-competition-titanic---machine-learning-from-disaster)
  - [🏆 Predicting Survival on the Titanic Kaggle Competition](#-predicting-survival-on-the-titanic-kaggle-competition)
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
This Kaggle competition involves predicting passenger survival in the **Titanic disaster** using demographic and ticket information. The objective is to develop a precise model that improves our understanding of survival factors. 🚢

---

## 2️⃣ ⚙️ Installation <a name="installation"></a>  
Before running the code, install the required libraries:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## 3️⃣ 📊 Data Preparation <a name="data-preparation"></a>  
Load the datasets:  
```python
import numpy as np
import pandas as pd 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
df = pd.concat([train, test])
```
Check for missing values and data types:  
```python
df.isnull().sum()
df.info()
```

---

## 4️⃣ 📈 Exploratory Data Analysis <a name="exploratory-data-analysis"></a>  
Gain insights into survival distribution and trends:
✅ Visualize **survival rates by passenger class & gender** 🎭  
✅ Analyze the impact of **age & family size** 👨‍👩‍👦  
✅ Explore **ticket price & cabin assignment effects** 💰  

---

## 5️⃣ 🛠 Feature Engineering <a name="feature-engineering"></a>  
🔹 Extract titles from passenger names (e.g., Mr., Miss, Master) 🏷️  
🔹 Create new features such as **family size** and **deck location** 🚢  
🔹 Normalize **Fare** and **Age** values 📏  

---

## 6️⃣ 🤖 Machine Learning <a name="machine-learning"></a>  
📌 **Key steps in model development:**  
🔹 Feature selection 🎯  
🔹 **Handling categorical variables** with encoding 🏷️  
🔹 Training **multiple models**: Logistic Regression, Decision Trees, Random Forest, XGBoost 🚀  
🔹 **Evaluating** performance using accuracy & F1-score 📊  

---

## 7️⃣ 📡 Modeling <a name="modeling"></a>  
Define **features & target variable**, split the data, and train the model using XGBoost:  
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

X, y = train.drop(columns=['Survived']), train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        # Model hyperparameters
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 8️⃣ 📉 Evaluation <a name="evaluation"></a>  
Evaluate model performance using **accuracy, precision, recall, and F1-score** 📈.  

---

## 9️⃣ 🔍 Feature Importance <a name="feature-importance"></a>  
Identify the most influential features in predicting **survival on the Titanic** 🧐.  

---

## 🔟 📤 Submission <a name="submission"></a>  
Generate predictions for the test dataset and prepare the submission file:  
```python
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pipeline.predict(test)})
submission.to_csv('submission.csv', index=False)
```

---

🚀 **Let's achieve top ranks on the Kaggle leaderboard!** 🏆
