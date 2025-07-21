# Predicting Diabetes Risk with Logistic Regression

A complete, end-to-end machine learning project using logistic regression to predict diabetes risk from patient health data.  
This project demonstrates data cleaning, feature engineering, model training, evaluation, and interpretation—**with real-world medical relevance**.

---

## 📊 Project Highlights

- **Dataset:** Pima Indians Diabetes (from Kaggle)
- **Goal:** Predict the risk of diabetes using health indicators
- **Model:** Logistic Regression (with all preprocessing and evaluation steps)
- **Key Skills:** Data cleaning, EDA, feature scaling, model evaluation, ROC/AUC, feature importance, result interpretation

---

## 🗂️ Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Workflow & Methods](#workflow--methods)
- [Results](#results)
- [Discussion](#discussion)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Diabetes is a global health challenge. Early prediction can lead to better patient outcomes.  
Here, we use **Logistic Regression**—one of the most interpretable and widely used classifiers—to estimate the probability that a patient has diabetes based on basic health indicators.

---

## Dataset Overview

- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (1 = Diabetes, 0 = No diabetes)
- **Source:** [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Workflow & Methods

1. **Data Loading & Exploration:** Checked for missing values and suspicious zeros in the dataset.
2. **Data Cleaning:** Replaced zeros in medically impossible fields (e.g., Glucose, Insulin) with NaN, then imputed missing values using the median.
3. **Feature/Target Split:** X = features, y = outcome.
4. **Train-Test Split:** 70% training, 30% test set; stratified to preserve class balance.
5. **Feature Scaling:** Standardized all features (zero mean, unit variance).
6. **Model Training:** Used `sklearn`’s LogisticRegression.
7. **Evaluation:**  
   - **Accuracy:** 74%
   - **Confusion Matrix:** Good at identifying negatives, moderate sensitivity to positives.
   - **Classification Report:** High precision, moderate recall (especially important in medical tasks)
   - **ROC Curve / AUC:** AUC = 0.84 (strong performance)
   - **Feature Importance:** Glucose, BMI, and Pregnancies most predictive
8. **Visualization:**  
   - ROC curve for classifier performance
   - Bar plot for feature importance

---

## Results

- **Accuracy:** 74%
- **AUC:** 0.84 (good model discrimination)
- **Most Important Features:** Glucose, BMI, Pregnancies, Diabetes Pedigree Function

### Confusion Matrix

|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| 0    |    129      |     21      |
| 1    |    38       |     43      |

### Classification Report

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     |   0.77    |  0.86  |   0.81   |
| 1     |   0.67    |  0.53  |   0.59   |

### Feature Importance

- Glucose: **1.13**
- BMI: **0.73**
- Pregnancies: **0.46**
- DiabetesPedigreeFunction: **0.22**
- Others less important

---

## Discussion

- **Interpretability:** Logistic regression is easy to interpret; most influential factors are medically plausible.
- **Strengths:** Robust performance on basic features, can be easily explained to medical professionals.
- **Limitations:** Sensitivity/recall for positive class could be improved; false negatives are critical in medical applications.
- **Next Steps:** Try threshold tuning, advanced models (Random Forest, XGBoost), and cross-validation.

---

## Future Work

- Hyperparameter tuning and regularization
- Model ensemble and comparison
- Deploy as a simple web app for non-technical use
- Ethical considerations: always validate model predictions with real clinicians

---

## References

- [Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- scikit-learn documentation
- Project PDF included in this repo

---

**Feel free to fork, clone, or contribute!**
