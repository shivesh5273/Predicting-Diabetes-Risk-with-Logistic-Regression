# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc)

# 2. Load Dataset
url = 'diabetes.csv'
df = pd.read_csv(url, header=None)
df = pd.read_csv(url)  # No header=None, just the default
print(df.columns)

# 3. Exploratory Data Analysis
print(df.head())
print(df.info())
print(df.describe())

# 4. Handle Missing Values
# Replace zeros with NaN for certain features
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
print(df.isnull().sum())

# Impute missing values with median (safer for medical data)
df.fillna(df.median(), inplace=True)

# 5. Feature/Target Split
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 7. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Predictions and Evaluation
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 10. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Diabetes Prediction')
plt.legend(loc='lower right')
plt.show()

# 11. Feature Importance (Coefficients)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)
print("\nFeature importance (absolute value):")
print(feature_importance)

sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()