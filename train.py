import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load local dataset file
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Preprocess numerical & target fields
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Select core features & apply One-Hot Encoding
features = ['tenure', 'MonthlyCharges', 'Contract', 'TechSupport']
X = pd.get_dummies(df[features], columns=['Contract', 'TechSupport'], drop_first=True)
y = df['Churn']

# 4. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Save Model Artifacts
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("Generated files successfully: churn_model.pkl, scaler.pkl, model_columns.pkl")
