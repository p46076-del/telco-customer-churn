import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

st.set_page_config(page_title="ABC Ltd. - Churn Evaluator", layout="wide")

st.title("ABC Ltd. | Customer Churn Risk Evaluator")
st.write("Adjust customer attributes below to evaluate immediate risk probability using Linear and Logistic Regression models.")

@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    features = ['tenure', 'MonthlyCharges', 'Contract', 'TechSupport']
    X = pd.get_dummies(df[features], columns=['Contract', 'TechSupport'], drop_first=True)
    y = df['Churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model 1: Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_scaled, y)

    # Model 2: Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_scaled, y)
    
    return log_model, lin_model, scaler, X.columns.tolist()

log_model, lin_model, scaler, model_columns = load_and_train()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Attributes")
    tenure = st.slider("Customer Tenure (Months)", 1, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Has Tech Support?", ["No", "Yes", "No internet service"])

    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    
    if f"Contract_{contract}" in input_data.columns:
        input_data[f"Contract_{contract}"] = 1
    if f"TechSupport_{tech_support}" in input_data.columns:
        input_data[f"TechSupport_{tech_support}"] = 1

    input_scaled = scaler.transform(input_data)
    
    # Predictions
    churn_prob_log = log_model.predict_proba(input_scaled)[0][1]
    churn_score_lin = lin_model.predict(input_scaled)[0]

with col2:
    st.subheader("Model Predictions")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Logistic Regression (Probability)", value=f"{churn_prob_log*100:.1f}%")
    with m2:
        st.metric(label="Linear Regression (Risk Index)", value=f"{churn_score_lin:.2f}")

    if churn_prob_log > 0.6:
        st.error("HIGH RISK: Recommend offering long-term contract discounts.")
    elif churn_prob_log > 0.3:
        st.warning("MEDIUM RISK: Recommend targeted email check-in.")
    else:
        st.success("LOW RISK: Account healthy.")

    st.markdown("---")
    st.subheader("Feature Drivers Comparison")
    coef_df = pd.DataFrame({
        'Feature': model_columns,
        'Logistic Coef': log_model.coef_[0],
        'Linear Coef': lin_model.coef_
    })
    st.dataframe(coef_df.sort_values(by='Logistic Coef', ascending=False), use_container_width=True)
