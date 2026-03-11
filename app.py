import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# ---- Load and prepare data ----
@st.cache_data
def load_and_train():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    # Fix missing values
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    # Feature engineering
    df["LoanIncomeRatio"] = df["LoanAmount"] / df["ApplicantIncome"]

    # Encode
    le = LabelEncoder()
    for col in ["Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"]:
        df[col] = le.fit_transform(df[col])

    X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
    Y = df["Loan_Status"]
    return train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_test, Y_train, Y_test = load_and_train()

# Train models
dt = DecisionTreeClassifier(random_state=42).fit(X_train, Y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, Y_train)
gb = GradientBoostingClassifier(random_state=42).fit(X_train, Y_train)

# ---- UI ----
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.markdown("Fill in the applicant details below to predict loan approval.")

st.sidebar.header("Applicant Details")

gender        = st.sidebar.selectbox("Gender", ["Male", "Female"])
married       = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents    = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education     = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
income        = st.sidebar.slider("Applicant Income (₹)", 1000, 100000, 5000)
co_income     = st.sidebar.slider("Coapplicant Income (₹)", 0, 50000, 0)
loan_amount   = st.sidebar.slider("Loan Amount (₹ thousands)", 10, 700, 150)
loan_term     = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit        = st.sidebar.selectbox("Credit History", ["Good (1)", "Bad (0)"])
model_choice  = st.sidebar.selectbox("Choose Model", ["Random Forest", "Decision Tree", "Gradient Boosting"])

# Encode inputs
gender_e     = 1 if gender == "Male" else 0
married_e    = 1 if married == "Yes" else 0
dep_e        = 3 if dependents == "3+" else int(dependents)
edu_e        = 0 if education == "Graduate" else 1
self_e       = 1 if self_employed == "Yes" else 0
prop_e       = 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
credit_e     = 1 if credit == "Good (1)" else 0
ratio        = loan_amount / income

input_data = pd.DataFrame([[
    gender_e, married_e, dep_e, edu_e, self_e,
    income, co_income, loan_amount, loan_term,
    credit_e, prop_e, ratio
]], columns=X_train.columns)

# Predict
if st.button("🔍 Predict Loan Approval"):
    model = rf if model_choice == "Random Forest" else (dt if model_choice == "Decision Tree" else gb)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ LOAN APPROVED — {probability:.1f}% confidence")
    else:
        st.error(f"❌ LOAN REJECTED — {100-probability:.1f}% confidence rejection")

    st.subheader("📊 All Models Comparison")
    results = {
        "Decision Tree"     : dt.predict(input_data)[0],
        "Random Forest"     : rf.predict(input_data)[0],
        "Gradient Boosting" : gb.predict(input_data)[0]
    }
    for m, r in results.items():
        st.write(f"{'✅ Approved' if r==1 else '❌ Rejected'} — {m}")
