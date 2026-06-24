import streamlit as st
import pandas as pd
import joblib

# 1. Page Config & Custom Styling (Dark/Modern theme vibes)
st.set_page_config(
    page_title="RiskRadar | Loan Approval Engine", 
    page_icon="🏦",
    layout="wide"
)

# Custom CSS to make it look premium
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #1f77b4; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allowed_html=True)

st.title("🏦 RiskRadar™ — Loan Approval Predictor")
st.caption("Because relying on a human credit officer's mood is so last century. Let the algorithm decide.")

# 2. Load Model Safely
@st.cache_resource
def load_model():
    try:
        return joblib.load('loan_approval_model.pkl')
    except FileNotFoundError:
        st.error("Error: 'loan_approval_model.pkl' not found. Did it get lost in the mail?")
        return None

model = load_model()

if model:
    # 3. Layout: Splitting into Sidebar and Main Section
    st.sidebar.header("👤 Applicant Profile")
    st.sidebar.subheader("Personal Metrics")
    
    age = st.sidebar.slider("Age", min_value=20, max_value=80, value=28)
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    education = st.sidebar.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master"])
    experience = st.sidebar.slider("Years of Employment Experience", min_value=0, max_value=40, value=5)
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    # Main Panel Layout
    st.subheader("💰 Loan Requirements & Financial Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        income = st.number_input("Annual Income ($)", min_value=0, value=55000, step=1000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=12000, step=500)
        
    with col2:
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_int_rate = st.slider("Interest Rate (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
        
    with col3:
        credit_score = st.slider("Credit (CIBIL) Score", min_value=300, max_value=850, value=680)
        credit_history = st.number_input("Credit History Length (Years)", min_value=0, max_value=30, value=4)

    # Calculated metrics shown dynamically
    loan_percent = loan_amount / max(income, 1)
    
    st.write("---")
    st.subheader("🚨 Risk Disclosures")
    previous_loan = st.radio("Has the applicant defaulted on a historical loan?", ["No", "Yes"], horizontal=True)

    st.write("---")
    
    # 4. Predict Trigger
    if st.button("Evaluate Credit Risk"):
        # Preparing features exactly how the ColumnTransformer expects them
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Education': education,
            'Person Income': income,
            'Employee Experience': experience,
            'Home Onwership': home_ownership,
            'Loan Amount': loan_amount,
            'Loan Intent': loan_intent,
            'Loan interest Rate': loan_int_rate,
            'Loan percentage': round(loan_percent, 2),
            'Credit History': credit_history,
            'Credit Score': credit_score,
            'Previous Loan': previous_loan
        }])
        
        with st.spinner("Crunching numbers... Please wait while we judge your financial life choices."):
            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[0]
            approval_prob = probabilities[1]
            
            # Big beautiful result blocks
            st.subheader("🎯 System Verdict")
            
            res_col1, res_col2 = st.columns([2, 1])
            
            if prediction[0] == 1:
                with res_col1:
                    st.success(f"### 🎉 Approved!\nThe system trusts you. Do not make us regret this.")
                with res_col2:
                    st.metric(label="Confidence Score", value=f"{approval_prob:.1%}")
            else:
                with res_col1:
                    st.error(f"### ❌ Rejected.\nYour profile looks a bit too risky for our taste. Try asking your parents?")
                with res_col2:
                    st.metric(label="Risk Factor", value=f"{probabilities[0]:.1%}")
