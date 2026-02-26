import streamlit as st
import pandas as pd
import joblib

# ---------------- Load Model ----------------
model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---------------- App Title ----------------
st.set_page_config(
    page_title="ChurnGuardAI",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📊 ChurnGuardAI - Customer Churn Prediction")
st.markdown(
    """
    Predict the likelihood of a customer churning from your service. 
    Enter the details below and get a **probability score** along with **risk category**.
    """
)
st.write("---")

# ------------------- UI Inputs -------------------
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0)
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service",
                                    ["Fiber optic", "No"])
with col2:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    payment_method = st.selectbox("Payment Method",
                                  ["Electronic check", "Mailed check",
                                   "Credit card (automatic)"])

# ------------------- Predict Button -------------------
if st.button("Predict Churn"):

    # ------------------- Validation -------------------
    if tenure == 0 or monthly_charges == 0.0 or total_charges == 0.0:
        st.warning("Please fill in all numeric fields before predicting!")
    else:

        # 1. Create empty dataframe with all training columns
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

        # 2. Fill numeric features
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly_charges
        input_data["TotalCharges"] = total_charges

        # 3. Map UI selections to training column names
        contract_map = {
            "Month-to-month": "Contract_Month-to-month",
            "One year": "Contract_One year",
            "Two year": "Contract_Two year"
        }

        payment_map = {
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
        }

        internet_map = {
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No"
        }

        # 4. Set categorical features safely
        def safe_set_column(col_name):
            if col_name in input_data.columns:
                input_data[col_name] = 1

        safe_set_column(contract_map[contract])
        safe_set_column(payment_map[payment_method])
        safe_set_column(internet_map[internet_service])

        # 5. Predict probability
        probability = model.predict_proba(input_data)[0][1]
        prediction = probability > 0.35

        # ------------------- Display Results -------------------
        st.write("---")
        st.header("Prediction Results")

        # Color mapping for risk
        if probability > 0.7:
            risk = "High Risk"
            color = "red"
        elif probability > 0.4:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "Low Risk"
            color = "green"

        # Two-column layout for metrics
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{probability:.2%}")
        col2.markdown(f"<h3 style='color:{color};'>{risk}</h3>", unsafe_allow_html=True)

        # Visual progress bar
        st.progress(min(int(probability * 100), 100))

        # Detailed text
        if prediction:
            st.error(
                f"⚠ Customer is likely to churn. Probability: {probability:.2%}. "
                f"Take proactive actions to retain the customer."
            )
        else:
            st.success(
                f"✅ Customer is likely to stay. Probability: {probability:.2%}. "
                f"Focus on maintaining satisfaction."
            )

        st.write("---")
        st.markdown(
            """
            **Tip:** Use this tool to identify high-risk customers and implement retention strategies.  
            Adjust marketing, offers, or support to reduce churn.
            """
        )