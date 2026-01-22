import streamlit as st
import requests
import json
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

from src.visualizations.shap_utils import make_shap_viz

API_URL = "http://127.0.0.1:8000/predict"
BATCH_API_URL = "http://127.0.0.1:8000/predict-batch"

st.title("Bank Churn Prediction Dashboard")

tab1, tab2 = st.tabs(['Single Prediction', 'Batch Prediction'])

with tab1:
    st.write("Enter customer details below to predict churn probability.")

    with st.form("prediction_form"):
        # Group inputs for better layout
        col1, col2 = st.columns(2)
    
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.slider("Tenure (Years)", 0, 10, 3)

        with col2:
            balance = st.number_input("Balance", min_value=0.0, value=60000.0)
            num_products = st.slider("Number of Products", 1, 4, 2)
            has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

        submit_button = st.form_submit_button("Predict Churn")

        if submit_button:

            payload = {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary,
                "compute_shap": True
            }

            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                churn_prob = result['churn_probability']
                is_churn = result['churn_prediction'] == 1

                # Display Results
                st.subheader("Prediction Results")
                if is_churn:
                    st.error(f"Churn Probability: {churn_prob:.2%}")
                else:
                    st.success(f"Churn Probability: {churn_prob:.2%}")
                
                # Display SHAP if computed
                if result.get('shap_values'):
                    st.pyplot(make_shap_viz(result['shap_values']))
            else:
                st.error(f"Error from API: {response.text}")


with tab2:
    st.write("Upload a CSV file with customer data to get batch predictions with SHAP analysis.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    predict_button = st.button("Predict with SHAP Analysis")
    
    if uploaded_file is not None and predict_button:
        # Send file to FastAPI for SHAP analysis
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(BATCH_API_URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            
            churn_rate = (result['churn_count'] / result['total_rows']) * 100
            st.success(f"Analyzed {result['total_rows']} customers | Churn Rate: {churn_rate:.2f}%")
            
            # Display SHAP beeswarm plot
            if result.get('shap_plot'):
                st.image(base64.b64decode(result['shap_plot']))
            
            # Download full results
            pred_df = pd.DataFrame(result['predictions'])
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Full Predictions CSV",
                data=csv,
                file_name="churn_predictions_with_shap.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Error from API: {response.text}")
                    