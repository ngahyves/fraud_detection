# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import requests 
import time

# --- Setting the page ---
st.set_page_config(
    page_title="Dashboard Monitoring - fraud detection",
    page_icon="🤖",
    layout="wide"
)

# --- Dashboard title ---
st.title("🤖 Dashboard of streaming in real time")
st.write("This dashboard simulates the monitoring of our fraud detection API..")

API_URL = "http://api:8000/predict"

# --- Real-time query simulation ---
st.header(" Simulation of API calls")

if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Simulate a transaction"):
    # We randomly choose between a ‘healthy’ case and a ‘fraud’ case.
    if np.random.rand() > 0.1: #Simulation of a LEGITIMATE transaction
        st.info("Simulation of a LEGITIMATE transaction.")
        test_data = {
          "V1": 1.19, "V2": 0.26, "V3": 0.16, "V4": 0.44, "V5": 0.06, "V6": -0.08, "V7": -0.07,
          "V8": 0.08, "V9": -0.25, "V10": -0.16, "V11": 1.61, "V12": 1.06, "V13": 0.48, "V14": -0.14,
          "V15": 0.63, "V16": 0.46, "V17": -0.11, "V18": -0.18, "V19": -0.14, "V20": -0.06, "V21": -0.22,
          "V22": -0.63, "V23": 0.10, "V24": -0.33, "V25": 0.16, "V26": 0.12, "V27": -0.00, "V28": 0.01,
          "Amount": 2.69, "hours": 10
        }
    else: # 10% chance of being a fraud
        st.warning("Simulation of being a fraud")
        test_data = {
          "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52, "V6": -1.42, "V7": -2.53,
          "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89, "V13": -0.59, "V14": -4.28,
          "V15": 0.38, "V16": -1.14, "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12, "V21": 0.51,
          "V22": -0.03, "V23": -0.46, "V24": 0.32, "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14,
          "Amount": 0, "hours": 12
        }
    
   #This part is linked to docker compose
    try:
        response = requests.post(API_URL, json=test_data)
        response.raise_for_status() # Throws an exception if the status is an error (4xx or 5xx)
        prediction = response.json()
        st.session_state.history.append({"data": test_data, "prediction": prediction})
    except requests.exceptions.RequestException as e:
        st.error(f"Error connection to the l'API : {e}")
        st.info("This is normal if you launch this dashboard on its own. It must be launched with Docker Compose to communicate with the API..")

# --- Metrics history ---
st.header("Metrics history ")

if not st.session_state.history:
    st.info("Click on the button above to simulate transactions..")
else:
    # Create a DataFrame from the history
    history_df = pd.DataFrame([item['prediction'] for item in st.session_state.history])
    
    total_transactions = len(history_df)
    frauds_detected = history_df[history_df['decision'] == 'FLAG_FOR_REVIEW'].shape[0]
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Sum of simulated transactions", total_transactions)
    
    if total_transactions > 0:
        fraud_rate = (frauds_detected / total_transactions) * 100
        col2.metric("percent of fraud detected", f"{fraud_rate:.2f}%")

    # Display a graph of fraud
    st.subheader("Decision distribution")
    st.bar_chart(history_df['decision'].value_counts())

    #Display historics details
    with st.expander("Display historics details"):
        st.dataframe(pd.DataFrame(st.session_state.history))