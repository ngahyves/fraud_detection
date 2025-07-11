#api.py

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

# --- 1. Setting --- 


RUN_ID = "04aa10ed62db4c0583c68f0848b6a095"
MODEL_NAME = "model_XGBoost"  

# Road to mlfow
LOGGED_MODEL_URI = f"runs:/{RUN_ID}/{MODEL_NAME}"
model_path = os.path.join(os.getcwd(), "mlruns", "333269949149011152", RUN_ID, "artifacts", MODEL_NAME)

# --- 2. Load the model
print(f"Load the model from mlflow from {model_path}")
model = mlflow.pyfunc.load_model(model_uri=model_path)
print("Model load successfully.")

# --- 3. Input data---
class Transaction(BaseModel):
    
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    hours: int 

# --- 4. API creation ---
app = FastAPI(
    title="fraud detection API",
    description="API to detect the probability of fraud",
    version="1.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API online"}

@app.post("/predict", tags=["Prediction"])
def predict(transaction: Transaction):
    """
    Prediction of probability
    """
    # Convert the input in DataFrame
    data_df = pd.DataFrame([transaction.dict()])
    
    # Make the prediction
    prediction = model.predict(data_df)

    #Decision
    result = int(prediction[0])
    decision = "APPROVE"
    if result == 1:
        decision = "FLAG_FOR_REVIEW"
        
    return {
        "prediction_code": result,
        "decision": decision
    }

