# train .py
#----------------------------------------------------------
# 1-Import the libraries
#----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import mlflow
import os
import duckdb 

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from smote_local import SMOTE  
from sklearn.pipeline import Pipeline

#Metrics
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix, 
classification_report,
auc,
roc_curve)

print ('libraries imported')

#------------------------------------------------------------------
# 2-Configurations and functions
#-----------------------------------------------------------------
PROJECT_ROOT=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(PROJECT_ROOT,'data')
DB_PATH = os.path.join(DATA_DIR, 'fraud_detection.db')

#-----Loading data and cleaning function-----------------

def load_data(dbpath, table_name='transactions'):
    print(f'load data from {dbpath} ...')
    conn=duckdb.connect (database=dbpath, read_only=True)
    df=conn.execute(f'SELECT * FROM {table_name}').fetchdf()
    conn.close()

    #data cleaning steps
    print('removing duplicates')
    df=df.drop_duplicates()

    print('removing missing rows')
    df=df.dropna()

    print('time variables creation')
    df['Time']=pd.to_datetime(df['Time'], unit='s')
    df['hours']=df['Time'].dt.hour #converting seconds in hours
    df.drop(columns=['Time'])
    print(f'new shape df has {df.shape[0]} rows and {df.shape[1]} columns')
    return df

##----Preprocessing and resampling function----
def preprocess_and_resample(df):
    #Defining target and independant features
    X=df.drop('Class', axis=1)
    y=df['Class']

    #Splitting the data set in train and test set
    X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    #1-preprocessing our columns
    num_cols=X_train.select_dtypes(include='number').columns.tolist()
    cat_cols=X_train.select_dtypes(include='object').columns.tolist()
    
    preprocessor=ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    #Applying our preprocessor
    X_train_processed=preprocessor.fit_transform(X_train)
    X_test_processed=preprocessor.transform(X_test)
    feature_names=preprocessor.get_feature_names_out()

    # 2-Resampling our data set
    print(' resampling our training set')
    print('y before sampling')
    print(y_train.value_counts())
    smote=SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print('y after sampling')
    print(pd.Series(y_train_resampled).value_counts())
    return X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor

#---Functions to log artifacts---
#For saving the classification report
def save_classification_report( report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
#for visualizing the confusion matrix
def plot_and_save_confusion_matrix(y_test, y_pred, file_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['allowed', 'fraud'], yticklabels=['allowed', 'fraud'])
    plt.title('confusion matrix')
    plt.ylabel('true class')
    plt.xlabel('predicted class')
    plt.savefig(file_path)
    plt.close()

#For ploting the auc_roc_curve for training and testing sets

def plot_roc_curves(y_train, y_train_probas, y_test, y_test_probas, file_path):
    plt.figure(figsize=(8, 6))

    # --- Testing set ---
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probas)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, 
             label=f'Test ROC curve (AUC = {roc_auc_test:.3f})')

    # ---Training set ---
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probas)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, linestyle='--',
             label=f'Train ROC curve (AUC = {roc_auc_train:.3f})')

    # --- Line of difference ---
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random guess')

    # --- Mise en forme du graphique ---
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # --- Save and log ---
    plt.savefig(file_path)
    plt.close()

    # Return AUC scores
    return roc_auc_train, roc_auc_test


#---------------------------------------------------
# 3-Script execution
#--------------------------------------------------
if __name__== "__main__" :
    #set the experiment name on mlflow
    mlflow.set_experiment('Fraud_Detection_Experiment')
     # Load data
    df = load_data(DB_PATH)
    
    # Preprocess data
    X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor = preprocess_and_resample(df)
    
    # Models to test
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    #Training the models
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"run_{model_name}_smote"):
            
            #log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("resampling_strategy", "SMOTE")
            mlflow.log_params(model.get_params())
            
            #run the model
            print(f"--- Model training : {model_name} ---")
            model.fit (X_train_resampled, y_train_resampled)

            #Predictions
            print("Predictions on processed on train and test set...")
            y_train_probas = model.predict_proba(X_train_resampled)[:, 1]
            y_test_pred = model.predict(X_test_processed)
            y_test_probas = model.predict_proba(X_test_processed)[:, 1]

            #Compute and log metrics
            print('metrics on test set')

            recall = recall_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            
            mlflow.log_metric("recall_score", recall)
            mlflow.log_metric("precision_score", precision)
            mlflow.log_metric("f1_score", f1)

            # Create and log artifacts
            # a. Classification report
            report = classification_report(y_test, y_test_pred)
            save_classification_report(report, "classification_report.txt")
            mlflow.log_artifact("classification_report.txt")
            
            # b. Confusion matrix
            plot_and_save_confusion_matrix(y_test, y_test_pred, "confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            
            # c-Roc curve
            roc_auc_train, roc_auc_test = plot_roc_curves(
                y_train_resampled, y_train_probas, 
                y_test, y_test_probas, 
                "roc_auc_curves.png"
            )
            #metrics auc
            mlflow.log_metric("roc_auc_train", roc_auc_train)
            mlflow.log_metric("roc_auc_test", roc_auc_test)
            mlflow.log_artifact("roc_auc_curves.png")

            # d_log the model
            from sklearn.pipeline import Pipeline
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                artifact_path=f"model_{model_name}",
            )

            print(f"--- Run {model_name} finished and is in MLflow ---")
            

            

            
        
    
    
    