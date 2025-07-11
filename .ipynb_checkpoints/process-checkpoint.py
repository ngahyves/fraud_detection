#process and data cleaning

import pandas as pd
import numpy as np
import kaggle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---
DATA_DIR = 'data'
RAW_DATA_FILE = os.path.join(DATA_DIR, 'creditcard.csv')
KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'


# --- 2. Functions for each task ---

#Loading data
def download_data():
    """Step A: Download data from kaggle if they don't exist."""
    print("--- Loading data ---")
    
    # Create the data folder
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_DATA_FILE):
        print("-> File exists.")
        return
    
    try:
        print("-> Download from kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
        print("-> Download successful.")
    except Exception as e:
        print(f"Error: Download failed. Error : {e}")
        raise

def load_data(file_path):
    """Step B : Create a pandas data frame"""
    print("\n--- Step B: Chargement des données ---")
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found.")
        return None
        
    df = pd.read_csv(file_path)
    print(f"-> Data shape : {df.shape}")
    return df

#Cleaning the data
def clean_data(df):
    """Step C :Remove duplicates and missings."""
    print("\n--- Step C: Data cleaning ---")
    
    # Missings
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)
        print("-> missings removed")
    else:
        print("->missings not found")

    # Duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"-> {num_duplicates} duplicates removed.")
    else:
        print("-> Duplicates not found.")
        
    print(f"-> New shape after cleaning : {df.shape}")
    return df

def exploratory_data_analysis(df):
    """Step D : Exploratory data analysis."""
    print("\n--- Step D : Exploratory data analysis ---")
    
    # Distribution of the target
    print("\nDistribution of the target 'Class:")
    print(df['Class'].value_counts(normalize=True) * 100)

    # Outliers anlysis
    print("\nOutliers analysis:")
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if num_outliers > 0:
            print(f"- Column '{col}': {num_outliers} outliers détected.")
            
    #Visualisations of amount and 'class' variables
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(x=df['Amount'], ax=axes[0], color='skyblue')
    axes[0].set_title("Distribution of amount")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Fréquency")
    
    # Pie chart de la variable 'Class'
    df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1], colors=['tomato', 'limegreen'])
    axes[1].set_title("Fraud pie chart")
    axes[1].set_ylabel("") 
    
    plt.tight_layout()
    plt.show()

def run_full_pipeline():
    """Main function for all the pipeline"""
    
    # Step A: Download
    download_data()
    
    # Step B: Load
    dataframe = load_data(RAW_DATA_FILE)
    
    
    if dataframe is None:
        return
    
    # Step C: Clean
    cleaned_df = clean_data(dataframe)
    
    # Step D: Analyze
    exploratory_data_analysis(cleaned_df)
    
    cleaned_df.to_csv(os.path.join(DATA_DIR, 'processed_data.csv'), index=False)
    print("\n Pipeline finished, file saved")

    print("\n--- Pipeline finished ! ---")


#Define the main function
if __name__ == '__main__':
    run_full_pipeline()