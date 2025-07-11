# File : scripts/ingest_data.py


import pandas as pd
import duckdb
import kaggle
import os

# --- settings ---
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'fraud_detection.db')
TABLE_NAME = 'transactions'
KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'


def ingest_data_from_kaggle(dataset_name, data_dir, db_path, table_name):
    """
    Download data from kaggle and read them with duckdb
    """
    try:
        # --- Step 1: Download---
        print("Downloading...")
        os.makedirs(data_dir, exist_ok=True)
        kaggle.api.authenticate() 
        kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        print("-> Download and uncompress successfully done.")

        # --- Step 2: Read the CSV with pandas ---
        csv_file_path = os.path.join(data_dir, 'creditcard.csv')
        print(f"\n step 2: read the csv file '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)
        print(f"-> {len(df)} rows and{len(df.columns)} columns found.")

        # --- Step 3: connect to the data base and ingestion---
        print(f"\nStep 3: connect to the data base and ingestion '{table_name}' of the data base '{db_path}'...")
        con = duckdb.connect(database=db_path)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        
        # Verification
        record_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"-> Sucess{record_count} records.")
        con.close()
        
        print("\n--- ingestion done ! ---")
        
    except Exception as e:
        print(f"\nError occured : {e}")
        print("Verify on kaggle api.")

# --- Entry point of the script
if __name__ == '__main__':
    # calling the function
    ingest_data_from_kaggle(KAGGLE_DATASET, DATA_DIR, DB_PATH, TABLE_NAME)