# Fichier : scripts/ingest_data.py
# C'est VOTRE structure de code, adaptée à notre projet !

import pandas as pd
import duckdb
import kaggle
import os

# --- Configuration ---
# On définit les chemins de manière relative, comme vous l'avez fait.
# '__file__' est le chemin du script actuel. 'os.path.dirname' récupère le dossier du script.
# 'os.path.join(..., '..')' remonte d'un niveau pour atteindre la racine du projet.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'fraud_detection.db')
TABLE_NAME = 'transactions'
KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'


def ingest_data_from_kaggle(dataset_name, data_dir, db_path, table_name):
    """
    Télécharge les données depuis Kaggle, les lit, et les charge dans une base DuckDB.
    La table est remplacée à chaque exécution pour assurer un état propre.
    """
    try:
        # --- Étape 1: Téléchargement automatique depuis Kaggle ---
        print("Étape 1: Téléchargement des données depuis Kaggle...")
        os.makedirs(data_dir, exist_ok=True)
        kaggle.api.authenticate()  # Doit fonctionner maintenant que le .kaggle/ est bien placé !
        kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        print("-> Téléchargement et décompression réussis.")

        # --- Étape 2: Lecture du CSV avec Pandas ---
        csv_file_path = os.path.join(data_dir, 'creditcard.csv')
        print(f"\nÉtape 2: Lecture du fichier CSV '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)
        print(f"-> {len(df)} lignes et {len(df.columns)} colonnes trouvées.")

        # --- Étape 3: Connexion à la base de données et ingestion ---
        print(f"\nÉtape 3: Chargement des données dans la table '{table_name}' de la base '{db_path}'...")
        con = duckdb.connect(database=db_path)
        # On remplace la table si elle existe, et on y insère le contenu du DataFrame.
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

        # Vérification
        record_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"-> Succès ! La table contient maintenant {record_count} enregistrements.")
        con.close()

        print("\n--- Ingestion des données terminée avec succès ! ---")

    except Exception as e:
        print(f"\nUNE ERREUR INATTENDUE EST SURVENUE : {e}")
        print("Veuillez vérifier votre configuration Kaggle (fichier kaggle.json) et vos permissions de dossier.")


# --- Point d'entrée du script ---
# C'est une excellente pratique de faire comme ça.
if __name__ == '__main__':
    # On a juste à appeler notre fonction principale.
    ingest_data_from_kaggle(KAGGLE_DATASET, DATA_DIR, DB_PATH, TABLE_NAME)