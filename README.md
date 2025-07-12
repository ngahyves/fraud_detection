# End-to-End MLOps: Real-Time Fraud Detection System

This project demonstrates a comprehensive, end-to-end MLOps pipeline to build, deploy, and monitor a real-time fraud detection system. It showcases best practices in data science, model development, and operationalization, from initial data analysis to a containerized, multi-service application ready for production.

## Table of Contents
- [1. Business Problem](#1-business-problem)
- [2. The Machine Learning Pipeline](#2-the-machine-learning-pipeline)
  - [2.1. Data Ingestion and Cleaning](#21-data-ingestion-and-cleaning)
  - [2.2. Exploratory Data Analysis (EDA) & Feature Engineering](#22-exploratory-data-analysis-eda--feature-engineering)
  - [2.3. Model Training and Experimentation](#23-model-training-and-experimentation)
  - [2.4. Model Selection](#24-model-selection)
- [3. The MLOps & Production Pipeline](#3-the-mlops--production-pipeline)
  - [3.1. Real-Time Scoring API](#31-real-time-scoring-api)
  - [3.2. Containerization with Docker](#32-containerization-with-docker)
  - [3.3. CI/CD Automation](#33-cicd-automation)
  - [3.4. Local Orchestration & Monitoring](#34-local-orchestration--monitoring)
- [4. Tech Stack](#4-tech-stack)
- [5. Project Structure](#5-project-structure)
- [6. How to Run This Project](#6-how-to-run-this-project)
  - [6.1. Prerequisites](#61-prerequisites)
  - [6.2. Running Locally with Docker Compose](#62-running-locally-with-docker-compose)
- [7. How to Use the System](#7-how-to-use-the-system)
  - [7.1. The API](#71-the-api)
  - [7.2. The Monitoring Dashboard](#72-the-monitoring-dashboard)
- [8. Future Improvements](#8-future-improvements)

---

## 1. Business Problem

An online payment company is facing financial losses due to fraudulent credit card transactions. The objective is to develop an automated system capable of scoring incoming transactions in real-time to flag and block potential fraud before it is finalized, thereby minimizing losses and protecting customers.

The primary challenge is the severe class imbalance in the dataset—fraudulent transactions are extremely rare compared to legitimate ones.

## 2. The Machine Learning Pipeline

### 2.1. Data Ingestion and Cleaning
- Data is loaded from a local source using **DuckDB** for efficient in-memory processing.
- The initial cleaning process involves removing duplicate records and handling any missing values to ensure data quality.

### 2.2. Exploratory Data Analysis (EDA) & Feature Engineering
- **EDA:** An analysis was performed on the cleaned data to understand the distribution of features and the scale of the class imbalance.
- **Feature Engineering:** A new temporal feature, `hours` (hour of the day), was extracted from the `Time` column to capture potential time-based fraud patterns.

### 2.3. Model Training and Experimentation
A robust training pipeline was developed using `train.py` with the following key steps:
- **Imbalanced Data Handling:** To address the severe class imbalance, the **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set. This technique synthesizes new minority class examples, helping the model learn its characteristics better.
- **Model Comparison:** Three different classification models were trained and compared:
    1.  **Logistic Regression:** As a simple, interpretable baseline.
    2.  **Random Forest Classifier:** A powerful ensemble model.
    3.  **XGBoost Classifier:** A highly optimized gradient boosting model, often state-of-the-art for tabular data.
- **Experiment Tracking with MLflow:** Every training run was logged as an experiment in **MLflow**. This allowed for systematic tracking and comparison of:
    - **Parameters:** Model hyperparameters and preprocessing steps (e.g., use of SMOTE).
    - **Metrics:** A comprehensive set of classification metrics.
    - **Artifacts:** The trained model pipeline, confusion matrices, and ROC curves for each run.

### 2.4. Model Selection
- **Metric-Driven Choice:** For fraud detection, simply maximizing accuracy is misleading. The primary goal is to catch as many fraudulent transactions as possible, even at the cost of misclassifying some legitimate ones. Therefore, the **Recall score** (True Positive Rate) was chosen as the primary metric for model selection.
- **The Winning Model:** Based on the MLflow experiments, the **XGBoost Classifier** provided the best recall score while maintaining reasonable precision, making it the chosen model for deployment.

## 3. The MLOps & Production Pipeline

### 3.1. Real-Time Scoring API
A production-grade REST API was built using **FastAPI** to serve the selected XGBoost model.
- It exposes a `/predict` endpoint that accepts transaction data in JSON format.
- It uses **Pydantic** for rigorous data validation, ensuring robustness.
- It returns a clear decision (`APPROVE` or `FLAG_FOR_REVIEW`) in real-time.

### 3.2. Containerization with Docker
The entire application (API, model, and all dependencies) is encapsulated in a **Docker image**.
- The `Dockerfile` provides a reproducible recipe for building this image.
- This guarantees that the application runs identically on any machine, eliminating "it works on my machine" issues.

### 3.3. CI/CD Automation
A **Continuous Integration (CI)** pipeline is set up with **GitHub Actions**.
- On every `git push` to the `main` branch, the pipeline automatically:
    1.  Builds the Docker image.
    2.  Logs into Docker Hub using secured secrets.
    3.  Pushes the new version of the image to the project's [Docker Hub repository](https://hub.docker.com/r/ngahyves/fraud-detection), tagging it for version control.

### 3.4. Local Orchestration & Monitoring
- **Docker Compose:** A `docker-compose.yml` file orchestrates the entire system, allowing both the API and a monitoring dashboard to be launched with a single command (`docker-compose up`).
- **Streamlit Dashboard:** A simple, interactive dashboard provides a simulated real-time view of the API's performance, showing key metrics like the number of transactions processed and the detected fraud rate.

## 4. Tech Stack
- **Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost, DuckDB
- **MLOps:** MLflow, Docker, Docker Compose, GitHub Actions
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit

## 5. Project Structure


## Project Structure
├── .github/

│ └── workflows/

│ └── ci.yml # GitHub Actions workflow for CI

├── mlruns/ # MLflow experiment tracking data (local)

├── api.py # FastAPI application for scoring

├── dashboard.py # Streamlit monitoring dashboard

├── Dockerfile # Instructions to build the application's Docker image

├── docker-compose.yml # Orchestration file for running all services

├── .dockerignore # Specifies files to ignore in the Docker build context

├── .gitignore # Specifies files to ignore for Git

├── requirements.txt # Python dependencies

├── smote_local.py # Local implementation of SMOTE to avoid library conflicts

└── train.py # Script for model training and MLflow logging


## Setup and Installation

### 6.1. Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed and running.
- [Git](https://git-scm.com/) installed.
- A GitHub account.
- A Docker Hub account.

### 6.2. Running Locally with Docker Compose
1.  **Clone the repository:** `git clone https://github.com/ngahyves/fraud_detection.git`
2.  **Navigate to the directory:** `cd fraud_detection`
3.  **Build and run:** `docker-compose up --build`

## 7. How to Use the System
    ```bash
    git clone https://github.com/ngahyves/fraud_detection.git
    cd fraud_detection
    ```

2.  **Build and run the services:**
    This command will build the Docker images (if they don't exist) and start the containers for the API and the dashboard.
    ```bash
    docker-compose up --build
    ```

The system is now running!

## Usage

### 7.1. The API
- **Health Check:** [http://localhost:8000](http://localhost:8000)
- **Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 7.2. The Monitoring Dashboard
- **URL:** [http://localhost:8501](http://localhost:8501)

#### Example API Request (using curl)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52, "V6": -1.42, "V7": -2.53,
    "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89, "V13": -0.59, "V14": -4.28,
    "V15": 0.38, "V16": -1.14, "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12, "V21": 0.51,
    "V22": -0.03, "V23": -0.46, "V24": 0.32, "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14,
    "Amount": 0, "hours": 12
  }'
```

Accessing the Monitoring Dashboard
Dashboard URL: Open your browser to http://localhost:8501
Click the "Simulate a transaction" button to send requests to the API and see the dashboard update in real-time.
#### 8-Future Improvements and containerisation details
CD Pipeline
This project is configured with a Continuous Integration (CI) pipeline using GitHub Actions.
Trigger: The workflow runs on every push to the main branch.
Process:
Checks out the code.
Logs into Docker Hub using secrets.
Builds the Docker image for the application.
Pushes the image to Docker Hub, tagging it with latest and the Git commit SHA for versioning.
Result: A new, tested, and versioned Docker image is available on Docker Hub after every update.

- **Continuous Deployment (CD):** Automate deployment to a cloud environment.
- **Data Persistence:** Use a real database for logging predictions.
- **Advanced Model Monitoring:** Implement data and concept drift detection.
- **Automated Retraining:** Build an orchestration pipeline for periodic retraining.
