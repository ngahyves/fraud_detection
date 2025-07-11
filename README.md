# Real-Time Fraud Detection System with MLOps

This project implements a complete end-to-end MLOps pipeline for a real-time fraud detection system. It includes data ingestion, model training with experiment tracking, a real-time scoring API, and a monitoring dashboard, all containerized with Docker and orchestrated with Docker Compose. The pipeline is also configured for Continuous Integration (CI) with GitHub Actions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup with Docker Compose](#local-setup-with-docker-compose)
- [Usage](#usage)
  - [Accessing the API](#accessing-the-api)
  - [Accessing the Monitoring Dashboard](#accessing-the-monitoring-dashboard)
- [CI/CD Pipeline](#cicd-pipeline)
- [Future Improvements](#future-improvements)

## Project Overview

The goal of this project is to build a robust and scalable machine learning system to identify fraudulent credit card transactions. The system ingests a dataset, trains several classification models, tracks experiments to select the best one, serves it via a high-performance REST API, and provides a simple dashboard for monitoring.

The core business problem is to minimize financial losses for an online payment company by flagging potentially fraudulent transactions in real-time before they are processed.

## Features

- **Experiment Tracking:** Uses **MLflow** to log model parameters, metrics (Recall, Precision, F1-Score, AUC), and artifacts (models, confusion matrices).
- **Imbalanced Data Handling:** Implements a local version of the **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm to handle the class imbalance inherent in fraud detection datasets.
- **Real-Time Scoring API:** A high-performance API built with **FastAPI** to provide low-latency fraud predictions.
- **Containerization:** The entire application stack (API, model, dependencies) is containerized using **Docker**, ensuring consistency across environments.
- **Multi-Service Orchestration:** Uses **Docker Compose** to define and run the multi-container application (API + Dashboard) with a single command.
- **Continuous Integration (CI):** An automated **GitHub Actions** workflow builds and pushes the API's Docker image to Docker Hub on every push to the `main` branch.
- **Monitoring Dashboard:** A simple, interactive dashboard built with **Streamlit** to simulate real-time monitoring of API predictions.

## Tech Stack

- **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy
- **MLOps & Experiment Tracking:** MLflow
- **API Development:** FastAPI, Uvicorn
- **Containerization & Orchestration:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Dashboarding:** Streamlit
- **Data Ingestion (local):** DuckDB

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

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed and running.
- [Git](https://git-scm.com/) installed.
- A GitHub account.
- A Docker Hub account.

### Local Setup with Docker Compose

This is the recommended way to run the entire system locally. It will start both the API and the monitoring dashboard.

1.  **Clone the repository:**
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

### Accessing the API

- **API Health Check:** Open your browser to [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger UI):** Open [http://localhost:8000/docs](http://localhost:8000/docs) to test the `/predict` endpoint interactively.

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

Accessing the Monitoring Dashboard
Dashboard URL: Open your browser to http://localhost:8501
Click the "Simulate a transaction" button to send requests to the API and see the dashboard update in real-time.
CI/CD Pipeline
This project is configured with a Continuous Integration (CI) pipeline using GitHub Actions.
Trigger: The workflow runs on every push to the main branch.
Process:
Checks out the code.
Logs into Docker Hub using secrets.
Builds the Docker image for the application.
Pushes the image to Docker Hub, tagging it with latest and the Git commit SHA for versioning.
Result: A new, tested, and versioned Docker image is available on Docker Hub after every update.
Future Improvements
Continuous Deployment (CD): Add a CD stage to the GitHub Actions workflow to automatically deploy the new Docker image to a cloud server (e.g., AWS EC2, Google Cloud Run).
Data Persistence: Connect the dashboard to a database (like PostgreSQL or InfluxDB) to log and persist prediction history instead of storing it in-session.
Model Monitoring: Implement data drift and concept drift detection (e.g., using tools like Evidently AI or WhyLogs) to trigger alerts or model retraining.
Automated Retraining: Create an orchestration pipeline (e.g., with Airflow or Prefect) to periodically retrain the model on new data and register the best-performing version in MLflow.
