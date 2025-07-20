#  Hourly Electricity Demand Forecasting System ⚡

## Project Overview

This project presents a **production-ready, end-to-end electricity demand forecasting system** designed to optimize energy supply, reduce operational costs, and minimize CO₂ emissions for a **public energy provider**. Leveraging MLOps best practices, the solution is fully automated — integrating data collection, feature engineering, model training, inference, and continuous monitoring into a seamless ML pipeline.

![GIF](vid.gif)


## Business Problem

The Energy Company faces resource misallocation: inefficient distribution of energy resources leads to increased operational costs, higher carbon footprint, and reduced efficiency.


## ML Problem

Build a model that predicts **next-hour electricity demand**, enabling the company to align generation with forecasted needs — ensuring optimal resource allocation, reducing costs, and improving reliability.


## Data Sources

To support accurate demand forecasting, the project integrates data from multiple sources:

- **Historical Electricity Demand**: Retrieved from the [EIA API](https://www.eia.gov/opendata/)  
- **Weather Data**: Pulled from the [Open-Meteo Weather API](https://open-meteo.com/)  
- **Calendar Events**: Public holidays extracted using `pandas.tseries.holiday`


##  Project Methodology

The project is structured around a **three-stage ML pipeline**, aligned with MLOps principles like modular architecture, feature stores, model registries, and automated inference.


###  1. Feature Pipeline

Prepares time-series data and stores it in the **Hopsworks Feature Store**.

**Steps:**
- Fetch electricity demand data from the **EIA API**
- Fetch temperature data from **Open-Meteo API**
- Merge both datasets by timestamp
- Transform into time-series format with lag features and holiday flags
- Store engineered features in the **Hopsworks Feature Store**


###  2. Training Pipeline

Builds and registers the forecasting model.

**Steps:**
- Load features from the feature store
- Define:
-  **Target**: Next-hour electricity demand
-  **Features**: Temperature, time-based lags, rolling stats, holiday indicators
- Train a **LightGBM model** using **Optuna** for hyperparameter tuning and 5-fold cross-validation
- Evaluate model using **Mean Absolute Error (MAE)**
- Save the model to the **Hopsworks Model Registry**


###  3. Inference Pipeline

Generates hourly forecasts using the trained model.

**Steps:**
- Fetch the latest features from the Feature Store
- Load the model from the Model Registry
- Predict next-hour electricity demand
- Compute MAE against actuals
- Run pipeline every hour via **GitHub Actions** using a serverless script (`inference_pipeline.py`)


### Deployment & Monitoring

**Batch Forecasting App (Streamlit)**
An interactive app that visualizes predicted hourly electricity demand across NYC using a map-based interface with regional breakdowns.

**Monitoring Dashboard (Streamlit)**
Tracks model performance in real time, featuring MAE trends, historical insights, and comparisons between predicted and actual demand.


### Summary
This project covers the following key components:

*  **Automated ML Pipeline**: Covers data ingestion, training, inference, and monitoring
*  **Feature Store & Model Registry**: Enables scalable versioning and reproducibility
*  **Batch Forecasting**: Generates accurate hourly predictions across NYC regions
*  **Interactive Dashboard**: Provides stakeholders with clear, actionable insights
*  **Production-Ready**: Scheduled inference, retraining hooks, and modular architecture


## Contact

Want to collaborate or ask questions about this project?

-  [LinkedIn](https://www.linkedin.com/in/hadeel-als-0a23702a6)
-  [alsaadonhadeel@gmail.com](mailto:alsaadonhadeel@gmail.com)

