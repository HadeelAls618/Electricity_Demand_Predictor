# Hourly Electricity Demand Batch Prediction Service

## Project Overview  
This project delivers an **end-to-end, production-grade machine learning service** for **batch electricity demand forecasting**. We designed, implemented, and deployed a complete ML pipeline that automates data collection, preprocessing, model training, inference, and monitoring.  

The service helps **Public Service Company (PNM)** optimize electricity supply by predicting hourly energy demand, reducing costs, and improving operational efficiency.

## Business Problem

PNM aims to enhance its operational efficiency by forecasting electricity demand on an hourly basis. This will help align production, inventory, and resource allocation with real-time customer needs, reducing costs and improving service reliability.

### **Challenges:**  
- **Overproduction** → Wasted energy and higher costs.  
- **Shortages** → Customer dissatisfaction and service disruptions.  
- **Inefficient resource allocation** → Poor financial and operational performance.  

### **Solution**  
We built a **machine learning-based demand forecasting system** that enables PNM to:  
✅ Reduce **overproduction** by aligning energy generation with expected demand.  
✅ Prevent **shortages** by predicting peak-hour demand accurately.  
✅ Optimize **resource allocation** through data-driven energy distribution.  
✅ Improve **operational efficiency** by integrating automation in forecasting.  

## Business Objectives (SMART Framework)

### **Chosen Business Objective**:

Reduce operational costs by implementing a demand forecasting model, as it offers high impact with moderate effort.

### **Defining Ways to Achieve the Objective**

1. **Specific**: Develop an electricity demand forecasting model that predicts hourly demand.
2. **Measurable**: Reduce forecast error measured by Mean Absolute Error (MAE).
3. **Achievable**: Utilize historical demand data, weather data, and calendar events to enhance forecasting accuracy.
4. **Relevant**: Helps in optimizing energy production and minimizing waste.
5. **Time-bound**: Achieve a stable and accurate model within a defined project timeline.

## **End-to-End Machine Learning Pipeline**  
We developed a **three-stage ML pipeline** using modern MLOps practices, including feature stores, model registries, and automated inference.

### **1. Feature Pipeline**

This pipeline fetches and preprocesses the required data, transforms it into a time-series format, and stores it in the **Hopsworks Feature Store**.

#### **Steps:**

1. Fetch historical electricity demand data from the **EIA API**.  https://www.eia.gov/opendata/
2. Fetch historical weather data (temperature) from **OpenWeatherMap API**. https://open-meteo.com/
3. Merge data sources based on timestamps.
4. Transform features:
   - **Prediction Target**: Hourly electricity demand for NY.
   - **Features**:
     - Hourly temperature for NY
     - Day and month as integer features
     - Bank holiday status (binary: True/False)
     - Lag-based features for time-series analysis
5. Store transformed data in the Hopsworks Feature Store.

### **2. Training Pipeline**

This pipeline fetches data from the feature store, processes it into feature-target format, trains a machine learning model, and evaluates it before saving the trained model in the **Hopsworks Model Registry**.

#### **Steps:**

1. Load preprocessed features and targets from the feature store.
2. Train a **LightGBM model** with hyperparameter tuning using **Optuna** with 5-fold cross-validation.
3. Implement **feature engineering**:
   - Adding US Federal Holidays using `pandas.tseries.holiday`.
   - Incorporating time-series features (lags, rolling statistics).
4. Evaluate the model using **Mean Absolute Error (MAE)**.
5. Store the trained model in the **Hopsworks Model Registry**.

#### **Performance**

- The best model configuration was selected based on MAE.
- The estimated MAE on the test set is **approximately 25 GWh**, aligning with EIA forecasts.

### **3. Inference Pipeline**

The inference pipeline is responsible for generating hourly electricity demand predictions using the trained model.

#### **Steps:**

1. Fetch the latest feature set from the Hopsworks Feature Store.
2. Load the trained model from the **Hopsworks Model Registry**.
3. Generate hourly demand predictions.
4. Compare predictions against actual demand data to compute **Mean Absolute Error (MAE)**.
5. The pipeline runs as a **serverless function in `inference_pipeline.py`**, scheduled via **GitHub Actions** to execute every hour.

## Monitoring and Deployment

### **Forecasting app**

- A **Streamlit application** visualizes forecasted electricity demand for different locations in NYC. https://electricitydemandpredictor-3gaww4pzqsw6orh3vnkc4f.streamlit.app/
- The application features an interactive **map** where the size of circles represents demand at specific locations.
- Users can explore electricity demand variations across locations in real-time.

### **Monitoring Dashboard**   https://electricitydemandmonitor-2pe99wef2a5mdvtjnjqt4d.streamlit.app/

- A separate **Streamlit dashboard** monitors model performance on an hourly basis.
- The dashboard includes:
  - A real-time chart of **MAE trends**
  - Historical performance records
  - Comparison of predicted vs. actual demand
- This allows continuous monitoring and optimization of the model.

### **Deployment**

- The forecasting model is integrated into a **production-ready service**.
- Both the **forecasting app** and **monitoring dashboard** are deployed using **Streamlit**.
- The entire system ensures real-time, reliable electricity demand prediction and performance tracking.

## Summary

This project successfully implements a batch prediction system for hourly electricity demand forecasting. By leveraging machine learning and data science, PNM can optimize energy production and improve efficiency while minimizing costs. The system includes both a **forecasting app** for visualizing demand across locations and a **monitoring dashboard** for evaluating model accuracy. This **production-ready** service demonstrates end-to-end data science and machine learning deployment, making it an excellent showcase of applied machine learning skills.



  
