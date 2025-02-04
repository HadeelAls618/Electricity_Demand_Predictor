# Batch Prediction Service for Hourly Electricity Demand in the US  

## Project Overview  
This project delivers an **end-to-end, production-grade machine learning service** for **batch electricity demand forecasting**. We designed, implemented, and deployed a complete ML pipeline that automates data collection, preprocessing, model training, inference, and monitoring.  

The service helps **Public Service Company(PNM)** optimize electricity supply by predicting hourly energy demand, reducing costs, and improving operational efficiency.  

## Business Problem  
PNM struggles with managing electricity **supply and demand**, leading to:  
- **Overproduction** → Wasted energy and higher costs.  
- **Shortages** → Customer dissatisfaction and service disruptions.  
- **Inefficient resource allocation** → Poor financial and operational performance.  

### Solution  
We built a **machine learning-based demand forecasting system** that enables PNM to:  
✅ Reduce **overproduction** by aligning energy generation with expected demand.  
✅ Prevent **shortages** by predicting peak-hour demand accurately.  
✅ Optimize **resource allocation** through data-driven energy distribution.  
✅ Improve **operational efficiency** by integrating automation in forecasting.  

## **End-to-End Machine Learning Pipeline**  
We developed a **three-stage ML pipeline** using modern MLOps practices, including feature stores, model registries, and automated inference.  

### **1️⃣ Feature Pipeline**  
📌 **Fetch & process data from multiple sources:**  
- **Electricity demand data** → U.S. Energy Information Administration (EIA) API.  https://www.eia.gov/opendata/
- **Weather data** → Open-Meteo API (historical temperature).  https://open-meteo.com/
- **Holiday data** → Pandas U.S. Federal Holiday Calendar.  pandas library pandas.tseries.holiday

📌 **Data processing steps:**  
- Merge datasets based on timestamps.  
- Transform data into a **time-series format**.  
- Store processed features in a **Feature Store**.  

### **2️⃣ Model Training Pipeline**  
📌 **Fetch features from Feature Store and train a machine learning model.**  
📌 Model selection:  
- **Baseline model** → **Linear Regression** (for interpretability).  
- **Final model** → **LightGBM** (for better performance on time-series data).  
📌 **Hyperparameter tuning** → Used **Optuna** for automated optimization.  
📌 **Model evaluation metric** → **Mean Absolute Percentage Error (MAPE)** (preferred for forecasting).  
📌 Store the trained model in a **Model Registry**.  

### **3️⃣ Inference Pipeline** (Automated Batch Predictions)  
📌 **Runs hourly as a scheduled GitHub Action.**  
📌 Fetches:  
- Latest features from the **Feature Store**.  
- Latest model from the **Model Registry**.  
📌 Generates **hourly electricity demand predictions**.  
📌 Compares predictions with actual values using **Mean Absolute Error (MAE)**.  
📌 Deploys the service as a **serverless function** in `inference_pipeline.py`.  

## **Model Monitoring & Dashboard**  
📌 Built a **Streamlit dashboard** for real-time model monitoring.  https://electricitydemandmonitor-2pe99wef2a5mdvtjnjqt4d.streamlit.app/
📌 Displays:  
- **Actual vs. Predicted Demand** (as time-series plots).  
- **Model Performance Metric** (MAE).  

📌 Helps track **model drift** and maintain prediction accuracy over time.  

## **Final Deployment (User Interface)**  
📌 Developed a **Streamlit web application** to allow users to:  
- View **batch demand predictions**.  
- Monitor **forecast accuracy**.  
- Analyze **past trends and seasonal effects**.  

## **Key Achievements** 🎯  
✅ **End-to-end ML service:** From data ingestion to deployment.  
✅ **Production-level MLOps:** Feature store, model registry, automated training & inference.  
✅ **Automated batch predictions:** Runs hourly via GitHub Actions.  
✅ **Scalable & explainable model:** LightGBM with feature engineering.  
✅ **Real-time monitoring:** Interactive dashboard for performance tracking.  
✅ **Deployed application:** Accessible via a **Streamlit web app**.  https://electricitydemandpredictor-3gaww4pzqsw6orh3vnkc4f.streamlit.app/



## **Technologies Used** 🛠️  
- **Machine Learning**: LightGBM, Optuna (Hyperparameter tuning)  
- **MLOps**: Feature Store, Model Registry  
- **Data Engineering**: Pandas, Open-Meteo API, EIA API  
- **Automation**: GitHub Actions (for scheduled inference)  
- **Deployment**: Streamlit (for monitoring and web app)

  
