# Hourly Electricity Demand Batch Prediction Service
## Project Overview  
This project presents an end-to-end, production-ready electricity demand forecasting system designed to optimize energy supply and reduce operational costs for Public Service Company (PNM). Leveraging Mlops techniques and historical demand/weather data, this system accurately predicts hourly electricity demand for various locations in NYC. The solution is fully automated, integrating data collection, feature engineering, model training, inference, and continous monitoring into a seamless ML pipeline.
The service helps **Public Service Company (PNM)** optimize electricity supply by predicting hourly energy demand, reducing costs, and improving operational efficiency, acess the app [here](https://electricitydemandpredictor-3gaww4pzqsw6orh3vnkc4f.streamlit.app/) .

#
![GIF](vid.gif)


## Data Sources

To ensure accurate predictions, we use multiple data sources:

- **Historical electricity Demand Data**: Fetched from **[EIA API](https://www.eia.gov/opendata/)**. 
- **Weather Data**: Historical weather information retrieved from **[Openmeteo Weather API](https://open-meteo.com/)**.
- **Calendar Events**: Public holidays extracted using `pandas.tseries.holiday`.
- 
## Project methodalgy
We built a machine learning-based demand forecasting service to help PNM:

**Reduce overproduction** by aligning energy generation with forecasted demand.
- **Prevent shortages** by accurately predicting peak-hour energy needs.
- **Optimize resource** allocation through data-driven energy distribution.
- **Enhance operational** efficiency with automated forecasting.


## **End-to-End Machine Learning Pipeline**  
Our solution features a three-stage ML pipeline, leveraging modern MLOps principles such as feature stores, model registries, and automated inference to ensure scalability and reliability.

### **1. Feature Pipeline**

This pipeline fetches and preprocesses the required data, transforms it into a time-series format, and stores it in the **Hopsworks Feature Store**.

#### **Steps:**

1. Fetch historical electricity demand data from the **EIA API**.  
2. Fetch historical weather data (temperature) from **OpenWeatherMap API**.
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


## Deployment & Monitoring

We developed **two interactive Streamlit applications** for batch demand forecasting and model performance monitoring.

### **batch demand forecasting app**

- A **Streamlit application** visualizes forecasted electricity demand for different locations in NYC. 
- The application features an interactive **map** where the size of circles represents demand at specific locations.
- Users can explore electricity demand variations across locations in real-time,
- you can acess the app [here](https://electricitydemandpredictor-3gaww4pzqsw6orh3vnkc4f.streamlit.app/) 

### **Monitoring Dashboard**
- A separate **Streamlit dashboard** monitors model performance on an hourly basis.
- The dashboard includes:
  - A real-time chart of **MAE trends**
  - Historical performance records
  - Comparison of predicted vs. actual demand
- This allows continuous monitoring and optimization of the model.
- you can acess the dashbored [here](https://electricitydemandmonitor-2pe99wef2a5mdvtjnjqt4d.streamlit.app/) 


## summery
This project successfully implements a batch prediction system for hourly electricity demand forecasting. By leveraging machine learning and data science, PNM can optimize energy production and improve efficiency while minimizing costs, by achiving the following tasks.

✅ **Automated ML Pipeline** → Fully automated process from data collection to inference.\
✅ **Feature Store & Model Registry** → Efficient feature storage & model versioning.\
✅ **batch Demand Forecasting** → Predicts hourly electricity demand across locations.\
✅ **Interactive app** → Provides easy-to-interpret forecasts & model insights.\
✅ **Production-Ready** → Scheduled batch inference, model monitoring and retrainning & scalable architecture.


  ## Contact
If you have any questions or would like to discuss this project further, feel free to reach out!
* [LinkedIn](https://www.linkedin.com/in/hadeel-als-0a23702a6?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app ) 
* [Email](alsadonhadeel@gmail.com) 
