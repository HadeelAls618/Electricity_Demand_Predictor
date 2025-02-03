# this file insert the recent batch of data into the feature store

import os
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

from src.component.data_info import load_full_data
from src.component.feature_group_config import FEATURE_GROUP_METADATA, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME
from src.component.feature_store_api import get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()


def get_historical_demand_values() -> pd.DataFrame:
    """
    Download historical demand values for all years from 2024 to the current year.
    
    Returns:
        pd.DataFrame: Combined historical electricity demand data.
    """
    now = datetime.now()
    start=datetime(2023,1,1)
    end = datetime(now.year,now.month, now.day)

    print(f'Downloading raw data from {start} to {end}')
        
        # Download data for the whole year
    electricity_demand_year = load_full_data(start, end)
        
        # Append the data if it's not empty
    if not electricity_demand_year.empty:
        return electricity_demand_year
    else:
        print("No historical data available.")
        return pd.DataFrame()



def run():

    logger.info('Fetching raw data from data warehouse')
    electricity_demand_data = get_historical_demand_values()

    # add new column with the timestamp in Unix seconds
    electricity_demand_data['date'] = pd.to_datetime(electricity_demand_data['date'], utc=True)    
    electricity_demand_data['seconds'] = electricity_demand_data['date'].astype(int) // 10**6 # Unix milliseconds

    # get a pointer to the feature group we wanna write to
    feature_group = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # start a job to insert the data into the feature group
    logger.info('Inserting data into feature group...')
    feature_group.insert(electricity_demand_data, write_options={"wait_for_job": False})

if __name__ == '__main__':
    run()