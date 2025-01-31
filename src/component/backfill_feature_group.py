# this file insert the recent batch of data into the feature store

import os
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd

from src.component.data_info import load_full_data
from src.component.feature_group_config import FEATURE_GROUP_METADATA, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME
from src.feature_store_api import get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()


def get_historical_demand_values() -> pd.DataFrame:
    """
    Download historical demand values for all years from 2024 to the current year.
    
    Returns:
        pd.DataFrame: Combined historical electricity demand data.
    """
    from_year = 2024
    to_year = datetime.now().year
    print(f'Downloading raw data from {from_year} to {to_year}')

    all_data = []  # List to store yearly data

    for year in range(from_year, to_year + 1):
        # Define the start and end dates for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Download data for the whole year
        electricity_demand_one_year = load_full_data(start_date, end_date)
        
        # Append the data if it's not empty
        if not electricity_demand_one_year.empty:
            all_data.append(electricity_demand_one_year)
        else:
            print(f"No data found for the year {year}")

    # Concatenate all yearly data into a single DataFrame
    if all_data:
        electricity_demand = pd.concat(all_data, ignore_index=True)
        print(f"Successfully downloaded and combined data from {from_year} to {to_year}")
        return electricity_demand
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