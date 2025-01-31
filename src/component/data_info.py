

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import requests_cache
from retry_requests import retry
import openmeteo_requests
from src.paths import *
from typing import Optional
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tqdm
from datetime import datetime
from typing import Tuple



def download_one_electricity_raw_data(year: int, month: int, day: int) -> pd.DataFrame:
    """
    Fetch raw electricity demand data from the EIA API for the specified date.

    Parameters:
        year (int): Year of the data.
        month (int): Month of the data.
        day (int): Day of the data.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    # Ensure correct date formatting
    start_date = datetime(year, month, day)
    end_date = start_date + timedelta(days=1)

    # API URL and parameters
    url = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "api_key": "dqRq8VpXSoyUrCbrPhuYFxGl6Rul9kmVcRshZ98c"
    }

    try:
        # Make GET request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Parse JSON response
        data = response.json()

        # Save JSON response to a file
        file_path = RAW_DATA_electricity_DIR / f"hourly_demand_{year}-{month:02d}-{day:02d}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Data successfully fetched and saved to {file_path}")

        # Convert the response data to a DataFrame
        if 'response' in data and 'data' in data['response']:
            return pd.DataFrame(data['response']['data'])
        elif 'data' in data:  # Adjust according to the actual structure
            return pd.DataFrame(data['data'])
        else:
            print("Unexpected data structure in API response.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()



# Define a directory to store the raw data filesd

# def load_daily_electricity_data(start_date,end_date) -> pd.DataFrame:
#     """
   
#     """
#     #start = datetime.strptime(start_date, "%Y-%m-%d")
#     #end = datetime.strptime(end_date, "%Y-%m-%d")
#     start_date=pd.to_datetime(start_date, utc=True)
#     end_date=pd.to_datetime(end_date, utc=True)
    
#     all_data = []  # Use a list to collect DataFrames for efficient concatenation
#     label_encoder = LabelEncoder()  # Initialize Label Encoder

#     current_date = start_date
#     while start_date <= end_date:
#         year = current_date.year
#         month = current_date.month
#         day = current_date.day

#         # Define file path for the current day's data
#         local_file = RAW_DATA_electricity_DIR / f"hourly_demand_{year}-{month:02d}-{day:02d}.json"

#         if local_file.exists():
#             print(f"Loading file {local_file}")
#             with open(local_file, "r") as f:
#                 data = json.load(f)
#             if 'response' in data and 'data' in data['response']:
#                 day_data = pd.DataFrame(data['response']['data'])
#             else:
#                 print(f"Unexpected structure in {local_file}")
#                 current_date += timedelta(days=1)
#                 continue
#         else:
#             print(f"File {local_file} not found. Fetching from API...")
#             day_data = download_one_electricity_raw_data(year, month, day)
#             if day_data.empty:
#                 current_date += timedelta(days=1)
#                 continue

#         # Preprocessing
#         if 'subba' in day_data:  # Ensure the column exists
#             day_data['sub_region_code'] = label_encoder.fit_transform(day_data['subba'])
#             day_data['sub_region_code'] = day_data['sub_region_code'].astype('int64')

#         else:
#             print("'subba' column not found in data. Skipping encoding for this day.")
#             day_data['sub_region_code'] = None

#         # Select and rename columns
#         day_data = day_data[['period', 'sub_region_code', 'value']]
#         day_data.rename(columns={
#             'value': 'demand',
#             'period': 'date',
#         }, inplace=True)

#         # Convert 'date' to datetime format
#         day_data['date'] = pd.to_datetime(day_data['date'], utc=True)
#         #day_data['date'] = day_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        

#         # Add the processed DataFrame to the list
#         all_data.append(day_data)

#         current_date += timedelta(days=1)

#     # Concatenate all daily DataFrames
#     if all_data:
#         combined_data = pd.concat(all_data, ignore_index=True)
#         print(f"Successfully loaded and processed data for {start_date} to {end_date}")
#         return combined_data
#     else:
#         print("No data found for the specified date range.")
#         return pd.DataFrame()


def load_daily_electricity_data(start_date, end_date) -> pd.DataFrame:

    start_date = pd.to_datetime(start_date, utc=True)
    end_date = pd.to_datetime(end_date, utc=True)

    all_data = []
    label_encoder = LabelEncoder()

    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day

        local_file = RAW_DATA_electricity_DIR / f"hourly_demand_{year}-{month:02d}-{day:02d}.json"

        if local_file.exists():
            print(f"Loading file {local_file}")
            with open(local_file, "r") as f:
                data = json.load(f)
            if 'response' in data and 'data' in data['response']:
                day_data = pd.DataFrame(data['response']['data'])
            else:
                print(f"Unexpected structure in {local_file}")
                current_date += timedelta(days=1)
                continue
        else:
            print(f"File {local_file} not found. Fetching from API...")
            day_data = download_one_electricity_raw_data(year, month, day)
            if day_data.empty:
                current_date += timedelta(days=1)
                continue

      
        day_data['sub_region_code'] = label_encoder.fit_transform(day_data['subba'])
        day_data['sub_region_code'] = day_data['sub_region_code'].astype('int64')

        required_columns = ['period', 'sub_region_code', 'value']
        missing_columns = [col for col in required_columns if col not in day_data.columns]
        if missing_columns:
            print(f"Missing columns {missing_columns} in data for {current_date}. Skipping this day.")
            current_date += timedelta(days=1)
            continue

        day_data = day_data[['period', 'sub_region_code', 'value']]
        day_data.rename(columns={'value': 'demand', 'period': 'date'}, inplace=True)
        day_data['date'] = pd.to_datetime(day_data['date'], utc=True)

        all_data.append(day_data)

        current_date += timedelta(days=1)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Successfully loaded and processed data for {start_date} to {end_date}")
        return combined_data
    else:
        print("No data found for the specified date range.")
        return pd.DataFrame()


# def download_and_load_weather_data(start_date, end_date):
#     """
#     """
#     # Setup the Open-Meteo API client with cache and retry on error
#     cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)

#     # API URL and parameters
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     params = {
#         "latitude": 52.52,  # Example coordinates for Berlin, Germany
#         "longitude": 13.41,
#         "start_date": start_date,
#         "end_date": end_date,
#         "hourly": ["temperature_2m"],
#         "timeformat": "unixtime",
#         "timezone": "America/New_York"
#     }

#     try:
#         # Fetch data from the Open-Meteo API
#         response = openmeteo.weather_api(url, params=params)[0]  # Assuming single location


#         # Extract hourly data
#         hourly = response.Hourly()
#         hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
#         hourly_data = {
#             "date": pd.date_range(
#                 start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#                 end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#                 freq=pd.Timedelta(seconds=hourly.Interval()),
#                 inclusive="left"
#             )}
        
#         hourly_data["temperature_2m"] = hourly_temperature_2m

#         # Convert to DataFrame and process timestamps
#         hourly_dataframe = pd.DataFrame(data = hourly_data)

#         hourly_dataframe["date"] = pd.to_datetime(hourly_dataframe["date"])
#        # hourly_dataframe['date']=hourly_dataframe['date'].dt.strftime('%Y-%m-%d %H:%M:%S')


#         # Save to file
#         file_path = RAW_DATA_weather_DIR / f"weather_data_{start_date}_to_{end_date}.csv"
#         hourly_dataframe.to_csv(file_path, index=False)
#         print(f"Weather data saved to {file_path}")

#         return hourly_dataframe

#     except Exception as e:
#         print(f"Error downloading weather data: {e}")
#         return pd.DataFrame()
    



# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below

def download_and_load_weather_data(start_date, end_date):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": start_date.strftime("%Y-%m-%d"),
	"end_date": end_date.strftime("%Y-%m-%d"),
	"hourly": ["temperature_2m", "weather_code"],
	"timeformat": "unixtime",
	"timezone": "America/New_York"}
    try:
        
        responses = openmeteo.weather_api(url, params=params)

   # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
		    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		    end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		    freq = pd.Timedelta(seconds = hourly.Interval()),
		    inclusive = "left"
	    )}

        hourly_data["temperature_2m"] = hourly_temperature_2m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
    # Save to file
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        file_path = RAW_DATA_weather_DIR / f"weather_data_{start_date_str}_to_{end_date_str}.csv"
        hourly_dataframe.to_csv(file_path, index=False)
        print(f"Weather data saved to {file_path}")

        return hourly_dataframe
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return pd.DataFrame()



def load_full_data(start_date, end_date):
    df1=load_daily_electricity_data(start_date, end_date)
    df2=download_and_load_weather_data(start_date, end_date)
    full_date=pd.merge(df1,df2, on="date", how="inner")
    return full_date

def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int):
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'date','demand', 'sub_region_code','temperature_2m'}

    region_codes = ts_data['sub_region_code'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for code in tqdm.tqdm(region_codes):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.sub_region_code == code, 
            ['date', 'temperature_2m','demand']
        ].sort_values(by=['date'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float64)
        y = np.ndarray(shape=(n_examples), dtype=np.float64)
        date_hours = []
        temperatures = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['demand'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['demand'].values[0]
            date_hours.append(ts_data_one_location.iloc[idx[1]]['date'])
            temperatures.append(ts_data_one_location.iloc[idx[1]]['temperature_2m'])


        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['date'] = date_hours
        features_one_location['sub_region_code'] = code
        features_one_location['temperature_2m'] = temperatures

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_demand_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_demand_next_hour']


#feature engineering on the merged data


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """ """
    df['date']=pd.to_datetime(df['date'], utc=True)
    train_data = df[df.date < cutoff_date].reset_index(drop=True)
    test_data = df[df.date >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test


def fetch_demand_values_from_data_warehouse(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e. 1 year),
    adjusted to use a load_full_data function that takes full start_date and end_date as input.
    """
    # Calculate historical date range (1 year back)
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching demand values from {from_date} to {to_date}')

    # Load data for the historical range using start_date and end_date
    demand_values = load_full_data(from_date_, to_date_)

    # Ensure the data is within the range (optional redundant check)
    demand_values = demand_values[(demand_values.date >= from_date_) & (demand_values.date < to_date_)]

    # Shift the data to pretend this is recent data
    demand_values['date'] += timedelta(days=7*52)

    # Sort the data by location and datetime for consistency
    demand_values.sort_values(by=['sub_region_code', 'date'], inplace=True)

    return demand_values

