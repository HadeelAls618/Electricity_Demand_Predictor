
from datetime import datetime, timedelta, timezone
from argparse import ArgumentParser

import pandas as pd

import src.component.feature_group_config as config
from src.logger import get_logger
from src.component.feature_group_config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
from src.component.feature_store_api import get_or_create_feature_group, get_feature_store

logger = get_logger()


def load_predictions_and_actual_values_from_store(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Fetches model predictions and actual values from `from_date` to `to_date`
    from the Feature Store and returns a DataFrame.
    
    The DataFrame is built by joining the predictions and actuals feature groups.
    A prefix is applied to columns from the actuals feature group to avoid ambiguity.
    """
    # Ensure datetime objects are timezone-aware (UTC)
    from_date = from_date.astimezone(timezone.utc)
    to_date = to_date.astimezone(timezone.utc)
    print(from_date)
    print(to_date)



    # Retrieve the two feature groups
    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # Build query to join the two feature groups
    query = predictions_fg.select_all() \
        .join(
            actuals_fg.select(['sub_region_code', 'date', 'demand']),
            on=['date', 'sub_region_code'],
            prefix="actuals_"  # columns from actuals_fg will have the prefix 'actual_'
        ) \
        #  .filter(predictions_fg.date >= from_date) \
        #  .filter(predictions_fg.date <= to_date)

    # Create the feature view if it does not exist yet
    feature_store = get_feature_store()
    try:
        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,
            version=config.MONITORING_FV_VERSION,
            query=query
        )
        logger.info("Feature view created successfully.")
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Feature view already existed. Skipping creation.")
        else:
            logger.error("Error creating feature view: %s", e)
            raise

    # Retrieve the feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,
        version=config.MONITORING_FV_VERSION
    )


 
    

    # Fetch data from the feature view with an extended time window
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7)
    )

    # Filter the returned data to the desired time period.
    # This uses the predictions' date column (assumed to be named 'date')
    #monitoring_df = monitoring_df[monitoring_df.date.between(from_date, to_date)]
    latest_fv_data = monitoring_fv.get_batch_data().sort_values(by="date", ascending=False).head(5)
    print("Latest data in Feature View:", latest_fv_data['date'].max())



    return monitoring_df


if __name__ == '__main__':
    # Parse command-line arguments for the date range
    parser = ArgumentParser()
    parser.add_argument(
        '--from_date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS'
    )
    parser.add_argument(
        '--to_date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS'
    )
    args = parser.parse_args()

    # Call the function with the provided date range
    
    monitoring_df = load_predictions_and_actual_values_from_store()
