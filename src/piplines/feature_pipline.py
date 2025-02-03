from argparse import ArgumentParser
from datetime import datetime, timedelta

import pandas as pd

from src.component import feature_group_config as config
from src.component.data_info import (
    fetch_demand_values_from_data_warehouse
)
from src.component.feature_store_api import get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()


def run(date: datetime):
    """_summary_

    Args:
        date (datetime): _description_

    Returns:
        _type_: _description_
    """
    logger.info('Fetching raw data from data warehouse')

    # fetch raw ride events from the datawarehouse for the last 28 days
    # we fetch the last 28 days, instead of the last hour only, to add redundancy
    # to the feature_pipeline. This way, if the pipeline fails for some reason,
    # we can still re-write data for that missing hour in a later run.
    demand_values = fetch_demand_values_from_data_warehouse(
        from_date=(date - timedelta(days=28)), to_date=date
    )

    # transform raw data into time-series data by aggregating rides per
    # pickup location and hour

    # add new column with the timestamp in Unix seconds
    logger.info('Adding column `seconds` with Unix seconds...')
    demand_values['date'] = pd.to_datetime(demand_values['date'], utc=True)
    demand_values['seconds'] = demand_values['date'].astype(int) // 10**6

    # get a pointer to the feature group we wanna write to
    logger.info('Getting pointer to the feature group we wanna save data to')
    feature_group = get_or_create_feature_group(config.FEATURE_GROUP_METADATA)

    # start a job to insert the data into the feature group
    # we wait, to make sure the job is finished before we exit the script, and
    # the inference pipeline can start using the new data
    logger.info('Starting job to insert data into feature group...')
    feature_group.insert(demand_values, write_options={'wait_for_job': False})
    # feature_group.insert(demand_values, write_options={"start_offline_backfill": False})

    logger.info('Finished job to insert data into feature group')

    # logger.info('Sleeping for 5 minutes to make sure the inference pipeline has time to run')
    # import time
    # time.sleep(5*60)


if __name__ == '__main__':
    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--datetime',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS',
    )
    args = parser.parse_args()

    # if args.datetime was provided, use it as the current_date, otherwise
    # use the current datetime in UTC
    if args.datetime:
        current_date = pd.to_datetime(args.datetime)
    else:
        current_date = pd.to_datetime(datetime.utcnow()).floor('H')

    logger.info(f'Running feature pipeline for {current_date=}')
    run(current_date)