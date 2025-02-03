from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import pandas as pd
from hsfs.client.exceptions import RestAPIError
from retry import retry

from src.component.feature_group_config import FEATURE_GROUP_PREDICTIONS_METADATA, MODEL_NAME
from src.component.feature_store_api import get_or_create_feature_group
from src.component.inference import get_model_predictions, load_batch_of_features_from_store
from src.logger import get_logger
from src.component.model_registry_api import get_latest_model_from_registry

logger = get_logger()


@retry(RestAPIError, tries=3, delay=60)
def save_predictions_to_feature_store(predictions: pd.DataFrame) -> None:
    """
    Saves model predictions to the feature store.

    We add retry to this function because sometimes the feature store API fails, because
    of too many concurrent jobs.
    """
    logger.info('Getting pointer to the feature group for model predictions')
    feature_group = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)

    try:
        logger.info('Saving predictions to the feature store')
        feature_group.insert(predictions, write_options={'wait_for_job': False})

    except RestAPIError as e:
        logger.info('Failed to save predictions to the feature store')
        logger.info('Retrying in 60 seconds...')
        raise e


def inference(
    current_date: Optional[pd.Timestamp] = pd.to_datetime(datetime.utcnow()).floor('H'),
) -> None:
    """"""
    logger.info(f'Running inference pipeline for {current_date}')

    logger.info('Loading batch of features from the feature store')
    features = load_batch_of_features_from_store(current_date)

    logger.info('Loading model from the model registry')
    model = get_latest_model_from_registry(model_name=MODEL_NAME, status='Production')

    logger.info('Generating predictions')
    predictions = get_model_predictions(model, features)

    # add `date` and `seconds` columns
    predictions['date'] = current_date
    predictions['seconds'] = predictions['date'].astype(int) // 10**6

    logger.info('Saving predictions to the feature store')
    save_predictions_to_feature_store(predictions)

    logger.info('Inference DONE!')


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
    # use the current datetime
    if args.datetime:
        current_date = pd.to_datetime(args.datetime)
    else:
        current_date = pd.to_datetime(datetime.utcnow()).floor('H')

    logger.info(f'Running feature pipeline for {current_date=}')
    inference(current_date)