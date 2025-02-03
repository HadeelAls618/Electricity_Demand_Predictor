import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from comet_ml import Experiment
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.component.feature_group_config import FEATURE_VIEW_METADATA, N_HYPERPARAMETER_SEARCH_TRIALS
import  src.component.feature_group_config as config
from src.component.data_info import transform_ts_data_into_features_and_target,train_test_split
#from src.discord import send_message_to_channel
from src.component.feature_store_api import get_or_create_feature_view
from src.logger import get_logger
from src.component.model_info import get_pipeline
from src.component.model_registry_api import push_model_to_registry
from src.paths import DATA_CACHE_DIR, PARENT_DIR

logger = get_logger()

# load variables from .env file as environment variables
load_dotenv(PARENT_DIR / '.env')


def fetch_features_and_targets_from_store(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    step_size: int,
) -> pd.DataFrame:
    """
    Fetches time-series data from the store, transforms it into features and
    targets and returns it as a pandas DataFrame.
    """
    # get pointer to featurew view
    logger.info('Getting pointer to feature view...')
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)

    # generate training data from the feature view
    logger.info('Generating training data')
    ts_data, _ = feature_view.training_data(
        description='Time-series hourly taxi rides',
    )

    # filter data based on the from_date and to_date expressed\
    # as Unix milliseconds
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    ts_data = ts_data[ts_data['seconds'].between(from_ts, to_ts)]

    # sort by pickup_location_id and pickup_hour in ascending order
    ts_data.sort_values(by=['sub_region_code', 'date'], inplace=True)

    # drop `seconds` column
    ts_data.drop('seconds', axis=1, inplace=True)

    # transform time-series data from the feature store into features and targets
    # for supervised learning
    features, targets = transform_ts_data_into_features_and_target(
        ts_data,
        input_seq_len=config.N_FEATURES,  # one month
        step_size=step_size,
    )

    features_and_target = features.copy()
    features_and_target['target_demand_values_next_hour'] = targets

    return features_and_target


def split_data(
    features_and_target: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train, X_test, y_test = train_test_split(
        features_and_target, cutoff_date, target_column_name='target_demand_values_next_hour'
    )
    logger.info(f'{X_train.shape=}')
    logger.info(f'{y_train.shape=}')
    logger.info(f'{X_test.shape=}')
    logger.info(f'{y_test.shape=}')

    return X_train, y_train, X_test, y_test


def find_best_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: Optional[int] = 10,
) -> dict:
    """"""

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Given a set of hyper-parameters, it trains a model and computes an average
        validation error based on a TimeSeriesSplit
        """
        # pick hyper-parameters
        hyperparams = {
            'metric': 'mae',
            'verbose': -1,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
        }

        tss = KFold(n_splits=3)
        scores = []
        for train_index, val_index in tss.split(X_train):
            # split data for training and validation
            X_train_, X_val_ = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
            y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]

            # train the model
            pipeline = get_pipeline(**hyperparams)
            pipeline.fit(X_train_, y_train_)

            # evaluate the model
            y_pred = pipeline.predict(X_val_)
            mae = mean_absolute_error(y_val_, y_pred)

            scores.append(mae)

        # Return the mean score
        return np.array(scores).mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logger.info(f'{best_params=}')

    return best_params


def load_features_and_target(
    local_path_features_and_target: Optional[Path] = None,
) -> pd.DataFrame:
    if local_path_features_and_target:
        logger.info('Loading features_and_target from local file')
        features_and_target = pd.read_parquet(local_path_features_and_target)
    else:
        logger.info('Fetching features and targets from the feature store')
        from_date = pd.to_datetime(date.today() - timedelta(days=52 * 7))
        to_date = pd.to_datetime(date.today())
        features_and_target = fetch_features_and_targets_from_store(
            from_date, to_date, step_size=24
        )

        # save features_and_target to local file
        try:
            local_file = DATA_CACHE_DIR / 'features_and_target.parquet'
            features_and_target.to_parquet(local_file)
            logger.info(f'Saved features_and_target to local file at {local_file}')
        except:
            logger.info('Could not save features_and_target to local file')
            pass

    # make sure pickup_hour is a datetime column
    features_and_target['date'] = pd.to_datetime(features_and_target['date'])

    return features_and_target


def train(
    local_path_features_and_target: Optional[Path] = None,
) -> None:
    """
    Trains model and pushes it to the model registry if it meets the minimum
    performance threshold.
    """
    logger.info('Start model training...')

    # start Comet ML experiment run
    logger.info('Creating Comet ML experiment')
    experiment = Experiment(
        api_key=os.environ['COMET_ML_API_KEY'],
        workspace=os.environ['COMET_ML_WORKSPACE'],
        project_name=os.environ['COMET_ML_PROJECT_NAME'],
    )

    # load features and targets
    features_and_target = load_features_and_target(local_path_features_and_target)
    experiment.log_dataset_hash(features_and_target)

    # split the data into training and validation sets
    cutoff_date = pd.to_datetime(date.today() - timedelta(days=28), utc=True)
    logger.info(f'Splitting data into training and test sets with {cutoff_date=}')
    X_train, y_train, X_test, y_test = split_data(
        features_and_target, cutoff_date=cutoff_date
    )
    experiment.log_parameters(
        {
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_test_shape': X_test.shape,
            'y_test_shape': y_test.shape,
        }
    )

    # find the best hyperparameters using time-based cross-validation
    logger.info('Finding best hyperparameters...')
    best_hyperparameters = find_best_hyperparameters(
        X_train, y_train, n_trials=N_HYPERPARAMETER_SEARCH_TRIALS
    )
    experiment.log_parameters(best_hyperparameters)
    experiment.log_parameter(
        'N_HYPERPARAMETER_SEARCH_TRIALS', N_HYPERPARAMETER_SEARCH_TRIALS
    )

    # train the model using the best hyperparameters
    logger.info('Training model using the best hyperparameters...')
    pipeline = get_pipeline(**best_hyperparameters)
    pipeline.fit(X_train, y_train)

    # evalute the model on test data
    predictions = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, predictions)
    logger.info(f'{test_mae=:.4f}')
    experiment.log_metric('test_mae', test_mae)

    # push the model to the Hopsworks model registry if it meets the minimum performance threshold
    experiment.log_parameter('MAX_MAE', config.MAX_MAE)
    if test_mae < config.MAX_MAE:
        logger.info('Pushing model to the model registry...')
        model_version = push_model_to_registry(
            pipeline,
            model_name=config.MODEL_NAME,
        )
        logger.info(f'Model version {model_version} pushed to the model registry.')

        # add model version to the experiment in CometML
        experiment.log_parameter('model_version', model_version)

        # send notification on Discord
        # send_message_to_channel(
        #     f'New model pushed to the model registry. {test_mae=:.4f}, {model_version=}'
        # )

    else:
        logger.info(
            'Model did not meet the minimum performance threshold. Skip pushing to the model registry.'
        )


if __name__ == '__main__':
    from fire import Fire

    Fire(train)