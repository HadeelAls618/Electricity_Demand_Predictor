import os

from dotenv import load_dotenv

from src.component.feature_store_api import FeatureGroupConfig, FeatureViewConfig
from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

try:
    HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception(
        'Create an .env file on the project root with the HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY'
    )

# TODO: remove FEATURE_GROUP_NAME and FEATURE_GROUP_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = 'electricity_demand_feature_group2'
FEATURE_GROUP_VERSION = 3
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='electricity_demand_feature_group2',
    version=3,
    description='Feature group with hourly time-series data of historical demand values',
    primary_key=['sub_region_code', 'date'],
    event_time='date',
    online_enabled=True,
)

# TODO: remove FEATURE_VIEW_NAME and FEATURE_VIEW_VERSION, and use FEATURE_VIEW_METADATA instead
FEATURE_VIEW_NAME = 'electricity_demand_feature_view2'
FEATURE_VIEW_VERSION = 3
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='electricity_demand_feature_view2',
    version=3,
    feature_group=FEATURE_GROUP_METADATA,
)

MODEL_NAME = 'electricity_demand_predictor_next_hour'
MODEL_VERSION=2
# added for monitoring purposes
# TODO remove FEATURE_GROUP_MODEL_PREDICTIONS and use FEATURE_GROUP_PREDICTIONS_METADATA instead
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description='Predictions generate by our production model',
    primary_key=['sub_region_code', 'date'],
    event_time='date',
)

# TODO remove FEATURE_VIEW_MODEL_PREDICTIONS and use FEATURE_VIEW_PREDICTIONS_METADATA instead
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=3,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)

MONITORING_FV_NAME = 'monitoring_feature_view5'
MONITORING_FV_VERSION = 5

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# number of iterations we want Optuna to pefrom to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 5

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 300