import os
from pathlib import Path
import pickle

import comet_ml
from comet_ml import API
from dotenv import load_dotenv
import hopsworks
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

import src.component.feature_group_config as config
from src.paths import MODELS_DIR, PARENT_DIR
from src.logger import get_logger

logger = get_logger()

# load variables from .env file as environment variables
load_dotenv(PARENT_DIR / '.env')

COMET_ML_API_KEY = os.environ["COMET_ML_API_KEY"]
COMET_ML_WORKSPACE = os.environ["COMET_ML_WORKSPACE"]
COMET_ML_PROJECT_NAME = os.environ['COMET_ML_PROJECT_NAME']


def get_model_registry() -> None:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_model_registry()

def push_model_to_registry(
    model: Pipeline,
    model_name: str,
) -> int:
    """"""
    # save the model to disk
    model_file = MODELS_DIR / 'model.pkl'
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Get the stale experiment from the global context to grab the API key and experiment ID.
    stale_experiment = comet_ml.get_global_experiment()
    
    # Resume the expriment using its API key and experiment ID.
    experiment = comet_ml.ExistingExperiment(
        api_key=stale_experiment.api_key, experiment_key=stale_experiment.id
    )

    # log model as an experiment artifact
    logger.info(f"Starting logging model to Comet ML")
    experiment.log_model(model_name, str(model_file))
    logger.info(f"Finished logging model {model_name}")
    
    # push model to the registry
    logger.info('Pushing model to the registry as "Production"')
    experiment.register_model(model_name, status='Production')

    # end the experiment
    experiment.end()

    # get model version of the latest production model
    return get_latest_model_version(model_name, status='Production')


def get_latest_model_version(model_name: str, status: str) -> str:
    """
    Returns the latest model version from the registry with the given `status`
    """
    # find all model versions from the given `model_name` registry and `status`
    api = API(COMET_ML_API_KEY)
    model_details = api.get_registry_model_details(COMET_ML_WORKSPACE, model_name)['versions']
    model_versions = [md['version'] for md in model_details if md['status'] == status]
    
    # return the latest model version
    return max(model_versions)


def get_latest_model_from_registry(model_name: str, status: str) -> Pipeline:
    """Returns the latest model from the registry"""
    
    # get model version to download
    model_version = get_latest_model_version(model_name, status)

    # download model from registry
    api = API(COMET_ML_API_KEY)
    api.download_registry_model(
        COMET_ML_WORKSPACE,
        registry_name=model_name,
        version=model_version,
        output_path=MODELS_DIR,
        expand=True
    )

    # load model from local file to memory
    with open(MODELS_DIR / 'model.pkl', "rb") as f:
        model = pickle.load(f)

    return model