from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_electricity_DIR = PARENT_DIR / 'data' / 'raw'/'electricity_raw_data'
RAW_DATA_weather_DIR = PARENT_DIR / 'data' / 'raw'/'weather_raw_data'
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'
DATA_CACHE_DIR = PARENT_DIR / 'data' / 'cache'

MODELS_DIR = PARENT_DIR / 'models'
Encoder_DIR = PARENT_DIR / 'encoder'
if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_electricity_DIR).exists():
    os.mkdir(RAW_DATA_electricity_DIR)

if not Path(RAW_DATA_weather_DIR).exists():
    os.mkdir(RAW_DATA_weather_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(Encoder_DIR).exists():
    os.mkdir(Encoder_DIR)

if not Path(DATA_CACHE_DIR).exists():
    os.mkdir(DATA_CACHE_DIR)