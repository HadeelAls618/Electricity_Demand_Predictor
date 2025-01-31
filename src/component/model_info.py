import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer

import xgboost as xgb
import lightgbm as lgb

def average_demand_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['average_demand_last_4_weeks'] = 0.25*(
    X[f'demand_previous_{7*24}_hour'] + \
    X[f'demand_previous_{2*7*24}_hour'] + \
    X[f'demand_previous_{3*7*24}_hour'] + \
    X[f'demand_previous_{4*7*24}_hour']
    )
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        X_['date']= pd.to_datetime(X_['date'], utc=True)
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['date'].dt.hour
        X_["day_of_week"] = X_['date'].dt.dayofweek
        X_['month'] = X_['date'].dt.month
        X_['is_weekend'] = X_['date'].isin([5, 6]).astype(int)

        holidays = calendar().holidays(start=X_['date'].min(), end=X_['date'].max())
        X_['is_holiday'] = X_['date'].isin(holidays).astype(int)       
        
        return X_.drop(columns=['date'])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_average_demand_last_4_weeks = FunctionTransformer(
        average_demand_last_4_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_demand_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )



def evaluate_model(y_test, y_pred):
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    return f"MAE is {test_mae:.4f} and MAPE is: {test_mape:.4f}"

