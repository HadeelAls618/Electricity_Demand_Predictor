from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 

def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    display_title: Optional[bool] = True,
):
    """"""
    features_ = features.iloc[example_id]
    
    if targets is not None:
        target_ = targets.iloc[example_id]
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith('demand_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['date'] - timedelta(hours=len(ts_columns)),
        features_['date'],
        freq='H'
    )
    
    # line plot with past values
    title = f'demand hour r={features_["date"]}, sub regions={features_["sub_region_code"]}' if display_title else None
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )
    
    if targets is not None:
        # green dot for the value we wanna predict
        fig.add_scatter(x=ts_dates[-1:], y=[target_],
                        line_color='green',
                        mode='markers', marker_size=10, name='actual value') 
        
    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers', marker_symbol='x', marker_size=15,
                        name='prediction')             
    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    sub_region_codes: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.sub_region_code.isin(sub_region_codes)] if sub_region_codes else ts_data

    fig = px.line(
        ts_data_to_plot,
        x="date",
        y="demand",
        color='sub_region_code',
        template='none',
    )

    fig.show()