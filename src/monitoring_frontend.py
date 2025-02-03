

from datetime import datetime, timedelta
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
import plotly.express as px

from src.component.monitoring import load_predictions_and_actual_values_from_store


st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Monitoring dashboard 🔎')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3


# # @st.cache_data
# def _load_predictions_and_actuals_from_store(
#     from_date: datetime,
#     to_date: datetime
#     ) -> pd.DataFrame:
#     """Wrapped version of src.monitoring.load_predictions_and_actual_values_from_store, so
#     we can add Streamlit caching

#     Args:
#         from_date (datetime): min datetime for which we want predictions and
#         actual values

#         to_date (datetime): max datetime for which we want predictions and
#         actual values
#     """
#     return load_predictions_and_actual_values_from_store(from_date, to_date)

with st.spinner(text="Fetching model predictions and actual values from the store"):
    
    monitoring_df = load_predictions_and_actual_values_from_store(
        from_date=current_date - timedelta(days=14),
        to_date=current_date
    )
    st.sidebar.write('✅ Model predictions and actual values arrived')
    progress_bar.progress(1/N_STEPS)
    print(monitoring_df)
    print("Unique hours in dataset:", monitoring_df['actuals_date'].dt.hour.unique())



with st.spinner(text="Plotting aggregate MAE hour-by-hour"):
    
    st.header('Mean Absolute Error (MAE) hour-by-hour')

    # MAE per pickup_hour
    # https://stackoverflow.com/a/47914634
    mae_per_hour = (
        monitoring_df
        .groupby('actuals_date')
        .apply(lambda g: mean_absolute_error(g['actuals_demand'], g['predicted_demand']))
        .reset_index()
        .rename(columns={0: 'mae'})
        .sort_values(by='actuals_date')
    )

    fig = px.bar(
        mae_per_hour,
        x='actuals_date', y='mae',
        template='plotly_dark',
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(2/N_STEPS)


with st.spinner(text="Plotting MAE hour-by-hour for top locations"):
    
    st.header('Mean Absolute Error (MAE) per location and hour')

    top_locations_by_demand = (
        monitoring_df
        .groupby('actuals_sub_region_code')['actuals_demand']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(6)['actuals_sub_region_code']
    )

    for location_id in top_locations_by_demand:
        
        mae_per_hour = (
            monitoring_df[monitoring_df.actuals_sub_region_code == location_id]
            .groupby('actuals_date')
            .apply(lambda g: mean_absolute_error(g['actuals_demand'], g['predicted_demand']))
            .reset_index()
            .rename(columns={0: 'mae'})
            .sort_values(by='actuals_date')
        )

        fig = px.bar(
            mae_per_hour,
            x='actuals_date', y='mae',
            template='plotly_dark',
        )
        st.subheader(f'{location_id=}')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(3/N_STEPS)