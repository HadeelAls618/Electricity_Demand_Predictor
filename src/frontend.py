

import zipfile 
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Point

from src.component.inference import (
    load_model_from_registry,
    load_batch_of_features_from_store,
    get_model_predictions
)
from src.component.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)

from paths import DATA_DIR
from plot import plot_one_sample

st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Electricity Demand Prediction ⚡')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

# Define NYISO Zones with approximate centers
nyiso_zones = {
     0: {'name': 'West', 'lat': 42.8864, 'lon': -78.8784},
     1: {'name': 'Genesee', 'lat': 43.1610, 'lon': -77.6109},
     2: {'name': 'Central', 'lat': 43.0481, 'lon': -76.1474},
     3: {'name': 'North', 'lat': 44.6995, 'lon': -73.4525},
     4: {'name': 'Mohawk Valley', 'lat': 43.1009, 'lon': -75.2327},
     5: {'name': 'Capital', 'lat': 42.6526, 'lon': -73.7562},
     6: {'name': 'Hudson Valley', 'lat': 41.7004, 'lon': -73.9210},
     7: {'name': 'Millwood', 'lat': 41.2048, 'lon': -73.8293},
     8: {'name': 'Dunwoodie', 'lat': 40.9142, 'lon': -73.8557},
     9: {'name': 'New York City', 'lat': 40.7128, 'lon': -74.0060},
    10: {'name': 'Long Island', 'lat': 40.7891, 'lon': -73.1350}
}

# Create a GeoDataFrame from the NYISO zones
def create_nyiso_geo_df():
    zones = []
    for zone_id, info in nyiso_zones.items():
        zones.append({
            'zone_id': zone_id,
            'name': info['name'],
            'latitude': info['lat'],
            'longitude': info['lon']
        })
    df = pd.DataFrame(zones)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude)
    )
    return gdf


# with st.spinner(text="Fetching batch of data"):
#     features = load_batch_of_features_from_store(current_date)
#     st.sidebar.write('✅ Data fetched and ready')
#     progress_bar.progress(2/N_STEPS)

# with st.spinner(text="Loading ML model from registry"):
#     model = load_model_from_registry()
#     st.sidebar.write('✅ ML model successfully loaded from registry')
#     progress_bar.progress(3/N_STEPS)

# with st.spinner(text="Computing model predictions"):
#     predictions = get_model_predictions(model, features)
#     st.sidebar.write('✅ Model predictions arrived')
#     progress_bar.progress(4/N_STEPS)



# @st.cache_data
# def _load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
#     """Wrapped version of src.inference.load_batch_of_features_from_store, so
#     we can add Streamlit caching

#     Args:
#         current_date (datetime): _description_
#     """
#     return load_batch_of_features_from_store(current_date)

# @st.cache_data
# def _load_predictions_from_store(
#     from_date: datetime,
#     to_date: datetime
#     ) -> pd.DataFrame:
#     """
#     Wrapped version of src.inference.load_predictions_from_store, so we
#     can add Streamlit caching

#     Args:
#         from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
#         predictions

#         to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
#         predictions

#     Returns:
#         pd.DataFrame: 2 columns: pickup_location_id, predicted_demand
#     """
#     return load_predictions_from_store(from_date, to_date)


with st.spinner(text="Creating NYISO zones data"):
    geo_df = create_nyiso_geo_df()
    st.sidebar.write('✅ NYISO zones data created')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = load_predictions_from_store(
        from_date=current_date - timedelta(hours=1),
        to_date=current_date
    )
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(2/N_STEPS)

# Here we are checking the predictions for the current hour have already been computed
# and are available
# next_hour_predictions_ready = \
#     False if predictions_df[predictions_df.date == current_date].empty else True
# prev_hour_predictions_ready = \
#     False if predictions_df[predictions_df.date == (current_date - timedelta(hours=1))].empty else True

# # breakpoint()

# if next_hour_predictions_ready:
#     # predictions for the current hour are available
#     predictions_df = predictions_df[predictions_df.date == current_date]

# elif prev_hour_predictions_ready:
#     # predictions for current hour are not available, so we use previous hour predictions
#     predictions_df = predictions_df[predictions_df.date == (current_date - timedelta(hours=1))]
#     current_date = current_date - timedelta(hours=1)
#     st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')

# else:
#     raise Exception('Features are not available for the last 2 hours. Is your feature \
#                     pipeline up and running? 🤔')
from datetime import timedelta

# Check if predictions for the current hour are available
next_hour_predictions_ready = not predictions_df[predictions_df.date == current_date].empty
prev_hour_predictions_ready = not predictions_df[predictions_df.date == (current_date - timedelta(hours=1))].empty

if next_hour_predictions_ready:
    # Predictions for the next hour are already available
    predictions_df = predictions_df[predictions_df.date == current_date]

else:
    # Attempt to fetch predictions for the next hour
    with st.spinner(text="Fetching batch of data"):
        features = load_batch_of_features_from_store(current_date)
        # st.sidebar.write('✅ Data fetched and ready')
        # progress_bar.progress(2 / N_STEPS)

    with st.spinner(text="Loading ML model from registry"):
        model = load_model_from_registry()
        # st.sidebar.write('✅ ML model successfully loaded from registry')
        # progress_bar.progress(3 / N_STEPS)

    with st.spinner(text="Computing model predictions"):
        predictions = get_model_predictions(model, features)
        # st.sidebar.write('✅ Model predictions arrived')
        # progress_bar.progress(4 / N_STEPS)

    # Update predictions availability after fetching
    next_hour_predictions_ready = not predictions_df[predictions_df.date == current_date].empty

    if next_hour_predictions_ready:
        predictions_df = predictions_df[predictions_df.date == current_date]

    elif prev_hour_predictions_ready:
        # If next-hour predictions are still unavailable, use previous hour predictions
        predictions_df = predictions_df[predictions_df.date == (current_date - timedelta(hours=1))]
        current_date = current_date - timedelta(hours=1)
        st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')

    else:
        raise Exception('Features are not available for the last 2 hours. Is your feature pipeline up and running? 🤔')


with st.spinner(text="Preparing data to plot"):
    def pseudocolor(val, minval, maxval, startcolor, stopcolor, alpha=300):
        f = float(val - minval) / (maxval - minval)
        rgb = tuple(int(f * (b - a) + a) for a, b in zip(startcolor, stopcolor))
        return rgb + (alpha,)
    # Merge your data
    df = pd.merge(geo_df, predictions_df,
                  right_on='sub_region_code',
                  left_on='zone_id',
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    
    # Scale the radius based on predicted demand
    scaling_factor = 5 # Adjust this factor as needed for your visualization
    df['radius'] = df['predicted_demand'] * scaling_factor
    progress_bar.progress(5 / N_STEPS)

with st.spinner(text="Generating NYISO Zones Map"):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=41.7004,  # Centered around Hudson Valley
        longitude=-73.9210,
        zoom=6,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",         # Use the dynamic radius here
        get_fill_color=[255, 0, 0],   # Use the computed fill colors
        pickable=True
    )

    tooltip = {
        "html": "<b>Zone:</b> [{zone_id}] {name} <br /> <b>Predicted demand:</b> {predicted_demand}"
    }

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)

    progress_bar.progress(4/N_STEPS)

with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Plotting time-series data"):
   
    predictions_df = df

    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 6

    for row_id in row_indices[:n_to_plot]:
        location_id = predictions_df['zone_id'].iloc[row_id]
        location_name = predictions_df['name'].iloc[row_id]
        st.header(f'Zone ID: {location_id} - {location_name}')

        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label="Predicted demand", value=int(prediction))
        
        fig = plot_one_sample(
            example_id=row_id,
            features=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
        
    progress_bar.progress(6/N_STEPS)



