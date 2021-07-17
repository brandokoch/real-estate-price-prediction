import streamlit as st
import pandas as pd
import pandas as pd
from joblib import load
from treeinterpreter import treeinterpreter as ti
from utils import *

#
# Config
# WARNING: This app is optimized for the random forest model
#
MODEL_PATH = "models/rf_best/rf_best.bin"
PREPROCESSOR_PATH = "models/rf_best/preprocessor_v2.bin"
DATA_PATH = "data/processed/zillow_Cleveland_clean.pkl"  # Used for visualization of nearby listings

#
# Model, Preprocessor and Data Loading
#
@st.cache
def load_data(data_path):
    df = pd.read_pickle(data_path)
    df = df[["lng", "lat", "price", "floorsize_m2"]]
    df["coordinates"] = df.apply(
        lambda row: get_polygon_coordinates(row["lng"], row["lat"]), axis=1
    )
    df["fill_color"] = df["price"].apply(
        lambda value: [
            translate_range(val) for val in cmap(int(translate_price(value)))
        ]
    )
    return df


preprocess_pipeline = load(PREPROCESSOR_PATH)
reg = load(MODEL_PATH)
df = load_data(DATA_PATH).copy()

#
# SideBar Input Fields
#
st.sidebar.subheader("Listing Info")
lat = st.sidebar.number_input("Latitude", value=41.463827)
lng = st.sidebar.number_input("Longitude", value=-81.708920)
floorsize_m2 = st.sidebar.number_input("Floor Size (m2)", step=int(), value=155)
lot_size_m2 = st.sidebar.number_input("Lot Size (m2)", step=int(), value=500)
year_built = st.sidebar.number_input(
    "Year built", min_value=1900, step=int(), value=1950
)
listing_type = st.sidebar.selectbox(
    "Type",
    ("Apartment", "Condo", "MultiFamily", "SingleFamily", "Townhouse"),
)
bedroom_cnt = st.sidebar.number_input(
    "Number of Bedrooms", min_value=1, step=int(), value=1
)
full_bathroom_cnt = st.sidebar.number_input(
    "Number of Full Bathrooms", min_value=1, step=int(), value=1
)
partial_bathroom_cnt = st.sidebar.number_input(
    "Number of Partial Bathrooms", min_value=0, step=int(), value=int()
)
infer_button = st.sidebar.button("Get Price")

#
# Main Page and Infer Loop
#
st.title("Cleveland Real Estate Price Prediction")

if infer_button:

    # Preparing Input
    input_dict = {
        "lat": [lat],
        "lng": [lng],
        "type": [listing_type],
        "year_built": [year_built],
        "bedroom_cnt": [bedroom_cnt],
        "full_bathroom_cnt": [full_bathroom_cnt],
        "partial_bathroom_cnt": [partial_bathroom_cnt],
        "floorsize_m2": [floorsize_m2],
        "lot_size_m2": [lot_size_m2],
    }
    input_df = pd.DataFrame.from_dict(input_dict)
    input_df = generate_features(input_df)
    input_df = input_df[
        [
            "lat",
            "lng",
            "type",
            "year_built",
            "bedroom_cnt",
            "full_bathroom_cnt",
            "partial_bathroom_cnt",
            "floorsize_m2",
            "floorsize_m2_per_bedroom",
            "lot_size_m2",
            "sale_year",
            "sale_month",
            "sale_day",
        ]
    ]

    # Price Prediction
    input_prepared = preprocess_pipeline.transform(input_df.head(1))
    pred, bias, contrib = ti.predict(reg, input_prepared)

    st.subheader("Prediction")
    st.markdown(f"Price Estimate: **${pred[0][0]:,.0f}**")

    # Adding our prediction to df for visualization
    pred_df = input_df.copy()
    pred_df["price"] = [round(pred[0][0])]
    pred_df["coordinates"] = [get_polygon_coordinates(lng, lat)]
    pred_df["fill_color"] = [[0.0, 0.0, 0.0, 255.0]]
    pred_df = pred_df[
        ["price", "coordinates", "fill_color", "floorsize_m2", "lng", "lat"]
    ]
    df_concat = pd.concat([df, pred_df]).reset_index(drop=True)

    # Feature Importance
    fi = rf_feat_importance(reg, cols)
    st.subheader("Feature Importances")
    st.markdown(
    """
    *Shows what are the most useful features when estimating price.*
    """
    )
    st.pyplot(plot_fi(fi).get_figure())

    # Plot Feature Contributions
    st.subheader("Feature Contributions")
    st.markdown(
    """
    *Shows which features influenced the price the most.*
    """
    )
    st.markdown(f"Bias: **${bias[0]:,.0f}**")
    fig, ax = waterfall_plot(
        cols,
        contrib[0],
        threshold=0.08,
        rotation_value=80,
        formatting="{:,.3f}",
        figsize=(12, 6),
    )
    st.pyplot(fig)

    # Plot Map Visualization
    st.subheader("Map Visualization")
    st.markdown(
        f"""
    *Shows nearby sold properties where color reflects price and height reflects floor size. For the ease of use the predicted listing is colored black.*
    #### Controls:
    - *Left Mouse* to **MOVE VIEW**
    - *Scroll Wheel* to **ZOOM**
    - *Right Mouse* to **ROTATE VIEW**
    - *Hover Mouse* to **SHOW INFO**
    """
    )
    st.pydeck_chart(get_pydeck_viz(df_concat, lat, lng), use_container_width=True)

else:
    st.text("")
    st.text("")
    st.subheader("Choose listing parameters and click the **Get Price** button.")
