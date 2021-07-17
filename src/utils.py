from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import datetime


#
# Constants
#
cmap = plt.get_cmap("jet")  # Takes values 1-256

cols = [
    "Latitude",
    "Longitude",
    "Year Built",
    "Bedroom Count",
    "Full Bathroom Count",
    "Partial Bathroom Count",
    "Floorsize m2",
    "Floorsize m2 per Bedroom",
    "Lot Size m2",
    "Sale Year",
    "Sale Month",
    "Sale Day",
    "IsApartment",
    "IsCondo",
    "IsMultiFamily",
    "IsSingleFamily",
    "IsTownhouse",
]

#
# Helper Functions
#
def generate_features(df):
    df["sale_date"] = datetime.datetime.now()
    df["floorsize_m2_per_bedroom"] = (df["floorsize_m2"] / df["bedroom_cnt"]).round()
    # df[df['floorsize_m2_per_bedroom']==float('inf')]['floorsize_m2_per_bedroom']=None
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_day"] = df["sale_date"].dt.day
    df = df.drop("sale_date", axis=1)

    return df


def translate_price(value, leftMin=20000, leftMax=500000, rightMin=0, rightMax=255):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def translate_range(value, leftMin=0, leftMax=1, rightMin=0, rightMax=255):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def get_polygon_coordinates(lng, lat):
    x1, y1 = lng - 0.0001, lat - 0.0001
    x2, y2 = lng + 0.0001, lat - 0.0001
    x3, y3 = lng + 0.0001, lat + 0.0001
    x4, y4 = lng - 0.0001, lat + 0.0001
    return [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]


def plot_fi(fi):
    return fi.plot("Features", "imp", "barh", figsize=(12, 7), legend=False)


def rf_feat_importance(m, cols):
    return pd.DataFrame({"Features": cols, "imp": m.feature_importances_}).sort_values(
        "imp", ascending=False
    )


#
# Feature contribution visualization
#
def waterfall_plot(
    index,
    data,
    Title="",
    x_lab="",
    y_lab="",
    formatting="{:,.1f}",
    green_color="#29EA38",
    red_color="#FB3C62",
    blue_color="#24CAFF",
    sorted_value=False,
    threshold=None,
    other_label="other",
    net_label="net",
    rotation_value=30,
    blank_color=(0, 0, 0, 0),
    figsize=(10, 10),
):
    """
    Given two sequences ordered appropriately, generate a standard waterfall chart.
    Optionally modify the title, axis labels, number formatting, bar colors,
    increment sorting, and thresholding. Thresholding groups lower magnitude changes
    into a combined group to display as a single entity on the chart.
    """

    # Convert data and index to np.array
    index = np.array(index)
    data = np.array(data)

    # Sorted by absolute value
    if sorted_value:
        abs_data = abs(data)
        data_order = np.argsort(abs_data)[::-1]
        data = data[data_order]
        index = index[data_order]

    # Group contributors less than the threshold into 'other'
    if threshold:

        abs_data = abs(data)
        threshold_v = abs_data.max() * threshold

        if threshold_v > abs_data.min():
            index = np.append(index[abs_data >= threshold_v], other_label)
            data = np.append(
                data[abs_data >= threshold_v], sum(data[abs_data < threshold_v])
            )

    changes = {"amount": data}

    # Define format formatter
    def money(x, pos):
        "The two args are the value and tick position"
        return formatting.format(x)

    formatter = FuncFormatter(money)

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_major_formatter(formatter)

    # Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=changes, index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)

    trans["positive"] = trans["amount"] > 0

    # Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc[net_label] = total
    blank.loc[net_label] = total

    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # When plotting the last element, we want to show the full bar,
    # Set the blank to 0
    blank.loc[net_label] = 0

    # define bar colors for net bar
    trans.loc[trans["positive"] > 1, "positive"] = 99
    trans.loc[trans["positive"] < 0, "positive"] = 99
    trans.loc[(trans["positive"] > 0) & (trans["positive"] < 1), "positive"] = 99

    trans["color"] = trans["positive"]

    trans.loc[trans["positive"] == 1, "color"] = green_color
    trans.loc[trans["positive"] == 0, "color"] = red_color
    trans.loc[trans["positive"] == 99, "color"] = blue_color

    my_colors = list(trans.color)

    # Plot and label
    my_plot = plt.bar(range(0, len(trans.index)), blank, width=0.5, color=blank_color)
    plt.bar(
        range(0, len(trans.index)),
        trans.amount,
        width=0.6,
        bottom=blank,
        color=my_colors,
    )

    # axis labels
    plt.xlabel("\n" + x_lab)
    plt.ylabel(y_lab + "\n")

    # Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)

    temp = list(trans.amount)

    # create dynamic chart range
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i - 1]

    trans["temp"] = temp

    plot_max = trans["temp"].max()
    plot_min = trans["temp"].min()

    # Make sure the plot doesn't accidentally focus only on the changes in the data
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0

    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)
    else:
        maxmax = abs(plot_min)

    pos_offset = maxmax / 40

    plot_offset = maxmax / 15  ## needs to me cumulative sum dynamic

    # Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count
        if row["amount"] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row["amount"]
        # Determine if we want a neg or pos offset
        if row["amount"] > 0:
            y += pos_offset * 2
            plt.annotate(
                formatting.format(row["amount"]),
                (loop, y),
                ha="center",
                color="g",
                fontsize=9,
            )
        else:
            y -= pos_offset * 4
            plt.annotate(
                formatting.format(row["amount"]),
                (loop, y),
                ha="center",
                color="r",
                fontsize=9,
            )
        loop += 1

    # Scale up the y-axis so there is room for the labels
    plt.ylim(
        plot_min - round(3.6 * plot_offset, 7), plot_max + round(3.6 * plot_offset, 7)
    )

    # Rotate the labels
    plt.xticks(range(0, len(trans)), trans.index, rotation=rotation_value)

    # Add zero line and title
    plt.axhline(0, color="black", linewidth=0.6, linestyle="dashed")
    plt.title(Title)
    plt.tight_layout()

    return fig, ax


#
# Map visualization
#
def get_pydeck_viz(df, lat, lng):

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        df,
        id="geojson",
        opacity=0.5,
        stroked=False,
        get_polygon="coordinates",
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="floorsize_m2",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    sunlight = {
        "@@type": "_SunLight",
        "timestamp": 1564696800000,  # Date.UTC(2019, 7, 1, 22),
        "color": [255, 255, 255],
        "intensity": 1.0,
        "_shadow": True,
    }
    tooltip = {
        "html": "<b>floorsize:</b> {floorsize_m2} m2 <br /><b>Price:</b> {price} $"
    }

    ambient_light = {
        "@@type": "AmbientLight",
        "color": [255, 255, 255],
        "intensity": 1.0,
    }

    lighting_effect = {
        "@@type": "LightingEffect",
        "shadowColor": [0, 0, 0, 0.5],
        "ambientLight": ambient_light,
        "directionalLights": [sunlight],
    }

    view_state = pdk.ViewState(
        **{"latitude": lat, "longitude": lng, "zoom": 14, "pitch": 50, "bearing": 0}
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lng,
        zoom=14,
        pitch=50,
    )

    pydeck_viz = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        effects=[lighting_effect],
        tooltip=tooltip,
        layers=[polygon_layer],
    )

    return pydeck_viz