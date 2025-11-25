import geopandas as gpd
import pandas as pd
from src.data.funs.get_osmnx import get_cycleways, get_highways  # noqa
from src.data.funs.get_meteostat import get_weather  # noqa
from src.data.funs.get_elevation import get_elevation  # noqa
from src.data.funs.get_aqi import get_aqi  # noqa
from src.data.funs.get_ohsome import get_amenities, get_landuse  # noqa


# -- DATA AND FUNCTION --
# Load GeoJSON file
ua_bbox = gpd.read_file("data/interim/urau.geojson")


# Define function to process city data
def process_city_data(ua_bbox, filename):
    all_data = pd.DataFrame()

    # Determine the appropriate get_data function based on filename
    get_data_function = globals().get(f"get_{filename}")

    if get_data_function is None:
        raise ValueError(f"No function found for filename: {filename}")

    # Loop through all rows in ua_bbox
    for index in range(len(ua_bbox)):
        city_info = get_data_function(ua_bbox, index)
        all_data = pd.concat([all_data, pd.DataFrame([city_info])], ignore_index=True)

    # Save the DataFrame to a file
    all_data.to_csv(f"data/interim/{filename}.csv", index=False)

    return all_data


# Define data types
data_types = [
    "highways",
    "cycleways",
    "amenities",
    "landuse",
    "weather",
    "aqi",
    "elevation",
]

# Load existing CSV files
data_frames = {}
for data_type in data_types:
    file_path = f"data/interim/{data_type}.csv"
    data_frames[data_type] = pd.read_csv(file_path)

# Process each type of data
data_frames = {}
for data_type in data_types:
    data_frames[data_type] = process_city_data(ua_bbox, data_type)

# For each dataframe, keep only eu_city_code and non-ID columns
id_columns = ["eu_country_code", "eu_city", "area_sqm"]
cleaned_frames = {}

for name, df in data_frames.items():
    value_cols = [
        col for col in df.columns if col not in id_columns or col == "eu_city_code"
    ]
    cleaned_frames[name] = df[value_cols]

combined_data = cleaned_frames[data_types[0]]
for name in data_types[1:]:
    combined_data = combined_data.merge(
        cleaned_frames[name], on="eu_city_code", how="left"
    )

# Drop elevation_source
combined_data = combined_data.drop("elevation_source", axis=1)


# -- SAVE --
combined_data.to_csv("data/interim/attributes.csv", index=False)
