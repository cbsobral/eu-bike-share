import geopandas as gpd
import pandas as pd

# -- SETUP --
# Load reference data
lookup_cities = pd.read_excel("data/dictionaries/lookup_cities.xlsx")

# Define URLs for GeoJSON data
url20_greater_cities = "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2020_4326_GREATER_CITIES.geojson"
url20_cities = "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2020_4326_CITIES.geojson"
url21_cities = "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2021_4326_CITIES.geojson"


# -- PROCESS 2020 UK DATA --
# Load and filter UK data
cities = gpd.read_file(url20_cities)
cities = cities[cities["CNTR_CODE"] == "UK"]

greater_cities = gpd.read_file(url20_greater_cities)
greater_cities = greater_cities[greater_cities["CNTR_CODE"] == "UK"]

# Bind cities and clean column names
urau_UK = pd.concat([cities, greater_cities])
urau_UK.columns = (
    urau_UK.columns.str.lower()
    .str.replace(" ", "_")
    .str.replace("[^a-z0-9_]", "", regex=True)
)

# Shorten URAU code
urau_UK["urau_code_short"] = urau_UK["urau_code"].str[:5]
urau_UK = urau_UK[["urau_code_short", "area_sqm", "geometry"]]

# Group and select largest area
urau_UK = (
    urau_UK.sort_values(["urau_code_short", "area_sqm"], ascending=[True, False])
    .groupby("urau_code_short")
    .first()
    .reset_index()
)

# Transform URAU code for lookup_cities and keep first instance only
lookup_UK = lookup_cities.copy()
lookup_UK = lookup_UK[lookup_UK["eu_country_code"] == "UK"]
lookup_UK = lookup_UK[["ua_code_short", "eu_city_code"]]

# Merge UK cities with lookup_UK
urau_20 = urau_UK.merge(lookup_UK, left_on="urau_code_short", right_on="ua_code_short")

# Rename and drop columns
urau_20 = urau_20[["eu_city_code", "area_sqm", "geometry"]]


# -- PROCESS 2021 DATA --
# Load and clean 2021 data
urau_21 = gpd.read_file(url21_cities)
urau_21.columns = (
    urau_21.columns.str.lower()
    .str.replace(" ", "_")
    .str.replace("[^a-z0-9_]", "", regex=True)
)

# Rename and drop columns
urau_21 = urau_21.rename(columns={"urau_code": "eu_city_code"})
urau_21 = urau_21[["eu_city_code", "area_sqm", "geometry"]]

# Combine 2020 and 2021 data
urau_2021 = pd.concat([urau_21, urau_20])

# Merge with lookup_cities
lookup_cities = lookup_cities[["eu_city_code", "eu_city", "eu_country_code"]]
urau_2021 = urau_2021.merge(lookup_cities, on="eu_city_code")

# Arrange and reorder columns
urau_2021 = urau_2021.sort_values("eu_city_code").reset_index(drop=True)
urau_2021 = urau_2021[
    ["eu_city_code", "eu_city", "eu_country_code", "area_sqm", "geometry"]
]


# -- SAVE OUTPUTS --
urau_2021.to_file("data/interim/urau.geojson", driver="GeoJSON")
urau_2021.drop(columns="geometry").to_csv("data/interim/urau.csv", index=False)
