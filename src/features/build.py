import pandas as pd

# -- CONFIG --
# Load from interim folder
urau = pd.read_csv("data/interim/urau.csv")
attributes = pd.read_csv("data/interim/attributes.csv")
eurostat = pd.read_csv("data/interim/eurostat.csv")

# Load from raw folder
pc = pd.read_csv("data/raw/pc_mode_share.csv")

# Load lookup_cities.xlsx
lookup_cities = pd.read_excel("data/dictionaries/lookup_cities.xlsx")


# -- PC MODE SHARE --
# Merge PC and lookup_cities
pc = pc.merge(
    lookup_cities[["eu_city_code", "pc_city"]],
    left_on="City",
    right_on="pc_city",
    how="left",
)

# Filter NAs
pc = pc.dropna(subset=["eu_city_code"]).reset_index(drop=True)

# Filter for last observation and specific cities in Europe
pc = (
    pc.query("LastObservation == 'YES'")
    .query("continent == 'Europe'")
    .query("City not in ['Nurnberg', 'Freiburg', 'Turin', 'Frankfurt']")
    .reset_index(drop=True)
)

# Keep only necessary columns
pc = pc[["eu_city_code", "Cycling"]]
pc = pc.rename(columns={"Cycling": "pc_cycling"})

# -- MERGE DATA --
# Fix area_sqkm
urau["area_sqkm"] = urau["area_sqm"] / 1e6
urau = urau.drop(columns=["area_sqm"])

# Merge PC and URAU
features = urau.merge(pc, on="eu_city_code", how="left")

# Merge attributes
features = features.merge(attributes, on="eu_city_code", how="left")

# Merge EUROSTAT
features = features.merge(eurostat, on="eu_city_code", how="left")

# Rename EUROSTAT mode share and coalesce with pc_cycling
features["eu_cycling"] = features["TT1007V"] / 100
features = features.drop(columns=["TT1007V"])
features["eu_cycling"] = features["eu_cycling"].combine_first(features["pc_cycling"])

# Convert subway and tram counts to binary (0/1) integers
features["subway_entrance_count"] = (features["subway_entrance_count"] > 0).astype(int)
features["tram_stop_count"] = (features["tram_stop_count"] > 0).astype(int)

# Create country dummies
country_dummies = pd.get_dummies(
    features["eu_country_code"].str.lower(), prefix="country"
).astype(int)

# Add the dummies to the main dataframe
features = pd.concat([features, country_dummies], axis=1)


# -- CLEANUP --
# Drop the original country code column
features = features.drop("eu_country_code", axis=1)

# Reorder columns
id_cols = ["eu_city_code", "eu_city", "pc_cycling", "gei_cycling", "eu_cycling"]
feature_cols = [
    col
    for col in features.columns
    if not col.startswith("country") and col not in id_cols
]
dummy_cols = [col for col in features.columns if col.startswith("country")]

features = features[id_cols + feature_cols + dummy_cols]


# -- SAVE --
features.to_csv("data/interim/features.csv", index=False)
