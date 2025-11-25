import pandas as pd
from pathlib import Path


# -- CONFIG --
# Setup paths and configs
Path("data/processed")
methods = ["mean", "median"]
id_cols = ["eu_city_code", "eu_city", "eu_cycling"]

# -- LOAD DATA --
features = pd.read_csv("data/interim/features.csv")


# -- FUNCTION TO NORMALIZE --
def normalize_and_filter(data):
    processing_rules = pd.read_excel("data/dictionaries/processing_rules.xlsx")

    # Filter out excluded variables
    processing_rules = processing_rules[processing_rules["keep"] == "include"]

    # Common variables (only those not excluded)
    common_vars = processing_rules["variable"]

    base_keep_vars = [
        "area_sqkm",
        "DE1001V",
        "EC2020V",  # Total employment
        "eu_city_code",
        "eu_city",
        "eu_cycling",
    ]

    # Get all columns that start with 'country_'
    country_vars = [col for col in data.columns if col.startswith("country_")]

    # Combine both lists
    keep_vars = base_keep_vars + country_vars

    full_set = set(common_vars) | set(keep_vars)

    # Now filter the data
    data = data[[col for col in data.columns if col in full_set]]

    # Normalize data
    norm_rules = processing_rules[~processing_rules["normalize_by"].isin(["ok"])][
        ["variable", "normalize_by"]
    ]

    norm_vars = processing_rules[processing_rules["nor_name"].notna()][
        ["variable", "nor_name"]
    ].drop_duplicates()

    norm_info = norm_rules.merge(
        norm_vars, left_on="normalize_by", right_on="nor_name", how="left"
    ).rename(
        columns={
            "variable_x": "variable_to_normalize",
            "variable_y": "normalization_variable",
        }
    )

    normalized_data = data.copy()
    new_cols = {}
    for _, row in norm_info.iterrows():
        var_norm = row["variable_to_normalize"]
        var_by = row["normalization_variable"]
        new_name = f"{var_norm}_normalized"
        if var_norm in normalized_data.columns and var_by in normalized_data.columns:
            new_cols[new_name] = normalized_data[var_norm] / normalized_data[var_by]
        else:
            print(f"Skipping normalization for {var_norm} as {var_by} not found")
    # Add all new columns at once to avoid fragmentation
    if new_cols:
        normalized_data = pd.concat([normalized_data, pd.DataFrame(new_cols)], axis=1)

    vars_to_drop = set(norm_info["variable_to_normalize"]) | set(
        norm_info["normalization_variable"]
    )
    vars_to_drop = list(vars_to_drop - set(keep_vars))

    cols_to_keep = [c for c in normalized_data.columns if c not in vars_to_drop]
    normalized_wide = normalized_data[cols_to_keep].rename(
        columns=lambda x: x.replace("_normalized", "")
    )

    return normalized_wide


# -- PREPARE DATA AND SAVE --
# Get normalized features once to ensure consistency
normalized = normalize_and_filter(features)
normalized.to_csv("data/processed/features_norm.csv", index=False)
