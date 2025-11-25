import pandas as pd
import eurostat


# -- DOWNLOAD DATA --
eu_datasets = {
    code: eurostat.get_data_df(code)
    for code in [
        "urb_cpop1",
        "urb_cpopstr",
        "urb_cpopcb",
        "urb_cfermor",
        "urb_clivcon",
        "urb_ceduc",
        "urb_ctour",
        "urb_clma",
        "urb_cecfi",
        "urb_ctran",
        "urb_cenv",
    ]
}


# -- PROCESS DATA --
all_data = pd.concat([
    df.melt(
        id_vars=["freq", "indic_ur", "cities\\TIME_PERIOD"],
        var_name="TIME_PERIOD",
        value_name="value",
    ).assign(dataset=code)
    for code, df in eu_datasets.items()
]).rename(columns={"cities\\TIME_PERIOD": "cities"})

# Filter relevant years and keep only most recent value
all_data_filtered = (
    all_data.query("TIME_PERIOD >= '2011' and TIME_PERIOD <= '2019'")
    .dropna(subset=["value"])
    .sort_values("TIME_PERIOD", ascending=False)
    .groupby(["indic_ur", "cities"])
    .first()
    .reset_index()
)


# -- CREATE BIKE MODE SHARE DATA --
# Filter for bike data and rename cols for consistency
mode_share = all_data_filtered.query("indic_ur == 'TT1007V'")
mode_share = mode_share.rename(
    columns={"cities": "eu_city_code", "TIME_PERIOD": "year", "value": "eu_cycling"}
)
mode_share = mode_share[["eu_city_code", "eu_cycling", "year"]]

# Export to CSV
mode_share.to_csv("data/interim/eu_mode_share.csv", index=False)


# -- CREATE WIDE FORMAT AND EXPORT --
eurostat_wide = (
    all_data_filtered.pivot(index="cities", columns="indic_ur", values="value")
    .rename_axis(None, axis=1)
    .rename_axis(None)
    .reset_index()
    .rename(columns={"index": "eu_city_code"})
)

# Export to CSV
eurostat_wide.to_csv("data/interim/eurostat.csv", index=False)
