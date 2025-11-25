import osmnx as ox
import pandas as pd
import numpy as np
import logging
import geopandas as gpd


# -- CYCLEWAYS --
def get_cycleways(geometries, index):
    # Define city name and print log
    city_name = f"{index}_{geometries.iloc[index]['eu_city'].strip().lower()}"
    logging.info(f"Retrieving cycleways data for {city_name}")

    # Create bbox
    city_geometry = geometries.geometry.iloc[index]
    city_geometry = city_geometry.buffer(0)

    # Create df for city
    city_df = geometries.iloc[index].drop(["geometry", "area_sqm"], errors="ignore")

    # Configure OSMnx cache,log and date
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 999
    ox.settings.useful_tags_way = [
        "highway",
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
        "cyclestreet",
        "bicycle_road",
        "surface",
        "smoothness",
    ]
    ox.settings.overpass_settings = '[out:json][date:"2019-12-31T00:00:00Z"]'

    try:
        # Retrieve map
        cycleways_g = ox.graph_from_polygon(city_geometry, network_type="bike")

        # To undirected
        cycleways_gu = cycleways_g.to_undirected()

        # Extract edge data
        edge_data = []

        for u, v, key, data in cycleways_gu.edges(data=True, keys=True):
            osmid = data.get("osmid", np.nan)  # Default to nan if not available
            highway = data.get("highway", np.nan)
            cycleway = data.get("cycleway", np.nan)
            cycleway_left = data.get("cycleway:left", np.nan)
            cycleway_right = data.get("cycleway:right", np.nan)
            cycleway_both = data.get("cycleway:both", np.nan)
            cyclestreet = data.get("cyclestreet", np.nan)
            bicycle_road = data.get("bicycle_road", np.nan)
            surface = data.get("surface", np.nan)
            smoothness = data.get("smoothness", np.nan)
            length = data.get("length", np.nan)

            edge_data.append((
                u,
                v,
                key,
                osmid,
                highway,
                cycleway,
                cycleway_left,
                cycleway_right,
                cycleway_both,
                cyclestreet,
                bicycle_road,
                surface,
                smoothness,
                length,
            ))

        cycleways_df = pd.DataFrame(
            edge_data,
            columns=[
                "u",
                "v",
                "key",
                "osmid",
                "highway",
                "cycleway",
                "cycleway:left",
                "cycleway:right",
                "cycleway:both",
                "cyclestreet",
                "bicycle_road",
                "surface",
                "smoothness",
                "length",
            ],
        )

        cycleway_types = [
            "lane",
            "track",
            "opposite_lane",
            "opposite_track",
            "opposite",
        ]

        cycleways_df = cycleways_df[
            (cycleways_df["highway"] == "cycleway")
            | (cycleways_df["cycleway"].isin(cycleway_types))
            | (cycleways_df["cycleway:left"].isin(cycleway_types))
            | (cycleways_df["cycleway:right"].isin(cycleway_types))
            | (cycleways_df["cycleway:both"].isin(cycleway_types))
            | (cycleways_df["cyclestreet"] == "yes")
            | (cycleways_df["bicycle_road"] == "yes")
        ]

    except Exception as e:
        logging.error(f"An error occurred for {city_name}: {str(e)}")
        return city_df  # Return city_df with no data

    # BIKE PATH AND TRACKS
    # Create 'path_type'
    if cycleways_df["cycleway"].notna().any():
        # If 'cycleway' exists, use 'highway' and 'cycleway' to determine 'path_type'
        cycleways_df["path_type"] = np.where(
            (cycleways_df["highway"] == "cycleway")
            | (cycleways_df["cycleway"] == "track"),
            "track",
            "lane",
        )
    else:
        # If 'cycleway' does not exist, use only 'highway' to determine 'path_type'
        cycleways_df["path_type"] = np.where(
            cycleways_df["highway"] == "cycleway",
            "track",
            "lane",  # cyclestreets included here
        )

    # LENGTHS
    # Calculate lengths for different path types
    city_df["bike_path_count"] = cycleways_df["path_type"].value_counts().sum()
    city_df["bike_lane_km"] = (
        cycleways_df[cycleways_df["path_type"] == "lane"]["length"].sum() / 1000
    )
    city_df["bike_track_km"] = (
        cycleways_df[cycleways_df["path_type"] == "track"]["length"].sum() / 1000
    )
    city_df["bike_network_km"] = cycleways_df["length"].sum() / 1000

    # CYCLESTREETS
    # Define conditions as a dictionary for potential columns
    conditions = {"cyclestreet": "yes", "bicycle_road": "yes"}

    # Create a mask using only the columns that exist in cycleways_df
    mask = (
        pd.concat(
            [
                (cycleways_df[col] == val)
                for col, val in conditions.items()
                if col in cycleways_df.columns
            ],
            axis=1,
        ).any(axis=1)
        if any(col in cycleways_df.columns for col in conditions)
        else pd.Series([False] * len(cycleways_df))
    )

    # Compute the total length for cyclestreets if there are any; otherwise, set to 0
    city_df["cyclestreet_km"] = (
        cycleways_df.loc[mask, "length"].sum() / 1000 if mask.any() else 0
    )

    # SURFACE
    # Define mapping dictionary for 'good' surfaces
    if cycleways_df["surface"].notna().any():
        # Convert to string
        cycleways_df["surface"] = cycleways_df["surface"].astype(str)

        surface_map = {
            "asphalt": "good",
            "concrete": "good",
            "concrete:plates": "good",
            "concrete:lanes": "good",
            "paving_stones": "good",
        }

        # Apply the mapping, with 'NA' for any missing keys
        cycleways_df["surface_quality"] = cycleways_df["surface"].map(surface_map)

        cycleways_df["surface_quality"] = np.where(
            pd.notna(cycleways_df["surface"])
            & pd.isna(
                cycleways_df["surface_quality"]
            ),  # Where surface is known but not in map
            "bad",  # Assign 'bad'
            cycleways_df["surface_quality"],  # Else keep current surface quality
        )

        # Calculate lengths
        good_surface_length = cycleways_df[cycleways_df["surface_quality"] == "good"][
            "length"
        ].sum()
        total_surface_length = cycleways_df[
            cycleways_df["surface_quality"].isin(["good", "bad"])
        ]["length"].sum()

        # Create columns
        city_df["surface_length_km"] = total_surface_length / 1000
        city_df["surface_good_prop"] = (
            good_surface_length / total_surface_length
            if total_surface_length > 0
            else np.nan
        )

    else:
        # Handle case where 'surface' is missing
        city_df["surface_length_km"] = np.nan
        city_df["surface_good_prop"] = np.nan

    # SMOOTHNESS
    if cycleways_df["smoothness"].notna().any():
        # Convert to string
        cycleways_df["smoothness"] = cycleways_df["smoothness"].astype(str)

        smooth_map = {"good": "good", "excellent": "good"}

        # Apply the mapping, with 'NA' for any missing keys
        cycleways_df["smooth_quality"] = cycleways_df["smoothness"].map(smooth_map)

        cycleways_df["smooth_quality"] = np.where(
            pd.notna(cycleways_df["smoothness"])
            & pd.isna(cycleways_df["smooth_quality"]),
            "bad",  # Assign 'bad'
            cycleways_df["smooth_quality"],
        )

        # Calculate lengths
        good_smooth_length = cycleways_df[cycleways_df["smooth_quality"] == "good"][
            "length"
        ].sum()
        total_smooth_length = cycleways_df[
            cycleways_df["smooth_quality"].isin(["good", "bad"])
        ]["length"].sum()

        # Create columns
        city_df["smooth_length_km"] = total_smooth_length / 1000
        city_df["smooth_good_prop"] = (
            good_smooth_length / total_smooth_length
            if total_smooth_length > 0
            else np.nan
        )

    else:
        # Handle case where 'smoothness' is missing
        city_df["smooth_length_km"] = np.nan
        city_df["smooth_good_prop"] = np.nan

    return city_df


# -- GRAPH --
def get_graph_cycleways(geometries, index):
    # Used to visualize the cycleways only -- not in production code
    # Create bbox
    city_geometry = geometries.geometry.iloc[index]
    city_geometry = city_geometry.buffer(0)

    city_gdf = gpd.GeoDataFrame(
        gpd.GeoSeries(city_geometry), columns=["geometry"]
    ).set_crs(epsg=4326, inplace=True)

    # Configure OSMnx cache, log, and date
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 999
    ox.settings.useful_tags_way = [
        "highway",
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
        "cyclestreet",
        "bicycle_road",
        "surface",
        "smoothness",
    ]
    ox.settings.overpass_settings = '[out:json][date:"2019-12-31T00:00:00Z"]'

    cycleways_g = ox.graph_from_polygon(city_geometry, network_type="bike")
    cycleways_gu = cycleways_g.to_undirected()

    # Convert to geodataframes
    nodes, edges = ox.graph_to_gdfs(cycleways_gu)

    # Define a list of cycleway types for reusability
    cycleway_types = ["lane", "track", "opposite_lane", "opposite_track", "opposite"]

    # Initialize a base mask with the 'highway' condition
    filtered_mask = edges["highway"] == "cycleway"

    # Check for the existence of each optional column and update the mask accordingly
    if "cycleway" in edges.columns:
        filtered_mask |= edges["cycleway"].isin(cycleway_types)
    if "cycleway:left" in edges.columns:
        filtered_mask |= edges["cycleway:left"].isin(cycleway_types)
    if "cycleway:right" in edges.columns:
        filtered_mask |= edges["cycleway:right"].isin(cycleway_types)
    if "cycleway:both" in edges.columns:
        filtered_mask |= edges["cycleway:both"].isin(cycleway_types)
    if "cyclestreet" in edges.columns:
        filtered_mask |= edges["cyclestreet"] == "yes"
    if "bicycle_road" in edges.columns:
        filtered_mask |= edges["bicycle_road"] == "yes"

    # Apply the complete mask to filter the edges
    filtered_edges = edges[filtered_mask]

    # Convert filtered edges back to a graph
    filtered_graph = ox.graph_from_gdfs(nodes, filtered_edges)
    ox.stats.edge_length_total(filtered_graph) / 1000

    return filtered_graph, city_gdf


# -- HIGHWAYS --
def get_highways(geometries, index):
    # Define city name and print log
    city_name = f"{index}_{geometries.iloc[index]['eu_city'].strip().lower()}"
    logging.info(f"Retrieving highways data for {city_name}")

    # Create bbox
    city_geometry = geometries.geometry.iloc[index]
    city_geometry = city_geometry.buffer(0)

    # Create df for city
    city_df = geometries.iloc[index].drop("geometry", errors="ignore")

    # Configure OSMnx cache,log and date
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 999
    ox.settings.overpass_settings = '[out:json][date:"2019-12-31T00:00:00Z"]'

    tags = '["highway"~"residential|unclassified|trunk|primary|secondary|tertiary"]'

    try:
        # Fetch data in graph format
        highways_g = ox.graph_from_polygon(
            city_geometry, network_type="drive", custom_filter=tags
        )

        area_sqm = city_df["area_sqm"]

        # Calculate basic stats
        stats = ox.stats.basic_stats(highways_g, area=area_sqm, clean_int_tol=0.15)

        # Store the necessary statistics in the city_df DataFrame using .get() for safer access
        city_df["street_length_avg"] = stats.get("street_length_avg", 0)
        city_df["street_density_sqkm"] = stats.get("street_density_km", 0)
        city_df["intersection_density_sqkm"] = stats.get("intersection_density_km", 0)
        city_df["circuity_avg"] = stats.get("circuity_avg", 0)

        # Convert the graph to an undirected graph
        highways_g = highways_g.to_undirected()

        # Extract edge data
        edge_data = []

        for u, v, key, data in highways_g.edges(data=True, keys=True):
            maxspeed = data.get("maxspeed", np.nan)
            lanes = data.get("lanes", np.nan)
            length = data.get("length", np.nan)

            edge_data.append((u, v, key, maxspeed, lanes, length))

        highways_df = pd.DataFrame(
            edge_data, columns=["u", "v", "key", "maxspeed", "lanes", "length"]
        )

    except Exception as e:
        logging.error(f"An error occurred for {city_name}: {str(e)}")
        return city_df

    # LENGTH
    # Calculate the sum of length and add to column road_network_km
    city_df["road_network_km"] = highways_df["length"].sum() / 1000

    # MAXSPEED

    # Extract the numeric part of maxspeed
    highways_df["maxspeed"] = highways_df["maxspeed"].str.extract(r"(\d+)").astype(float)

    # Convert to numeric and coerce errors
    highways_df["maxspeed"] = pd.to_numeric(highways_df["maxspeed"], errors="coerce")

    # Check the country in city_df and apply conversion if necessary
    if city_df["eu_country"] in ["United Kingdom", "Ireland"]:
        highways_df["maxspeed"] *= 1.60934  # in place conversion to km/h
    else:
        highways_df["maxspeed"] = highways_df["maxspeed"]

    # Apply mapping, with 'NA' for any missing keys
    highways_df["maxspeed_type"] = np.where(
        pd.isna(highways_df["maxspeed"]),
        None,  # always use instead of np.nan if object
        np.where(
            highways_df["maxspeed"] <= 33,  # 33 to include 20 mph for Wales
            "slow",
            "fast",  # Set 'slow' for <=33, 'fast' otherwise
        ),
    )

    # Calculate lengths
    slow_maxspeed_length = highways_df[highways_df["maxspeed_type"] == "slow"][
        "length"
    ].sum()

    total_maxspeed_length = highways_df[
        highways_df["maxspeed_type"].isin(["slow", "fast"])
    ]["length"].sum()

    # Update city DataFrame with maxspeed info
    city_df["maxspeed_length_km"] = total_maxspeed_length / 1000
    city_df["maxspeed_slow_prop"] = (
        slow_maxspeed_length / total_maxspeed_length
        if total_maxspeed_length > 0
        else np.nan
    )

    # LANES
    # Convert to numeric and coerce errors
    highways_df["lanes"] = pd.to_numeric(highways_df["lanes"], errors="coerce")

    # Apply the mapping, with 'NA' for any missing keys
    highways_df["lanes_quantity"] = np.where(
        pd.isna(highways_df["lanes"]),  # First check for NaN values
        None,  # Keep NaN values as NaN
        np.where(
            highways_df["lanes"] == 1,
            "low",  # Assign 'low' for 1 lane
            "high",  # Assign 'high' for > 1
        ),
    )

    # Calculate lengths
    low_lanes_length = highways_df[highways_df["lanes_quantity"] == "low"]["length"].sum()

    total_lanes_length = highways_df[highways_df["lanes_quantity"].isin(["low", "high"])][
        "length"
    ].sum()

    city_df["lanes_length_km"] = total_lanes_length / 1000
    city_df["lanes_low_prop"] = (
        low_lanes_length / total_lanes_length if total_lanes_length > 0 else np.nan
    )

    return city_df
