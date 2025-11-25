from ohsome import OhsomeClient
import geopandas as gpd
import pandas as pd
import numpy as np
import logging


def prepare_city_data(geometries, index):
    # Define city name and print log
    city_name = f"{index}_{geometries.iloc[index]['eu_city'].strip().lower()}"
    logging.info(f"Retrieving ohsome data for {city_name}")

    # Create bbox
    city_geometry = geometries.geometry.iloc[index].buffer(0)
    city_gdf = gpd.GeoDataFrame(
        gpd.GeoSeries(city_geometry), columns=["geometry"]
    ).set_crs(epsg=4326, inplace=True)

    # Create df for city
    city_df = geometries.iloc[index].drop(["geometry", "area_sqm"], errors="ignore")

    return city_name, city_gdf, city_df


# -- AMENITIES --
def get_amenities(geometries, index):
    city_name, city_gdf, city_df = prepare_city_data(geometries, index)
    client = OhsomeClient()

    # Define tags directly in a format suitable for the query
    tags = {
        "highway": [
            "traffic_signals",
            "stop",
            "bus_stop",
            "traffic_calming",
        ],
        "amenity": [
            "bicycle_parking",
            "bicycle_rental",
            "cyclist_waiting_aid",
            "compressed_air",
            "bicycle_repair_station",
        ],
        "railway": ["subway_entrance", "tram_stop"],
    }

    # Build the filter string from the tags dictionary
    filter_parts = []
    for key, values in tags.items():
        filter_parts.extend(f"{key}={value}" for value in values)

    filter_string = "type:node and (" + " or ".join(filter_parts) + ")"

    # Perform the query using the ohsome API
    try:
        response = client.elements.geometry.post(
            bpolys=city_gdf,
            filter=filter_string,
            time="2019-12-31",
            properties="tags",
            clipGeometry=True,
        )

        response_gdf = response.as_dataframe()
        response_df = pd.DataFrame(response_gdf).drop(columns="geometry")

        # Expand the @other_tags dictionaries into separate columns
        tags_df = response_df["@other_tags"].apply(pd.Series)

        # Join the expanded tags back to the original GeoDataFrame
        amenities_df = response_df.join(tags_df).drop(columns="@other_tags")

    except Exception as e:
        logging.error(f"An error occurred for {city_name}: {str(e)}")

        return city_df

    # HIGHWAY AMENITIES
    for highway_tag in tags["highway"]:
        # Check if 'highway' column exists and if tag is present, then count
        city_df[f"{highway_tag}_count"] = (
            (amenities_df["highway"] == highway_tag).sum()
            if "highway" in amenities_df.columns
            else 0
        )

    # PUBLIC TRANSPORT AMENITIES
    if "railway" in amenities_df.columns:
        # Subway entrances
        city_df["subway_entrance_count"] = amenities_df[
            amenities_df["railway"] == "subway_entrance"
        ].shape[0]

        # Tram stops
        city_df["tram_stop_count"] = amenities_df[
            amenities_df["railway"] == "tram_stop"
        ].shape[0]
    else:
        city_df["subway_entrance_count"] = 0
        city_df["tram_stop_count"] = 0

    # BIKE AMENITIES
    # Initialize counts and capacities to 0
    for tag in tags["amenity"]:
        city_df[f"{tag}_count"] = 0

    # Check for the presence of 'amenity' column
    if "amenity" in amenities_df.columns:
        # Count occurrences of each amenity tag
        for amenity_tag in tags["amenity"]:
            city_df[f"{amenity_tag}_count"] = (
                amenities_df["amenity"] == amenity_tag
            ).sum()

    return city_df


# -- LANDUSE --
def get_landuse(geometries, index):
    city_name, city_gdf, city_df = prepare_city_data(geometries, index)
    client = OhsomeClient()

    filter_string_green = (
        "geometry:point and natural=tree or "
        "geometry:line and (waterway=river or waterway=stream or waterway=drain or waterway=canal) or "
        "geometry:polygon and ("
        "natural in (grassland,heath,scrub,wood,bay,beach,water,wetland) or "
        "leisure in (disc_golf_course,dog_park,garden,nature_reserve,park,pitch,swimming_pool,water_park) or "
        "sport in (baseball,dog_training,free_flying,multi,soccer,rugby_union,ultimate_frisbee,swimming) or "
        "landuse in (allotments,Basin,farmland,flowerbed,forest,grass,greenfield,greenhouse_horticulture,meadow,orchard,plant_nursery,recreation_ground,village_green,vineyard) or "
        "barrier=hedge"
        ")"
    )

    filter_string_urban = (
        "geometry:polygon and ("
        "landuse in (commercial, education, religious) or "
        "shop=* or "
        "office in (educational_institution, religion, research, university) or "
        "building in (church, civic, college, commercial, public, retail, school, supermarket, synagogue, toilets, university, sports_centre, sports_hall, station, train_station) or "
        "amenity in (arts_centre, childcare, kindergarten, college, community_centre, library, place_of_worship, public_building, research_institute, school, social_facility, theatre, townhall, university, clinic, doctors, fire_station, grave_yard, hospital, police, retirement_home, bank, bar, biergarten, brothel, cafe, canteen, car_rental, car_wash, cinema, club, driving_school, fast_food, fuel, marketplace, pub, restaurant, vehicle_inspection, veterinary) or "
        "leisure in (club, common, dance, fitness_centre, fitness_station, horse_riding, mini_golf, piste, playground, school_yard, sports_centre, sports_hall, stadium, track)"
        ")"
    )

    try:
        # Query for urban space
        response_urban = client.elements.area.post(
            bpolys=city_gdf, filter=filter_string_urban, time="2019-12-31"
        )
        response_urban_df = response_urban.as_dataframe()
        city_df["urban_space_sqkm"] = (
            response_urban_df["value"].iloc[0] / 1e6 if not response_urban_df.empty else 0
        )

        # Query for green space
        response_green = client.elements.area.post(
            bpolys=city_gdf, filter=filter_string_green, time="2019-12-31"
        )
        response_green_df = response_green.as_dataframe()
        city_df["green_space_sqkm"] = (
            response_green_df["value"].iloc[0] / 1e6 if not response_green_df.empty else 0
        )

    except Exception as e:
        logging.error(
            f"An error occurred while retrieving area data for {city_name}: {str(e)}"
        )
        city_df["urban_space_sqkm"] = np.nan
        city_df["green_space_sqkm"] = np.nan

    return city_df
