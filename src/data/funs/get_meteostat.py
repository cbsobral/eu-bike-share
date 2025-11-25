import logging
from meteostat import Daily, Stations, Point
from datetime import datetime


def get_weather(geometries, index):
    # Define city name and print log
    city_name = f"{index}_{geometries.iloc[index]['eu_city'].strip().lower()}"
    logging.info(f"Retrieving weather data for {city_name}")

    # Create city_df
    city_df = geometries.iloc[index].drop(["geometry", "area_sqm"], errors="ignore")

    # Create centroid
    city_centroid = geometries.geometry.iloc[index].buffer(0).centroid

    # Define the time period
    start = datetime(2004, 1, 1)
    end = datetime(2023, 12, 31)

    try:
        # Fetch data using location
        location = Point(city_centroid.y, city_centroid.x)
        response = Daily(location, start, end).fetch()

        if response.empty:
            logging.info(
                f"No direct weather data found for {city_name}. Searching nearby stations."
            )
            stations = Stations()
            # Get weather stations up to 90km
            nearby_stations = stations.nearby(
                lat=city_centroid.y, lon=city_centroid.x, radius=90000
            ).fetch()

            # Try the first station with a non-NaN ICAO code
            valid_stations = nearby_stations[nearby_stations["icao"].notna()]
            if not valid_stations.empty:
                station = valid_stations.iloc[0].name
                response = Daily(station, start, end).fetch()

            # If still no data, log and return city_df with no additional data
            if response.empty:
                logging.error(
                    f"No weather data found from nearby stations for {city_name}"
                )
                return city_df

    except Exception as e:
        logging.error(f"An error occurred for {city_name}: {str(e)}")
        return city_df  # Return city_df with no data

    # Group by month-year then month
    response_month_year = response.groupby([
        response.index.year,
        response.index.month,
    ]).mean()
    response_month = response_month_year.groupby(level=1).mean()

    # Calculate mean for each column
    city_df["coldest_month_tavg"] = response_month["tavg"].min()
    city_df["hottest_month_tavg"] = response_month["tavg"].max()
    city_df["wind_speed_avg"] = response_month["wspd"].mean()
    city_df["sun_avg"] = response_month["tsun"].mean()
    city_df["snow_avg"] = response_month["snow"].mean()

    # Calculate the number of rainy days per year
    response["year"] = response.index.year
    rainy_days = response[response["prcp"] > 1].groupby("year").size()

    # Calculate the number of days with prcp data per year
    total_days = response[response["prcp"].notna()].groupby("year").size()

    # Calculate proportion of rainy days per year
    prop_rainy_days = (rainy_days / total_days).dropna()

    # Calculate mean values across all years
    city_df["prcp_days_prop"] = prop_rainy_days.mean()

    return city_df
