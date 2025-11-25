import logging
import pandas as pd
import requests

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_aqi(geometries, index):
    # Define city name and print log
    city_name = f"{index}_{geometries.iloc[index]['eu_city'].strip().lower()}"
    logging.info(f"Retrieving air quality data for {city_name}")

    # Create city_df
    city_df = geometries.iloc[index].drop(["geometry", "area_sqm"], errors="ignore")

    # Get the centroid of the city
    city_geometry = geometries.geometry.iloc[index]
    centroid = city_geometry.centroid
    lat, lon = centroid.y, centroid.x

    try:
        # Fetch air quality data
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "european_aqi",
            "start_hour": "2018-09-01T00:00",  # start date
            "end_hour": "2019-08-30T00:00",  # end date
            "timezone": "GMT",
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception("Failed to retrieve air quality data")

    except Exception as e:
        logging.error(f"Error retrieving air quality data for {city_name}: {e}")
        return city_df

    data = response.json()

    # Condition for good air quality (EAQI < 50)
    good_air_quality_threshold = 50

    # Create a DataFrame from the hourly data
    hourly_df = pd.DataFrame(data["hourly"])
    hourly_df["date"] = hourly_df["time"].str[:10]  # Extract date from time

    # Group by date and get the maximum EAQI for each day
    daily_max_aqi = hourly_df.groupby("date")["european_aqi"].max()

    # Calculate the number of days with good air quality
    good_air_days = (daily_max_aqi <= good_air_quality_threshold).sum()
    total_days = daily_max_aqi.count()

    # Calculate the proportion of good air days
    proportion_good_air_days = good_air_days / total_days if total_days > 0 else None

    # Update city_df
    city_df["aqi_good_days"] = good_air_days
    city_df["aqi_total_days"] = total_days
    city_df["aqi_good_prop"] = proportion_good_air_days

    return city_df
