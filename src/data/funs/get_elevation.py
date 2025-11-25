import logging
import numpy as np
from shapely.geometry import Point
import requests

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_open_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["results"][0]["elevation"]
    except Exception as e:
        logging.error(f"Error with Open Elevation API: {e}")
    return None


def get_meteo_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["elevation"]
    except Exception as e:
        logging.error(f"Error with Meteo Elevation API: {e}")
    return None


def get_elevation(geometries, index):
    city = geometries.iloc[index]
    city_name = f"{index}_{city['eu_city'].strip().lower()}"
    logging.info(f"Retrieving elevation data for {city_name}")

    city_df = city.drop(["geometry", "area_sqm"], errors="ignore").copy()

    minx, miny, maxx, maxy = city.geometry.bounds
    grid_spacing = 0.022  # approximately 2500m
    x = np.arange(minx, maxx, grid_spacing)
    y = np.arange(miny, maxy, grid_spacing)
    points = [
        Point(xi, yi) for xi in x for yi in y if city.geometry.contains(Point(xi, yi))
    ]

    # Try Open Elevation API
    elevations = [get_open_elevation(p.y, p.x) for p in points]
    valid_elevations = [e for e in elevations if e is not None and e != 0]

    if valid_elevations:
        city_df["elevation_std"] = np.std(valid_elevations)
        city_df["elevation_n"] = len(valid_elevations)
        city_df["elevation_avg"] = np.mean(valid_elevations)
        city_df["elevation_source"] = "open_elevation"
        logging.info(
            f"Successfully retrieved elevation data for {city_name} from Open Elevation"
        )
        return city_df

    logging.warning(
        f"Open Elevation API failed for {city_name}. Trying Meteo Elevation API."
    )

    # If Open Elevation API fails, try Meteo Elevation API
    elevations = [get_meteo_elevation(p.y, p.x) for p in points]
    valid_elevations = [e for e in elevations if e is not None and e != 0]

    if valid_elevations:
        city_df["elevation_std"] = np.std(valid_elevations)
        city_df["elevation_avg"] = np.mean(valid_elevations)
        city_df["elevation_n"] = len(valid_elevations)
        city_df["elevation_source"] = "meteo_elevation"
        logging.info(
            f"Successfully retrieved elevation data for {city_name} from Meteo Elevation"
        )
        return city_df

    logging.error(f"Both APIs failed for {city_name}")
    return city_df
