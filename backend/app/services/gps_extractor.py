"""GPS EXIF extraction, reverse geocoding, and location grouping utilities."""

import logging
import math
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)

# Rate limiter for Nominatim (max 1 req/sec per usage policy)
_geocode_lock = threading.Lock()
_last_geocode_time = 0.0


def _dms_to_decimal(dms, ref: str) -> float:
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees."""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(image_path: Path) -> Optional[tuple[float, float]]:
    """Extract GPS coordinates from image EXIF data.

    Returns (latitude, longitude) or None if no GPS data.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
                break

        if not gps_info:
            return None

        lat_dms = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lon_dms = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef")

        if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
            return None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)

        # Sanity check
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None

        return (lat, lon)
    except Exception as e:
        logger.debug(f"GPS extraction failed for {image_path}: {e}")
        return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in meters between two GPS coordinates using Haversine formula."""
    R = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@lru_cache(maxsize=512)
def _cached_geocode(grid_lat: float, grid_lon: float) -> Optional[str]:
    """Reverse geocode with ~100m grid deduplication via lru_cache.

    grid_lat/grid_lon are rounded to 3 decimal places (~111m resolution).
    """
    global _last_geocode_time

    try:
        from geopy.geocoders import Nominatim

        # Rate limit: 1 request per second
        with _geocode_lock:
            elapsed = time.time() - _last_geocode_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            _last_geocode_time = time.time()

        geolocator = Nominatim(user_agent="flowpic-video-generator/1.0")
        location = geolocator.reverse(
            f"{grid_lat}, {grid_lon}",
            language="en",
            timeout=10,
        )

        if location and location.raw.get("address"):
            return _format_place_name(location.raw["address"])
        return None

    except Exception as e:
        logger.warning(f"Reverse geocode failed for ({grid_lat}, {grid_lon}): {e}")
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Reverse geocode coordinates to a concise place name.

    Rounds to ~100m grid for lru_cache deduplication.
    """
    grid_lat = round(lat, 3)
    grid_lon = round(lon, 3)
    return _cached_geocode(grid_lat, grid_lon)


def _format_place_name(address: dict) -> str:
    """Format Nominatim address into concise subtitle: max 2 components.

    Priority: neighbourhood/suburb + city, or city + state/country.
    Examples: "Shibuya, Tokyo", "Brooklyn, New York", "Paris, France"
    """
    # Try to get a local area name
    local = (
        address.get("neighbourhood")
        or address.get("suburb")
        or address.get("quarter")
        or address.get("village")
        or address.get("town")
    )

    # Try to get a city/region name
    city = (
        address.get("city")
        or address.get("municipality")
        or address.get("county")
        or address.get("town")
        or address.get("village")
    )

    # Broader region
    region = (
        address.get("state")
        or address.get("province")
        or address.get("country")
    )

    if local and city and local != city:
        return f"{local}, {city}"
    elif city and region:
        return f"{city}, {region}"
    elif local and region:
        return f"{local}, {region}"
    elif city:
        return city
    elif local:
        return local
    elif region:
        return region

    # Last resort: use whatever we have
    for key in ("display_name",):
        if key in address:
            parts = address[key].split(",")
            if len(parts) >= 2:
                return f"{parts[0].strip()}, {parts[1].strip()}"
            return parts[0].strip()

    return "Unknown Location"
