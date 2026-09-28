"""Zip-prefix centroids and great-circle distance.

``raw.geolocation`` holds 1,000,163 rows for only 19,015 distinct zip prefixes — up to
1,146 points for a single prefix — so it has to be reduced to one coordinate per prefix
before it can be joined to an order.

Two measured facts drive the reduction:

**Median, not mean.** 203 prefixes have a mean that sits more than 0.1 degrees from their
median, because the source geocoding is contaminated. The clearest case is prefix ``68275``
(Porto Trombetas, Pará): four points sit correctly at latitude -1.74, and five are geocoded
to **Portugal** at latitude ~42, having matched "Porto". A mean lands in the Atlantic; a
median lands in Pará.

**Bounds filter first.** 31 points fall outside Brazil altogether, the worst at longitude
+121. For prefix ``68275`` the bad points are the *majority*, so the median alone still
returns Portugal (41.15). Dropping out-of-Brazil points first moves it to -1.7435 — a
correction of roughly 4,800 km. Median then handles the remaining within-Brazil noise.

Four prefixes lose every point to the bounds filter, and 157 customer plus 7 seller
prefixes never appear in ``raw.geolocation`` at all, so the state-level fallback is load
bearing rather than defensive.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

FEATURES_SCHEMA = "features"

# Generous bounding box for Brazil, including its Atlantic islands. Deliberately loose:
# the job is to reject geocoding failures on other continents, not to clip coastlines.
LAT_MIN, LAT_MAX = -34.0, 6.0
LNG_MIN, LNG_MAX = -74.0, -28.0

# IUGG mean earth radius. Any value in 6357-6378 changes results by well under the error
# already present in a zip-prefix centroid, but pin it so the number is reproducible.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle distance in kilometres.

    Args:
        lat1: Latitude of the first point, in degrees. Scalar or array.
        lon1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        lon2: Longitude of the second point, in degrees.

    Returns:
        Distance in kilometres, broadcast over the inputs. ``NaN`` propagates, so an order
        with an unresolvable coordinate yields ``NaN`` rather than a plausible-looking
        wrong number.

    The haversine form is used rather than the simpler spherical law of cosines because the
    latter loses precision for short distances, and most orders here are short.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


ZIP_CENTROIDS_SQL = f"""
CREATE TABLE {FEATURES_SCHEMA}.zip_centroids AS
WITH clean AS (
    SELECT geolocation_zip_code_prefix AS zip_code_prefix,
           geolocation_lat  AS lat,
           geolocation_lng  AS lng,
           geolocation_state AS state
    FROM raw.geolocation
    WHERE geolocation_lat BETWEEN :lat_min AND :lat_max
      AND geolocation_lng BETWEEN :lng_min AND :lng_max
)
SELECT zip_code_prefix,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS lat,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY lng) AS lng,
       mode() WITHIN GROUP (ORDER BY state)             AS state,
       count(*)                                         AS n_points
FROM clean
GROUP BY zip_code_prefix
"""

STATE_CENTROIDS_SQL = f"""
CREATE TABLE {FEATURES_SCHEMA}.state_centroids AS
WITH clean AS (
    SELECT geolocation_state AS state,
           geolocation_lat   AS lat,
           geolocation_lng   AS lng
    FROM raw.geolocation
    WHERE geolocation_lat BETWEEN :lat_min AND :lat_max
      AND geolocation_lng BETWEEN :lng_min AND :lng_max
)
SELECT state,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS lat,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY lng) AS lng,
       count(*)                                         AS n_points
FROM clean
GROUP BY state
"""


def build_centroid_tables(conn: Connection) -> dict[str, int]:
    """Rebuild ``features.zip_centroids`` and ``features.state_centroids``.

    Args:
        conn: Open connection inside a transaction. The caller owns the transaction.

    Returns:
        Row count per table.

    Dropped and recreated rather than truncated: these are pure derived tables with no
    identity of their own, so there is nothing to preserve, and ``CREATE TABLE AS`` keeps
    the median logic in one statement instead of splitting it across DDL and DML.
    """
    bounds = {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lng_min": LNG_MIN, "lng_max": LNG_MAX}
    counts: dict[str, int] = {}

    for table, sql, key in (
        ("zip_centroids", ZIP_CENTROIDS_SQL, "zip_code_prefix"),
        ("state_centroids", STATE_CENTROIDS_SQL, "state"),
    ):
        qualified = f"{FEATURES_SCHEMA}.{table}"
        conn.execute(text(f"DROP TABLE IF EXISTS {qualified}"))
        conn.execute(text(sql), bounds)
        conn.execute(text(f"ALTER TABLE {qualified} ADD PRIMARY KEY ({key})"))
        counts[table] = conn.execute(text(f"SELECT count(*) FROM {qualified}")).scalar_one()
        logger.info("%-28s %7d rows", qualified, counts[table])

    dropped = conn.execute(
        text(
            "SELECT count(*) FROM raw.geolocation "
            "WHERE geolocation_lat NOT BETWEEN :lat_min AND :lat_max "
            "   OR geolocation_lng NOT BETWEEN :lng_min AND :lng_max"
        ),
        bounds,
    ).scalar_one()
    logger.info("Dropped %d out-of-Brazil geolocation points before taking medians", dropped)
    return counts
