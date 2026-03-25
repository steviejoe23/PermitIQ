"""
PostGIS database connection for parcel lookups.
Replaces in-memory GeoJSON loading (140MB per worker → ~0 memory).

Connection string configurable via DATABASE_URL env var.
Default: postgresql://postgres:permitiq123@localhost/permitiq
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# psycopg2 is optional — if not installed, all functions return None/empty
try:
    import psycopg2
    from psycopg2 import pool as _psycopg2_pool
    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    logger.info("psycopg2 not installed — PostGIS features disabled")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:permitiq123@localhost/permitiq"
)

_pool = None


def get_pool():
    """Get or create connection pool. Returns None if DB unavailable."""
    global _pool
    if not _HAS_PSYCOPG2:
        return None
    if _pool is not None:
        return _pool
    try:
        _pool = _psycopg2_pool.SimpleConnectionPool(1, 5, DATABASE_URL)
        logger.info("PostGIS connection pool created")
        return _pool
    except Exception as e:
        logger.warning(f"PostGIS not available, falling back to GeoJSON: {e}")
        return None


def query_parcel(parcel_id: str):
    """Look up a parcel by ID. Returns dict with zoning + GeoJSON geometry."""
    p = get_pool()
    if p is None:
        return None
    conn = p.getconn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT parcel_id, primary_zoning, all_zoning_codes, article, multi_zoning,
                   ST_AsGeoJSON(geom)::json as geometry
            FROM parcels WHERE parcel_id = %s
        """, (parcel_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return {
            "parcel_id": row[0],
            "primary_zoning": row[1],
            "all_zoning_codes": row[2],
            "article": row[3],
            "multi_zoning": row[4],
            "geometry": row[5],
        }
    except Exception as e:
        logger.error(f"Parcel query error: {e}")
        conn.rollback()
        return None
    finally:
        p.putconn(conn)


def query_parcels_nearby(lat: float, lon: float, radius_meters: float = 500, limit: int = 20) -> list:
    """Find parcels within radius of a point. For site selection."""
    p = get_pool()
    if p is None:
        return []
    conn = p.getconn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT parcel_id, primary_zoning, all_zoning_codes,
                   ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as dist_meters,
                   ST_AsGeoJSON(geom)::json as geometry
            FROM parcels
            WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            ORDER BY dist_meters
            LIMIT %s
        """, (lon, lat, lon, lat, radius_meters, limit))
        rows = cur.fetchall()
        cur.close()
        return [{
            "parcel_id": r[0],
            "primary_zoning": r[1],
            "all_zoning_codes": r[2],
            "distance_meters": round(r[3], 1),
            "geometry": r[4],
        } for r in rows]
    except Exception as e:
        logger.error(f"Nearby parcels query error: {e}")
        conn.rollback()
        return []
    finally:
        p.putconn(conn)


def query_parcels_by_zoning(zoning: str, limit: int = 50) -> list:
    """Find parcels by zoning type. For site selection."""
    p = get_pool()
    if p is None:
        return []
    conn = p.getconn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT parcel_id, primary_zoning, all_zoning_codes,
                   ST_AsGeoJSON(ST_Centroid(geom))::json as centroid
            FROM parcels
            WHERE primary_zoning = %s
            LIMIT %s
        """, (zoning, limit))
        rows = cur.fetchall()
        cur.close()
        return [{
            "parcel_id": r[0],
            "primary_zoning": r[1],
            "all_zoning_codes": r[2],
            "centroid": r[3],
        } for r in rows]
    except Exception as e:
        logger.error(f"Zoning query error: {e}")
        conn.rollback()
        return []
    finally:
        p.putconn(conn)


def db_available() -> bool:
    """Check if PostGIS is available."""
    return get_pool() is not None
