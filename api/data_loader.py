import geopandas as gpd
from pathlib import Path
import pandas as pd

# -----------------------
# Load Base Parcel Data
# -----------------------

PARCEL_PATH = Path(__file__).parent.parent / "parcels_2025_clean" / "Parcels_2025.shp"
ZONING_PATH = Path(__file__).parent.parent / "boston_zoning_district_clean" / "Boston_Zoning_Districts.shp"

parcels = gpd.read_file(PARCEL_PATH)
zoning = gpd.read_file(ZONING_PATH)

# Ensure IDs are strings
parcels["MAP_PAR_ID"] = parcels["MAP_PAR_ID"].astype(str)

# Ensure CRS matches
if parcels.crs != zoning.crs:
    zoning = zoning.to_crs(parcels.crs)

# -----------------------
# Spatial Join
# -----------------------

parcels_zoned = gpd.sjoin(
    parcels,
    zoning,
    how="left",
    predicate="intersects"
)

# -----------------------
# Aggregate to one row per parcel
# -----------------------

zoning_agg = (
    parcels_zoned
    .groupby("MAP_PAR_ID")
    .agg({
        "MAPNO": lambda x: sorted({str(v) for v in x if pd.notna(v)}),
        "DISTRICT": lambda x: sorted({str(v) for v in x if pd.notna(v)}),
        "ARTICLE": lambda x: sorted({str(v) for v in x if pd.notna(v)}),
        "VOLUME": lambda x: sorted({str(v) for v in x if pd.notna(v)}),
        "geometry": "first"
    })
    .reset_index()
)

# Convert lists to API-friendly strings
zoning_agg["MAPNO_STR"] = zoning_agg["MAPNO"].apply(lambda x: ", ".join(x))
zoning_agg["DISTRICT_STR"] = zoning_agg["DISTRICT"].apply(lambda x: ", ".join(x))
zoning_agg["ARTICLE_STR"] = zoning_agg["ARTICLE"].apply(lambda x: ", ".join(x))
zoning_agg["VOLUME_STR"] = zoning_agg["VOLUME"].apply(lambda x: ", ".join(x))

# Flags
zoning_agg["multi_zoning"] = zoning_agg["MAPNO"].apply(lambda x: len(x) > 1)
zoning_agg["zoning_count"] = zoning_agg["MAPNO"].apply(len)

# Primary zoning
zoning_agg["primary_zoning"] = zoning_agg["MAPNO_STR"].apply(
    lambda x: x.split(", ")[0] if x else None
)

# User summary
zoning_agg["summary"] = zoning_agg.apply(
    lambda r: (
        f"Parcel is governed by {r['MAPNO_STR']} "
        f"under Article {r['ARTICLE_STR']} "
        f"({r['VOLUME_STR']})."
        + (" Multiple zoning districts apply." if r["multi_zoning"] else "")
    ),
    axis=1
)

# Final GeoDataFrame
final_parcels = zoning_agg.set_geometry("geometry")
