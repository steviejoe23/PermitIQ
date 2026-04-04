"""
Shared mutable state for PermitIQ API.

All globals that endpoints read. Populated by data_loader at startup.
Routers import these directly: `from api.state import gdf, zba_df, ...`
"""

# GeoDataFrame — 98K Boston parcels with zoning data
gdf = None

# DataFrame — ZBA cases (cleaned, feature-enriched)
zba_df = None

# dict — trained ML model package (model + feature_cols + metadata)
model_package = None

# DataFrame — address->parcel lookup (175K property records)
parcel_addr_df = None

# dict — pre-computed timeline stats from ZBA tracker
timeline_stats = None

# DataFrame — geocoded ZBA cases for geographic nearby search
_case_coords = None

# SHAP TreeExplainer — cached at model load to avoid per-request overhead
shap_explainer = None
