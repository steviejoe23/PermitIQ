# main.py Refactoring Plan

**Date:** 2026-03-31
**Current state:** `api/main.py` is 3,500 lines. A single bad regex change in `normalize_address()` broke every endpoint simultaneously because search, prediction, geocoding, and startup all call it.

---

## 1. Current File Anatomy (line ranges approximate)

| Lines     | Section                           | Functions/Endpoints                                                                 |
|-----------|-----------------------------------|-------------------------------------------------------------------------------------|
| 1-38      | Imports + sys.path hacks          | model_classes import, database import                                               |
| 39-137    | App setup + middleware            | `FastAPI()`, CORS, API key auth, `RequestLoggingMiddleware`, rate limiting           |
| 139-331   | Data loading                      | Constants, globals, `safe_float/int/str`, `_format_date`, `_precompute_timeline_stats`, `load_data()` startup |
| 333-620   | Parcel endpoints                  | `GET /parcels/{id}`, `_enrich_parcel_result`, `_haversine_m`, `_build_case_coords`, `GET /parcels/{id}/nearby_cases` |
| 622-697   | Geocoding                         | `GET /geocode`                                                                      |
| 700-881   | Address search                    | `normalize_address`, `_cached_search`, `_do_search`                                 |
| 883-1361  | Zoning analysis                   | `_get_parcel_zoning`, `GET /zoning/districts`, `GET /zoning/{id}`, `POST /zoning/check_compliance` |
| 1363-1443 | Full zoning analysis              | `POST /zoning/full_analysis`                                                        |
| 1449-1607 | Variance analysis                 | `POST /variance_analysis`                                                           |
| 1610-1692 | Search endpoints                  | `GET /search`, `GET /address/{address}/cases`                                       |
| 1695-1767 | Pydantic response models          | `HealthResponse`, `SearchResult`, `PredictionResponse`, etc.                        |
| 1769-2007 | Feature building                  | `ProposalInput`, `VARIANCE_TYPES`, `PROJECT_TYPES`, `build_features()`              |
| 2009-2299 | Prediction helpers                | `_clean_case_address`, `_clean_case_date`, `get_similar_cases`, `_estimate_timeline`, `FEATURE_LABELS` |
| 2302-2483 | Variance history + key factors    | `_get_variance_history`, `_build_key_factors`                                       |
| 2486-2651 | Recommendations                   | `_build_recommendations`                                                            |
| 2653-2946 | Prediction endpoint               | `POST /analyze_proposal` (295 lines), heuristic fallback                            |
| 2949-2977 | Batch prediction                  | `POST /batch_predict`                                                               |
| 2980-3174 | Compare/what-if                   | `POST /compare`                                                                     |
| 3177-3186 | Market intel router include       | `from routes.market_intel import ...`                                               |
| 3189-3341 | Platform endpoints                | `GET /stats`, `GET /autocomplete`, `GET /model_info`, `GET /health`, `GET /data_status` |
| 3344-3479 | Recommendation                    | `GET /recommend`                                                                    |
| 3482-3501 | Global error handler              | `exception_handler`                                                                 |

---

## 2. Shared State (Globals)

These module-level variables are read by almost everything:

| Variable         | Set in          | Read by                                                        |
|------------------|-----------------|----------------------------------------------------------------|
| `gdf`            | `load_data()`   | Parcels, geocode, zoning, prediction, compare, recommend, health |
| `zba_df`         | `load_data()`   | Search, address cases, zoning, variance analysis, prediction, similar cases, stats, compare |
| `model_package`  | `load_data()`   | Prediction, compare, recommend, build_features, model_info, health |
| `parcel_addr_df` | `load_data()`   | Geocode, autocomplete, compliance (lot size lookup), build_case_coords |
| `timeline_stats` | `load_data()`   | `_estimate_timeline()`, market_intel router                    |
| `_case_coords`   | `_build_case_coords()` | `nearby_cases()`                                        |

**Key observation:** Every router module will need access to these. The cleanest approach is a shared `state` module (or dataclass) that `load_data()` populates and routers import.

---

## 3. Shared Utility Functions (used across multiple logical domains)

| Function              | Used by                                              |
|-----------------------|------------------------------------------------------|
| `normalize_address()` | Search, address cases, geocode, load_data (startup precompute), build_case_coords |
| `safe_float/int/str()`| Parcel, search, prediction, zoning, everywhere       |
| `_format_date()`      | Search, address cases                                |
| `_clean_case_date()`  | Nearby cases, similar cases, address cases            |
| `_clean_case_address()`| Similar cases                                       |
| `build_features()`    | Prediction, compare, recommend (200+ lines)          |
| `VARIANCE_TYPES`      | build_features, variance_analysis, market_intel       |
| `PROJECT_TYPES`       | build_features, market_intel                          |
| `FEATURE_LABELS`      | Prediction (SHAP labels)                              |
| `_haversine_m()`      | nearby_cases only (but pure utility)                  |

---

## 4. Proposed Module Structure

```
api/
  main.py                    (~200 lines) — App creation, middleware, startup, error handler, router includes
  state.py                   (~30 lines)  — Shared mutable state: gdf, zba_df, model_package, parcel_addr_df, timeline_stats, _case_coords
  utils.py                   (~120 lines) — normalize_address, safe_float/int/str, _format_date, _clean_case_date, _clean_case_address, _haversine_m
  constants.py               (~100 lines) — VARIANCE_TYPES, PROJECT_TYPES, FEATURE_LABELS, DISCLAIMER text
  models.py                  (expand)     — All Pydantic models: ProposalInput, HealthResponse, SearchResult, PredictionResponse, etc.
  routes/
    __init__.py
    parcels.py               (~300 lines) — GET /parcels/{id}, GET /parcels/{id}/nearby_cases, GET /geocode, _enrich_parcel_result, _build_case_coords
    search.py                (~250 lines) — GET /search, GET /address/{address}/cases, GET /autocomplete, _cached_search, _do_search
    zoning.py                (~500 lines) — GET /zoning/districts, GET /zoning/{id}, POST /zoning/check_compliance, POST /zoning/full_analysis, POST /variance_analysis, _get_parcel_zoning
    prediction.py            (~700 lines) — POST /analyze_proposal, POST /batch_predict, POST /compare, build_features, get_similar_cases, _get_variance_history, _build_key_factors, _build_recommendations, _estimate_timeline
    platform.py              (~200 lines) — GET /stats, GET /health, GET /model_info, GET /data_status
    recommend.py             (~150 lines) — GET /recommend
    market_intel.py          (existing, no change)
  services/
    feature_builder.py       (existing — FEATURE_COLS list, used by training pipeline)
    zoning_code.py           (existing)
    database.py              (existing)
    model_classes.py         (existing)
    data_loader.py           (~180 lines) — Extract load_data(), _precompute_timeline_stats() from main.py
```

---

## 5. What Stays in main.py

After refactoring, main.py should contain only:

1. **App creation** — `FastAPI(...)` with metadata
2. **Middleware** — CORS, `RequestLoggingMiddleware`, rate limiting
3. **API key auth** — `verify_api_key` dependency
4. **Startup event** — calls `data_loader.load_all()` which populates `state.*`
5. **Router includes** — `app.include_router(...)` for each route module
6. **Global error handler**

Target: ~200 lines.

---

## 6. What Goes Where

### `state.py` — Shared Mutable State
```python
# All globals that endpoints read. Populated by data_loader at startup.
gdf = None           # GeoDataFrame (98K parcels)
zba_df = None        # DataFrame (ZBA cases)
model_package = None # dict (model + metadata)
parcel_addr_df = None # DataFrame (address->parcel lookup)
timeline_stats = None # dict (pre-computed timeline)
_case_coords = None  # DataFrame (geocoded cases for nearby search)
```

### `utils.py` — Pure Functions (no state dependency)
- `normalize_address(addr)` -- the regex-heavy function that broke everything
- `safe_float(val, default)`, `safe_int(val, default)`, `safe_str(val, default)`
- `_format_date(val)`
- `_clean_case_date(row)`
- `_clean_case_address(row)`
- `_haversine_m(lat1, lon1, lat2, lon2)`

### `constants.py` — Static Data
- `VARIANCE_TYPES` list
- `PROJECT_TYPES` list
- `FEATURE_LABELS` dict
- `DISCLAIMER` string (the long legal disclaimer, currently duplicated in 2 places)

### `services/data_loader.py` — Startup Data Loading
- `load_all()` — populates `state.*` globals
- `_precompute_timeline_stats(tracker_path)` (~70 lines)
- `_build_case_coords()` (~65 lines) — moved from main, depends on state.gdf, state.zba_df, state.parcel_addr_df

### `routes/parcels.py`
- `GET /parcels/{parcel_id}` + `_enrich_parcel_result()`
- `GET /parcels/{parcel_id}/nearby_cases`
- `GET /geocode`
- Imports: `state`, `utils`

### `routes/search.py`
- `GET /search`
- `GET /address/{address}/cases`
- `GET /autocomplete`
- `_cached_search()`, `_do_search()`
- Imports: `state`, `utils`

### `routes/zoning.py`
- `_get_parcel_zoning()` (shared helper within this module)
- `GET /zoning/districts`
- `GET /zoning/{parcel_id}`
- `POST /zoning/check_compliance`
- `POST /zoning/full_analysis` (calls zoning + compliance + prediction)
- `POST /variance_analysis`
- Imports: `state`, `utils`, `services.zoning_code`
- **Cross-router dependency:** `full_analysis` calls `prediction.analyze_proposal()` -- will need to import from prediction router or extract the core logic to a service

### `routes/prediction.py`
- `build_features()` (~200 lines)
- `get_similar_cases()`
- `_get_variance_history()`
- `_build_key_factors()`
- `_build_recommendations()`
- `_estimate_timeline()`
- `POST /analyze_proposal`
- `POST /batch_predict`
- `POST /compare`
- Imports: `state`, `utils`, `constants`

### `routes/platform.py`
- `GET /stats`
- `GET /health`
- `GET /model_info`
- `GET /data_status`
- Imports: `state`

### `routes/recommend.py`
- `GET /recommend`
- Imports: `state`, `prediction.build_features`

---

## 7. Order of Operations

Extract in dependency order (leaf modules first, to avoid circular imports):

1. **`state.py`** — Create first. Just global variables, no imports.
2. **`utils.py`** — Pure functions, no dependency on state. This is the highest-value extraction because `normalize_address` is the function that caused the cascading failure.
3. **`constants.py`** — Static data, no dependencies.
4. **`models.py`** (expand) — Pydantic models, depends only on stdlib.
5. **`services/data_loader.py`** — Move `load_data()` and `_precompute_timeline_stats()`. Depends on `state`, `utils`.
6. **`routes/platform.py`** — Simplest endpoints, fewest dependencies. Good first router to extract.
7. **`routes/search.py`** — Depends on `state`, `utils` only.
8. **`routes/parcels.py`** — Depends on `state`, `utils`.
9. **`routes/prediction.py`** — Large but self-contained. Depends on `state`, `utils`, `constants`.
10. **`routes/zoning.py`** — Depends on `state`, `utils`, `services.zoning_code`, and calls `prediction.analyze_proposal` from `full_analysis`.
11. **`routes/recommend.py`** — Depends on `prediction.build_features`.

**After each step:** Run the test suite (`make test` -- 75 tests) before proceeding to the next extraction.

---

## 8. Dependency Graph

```
main.py
  ├── state.py (shared globals)
  ├── utils.py (pure functions)
  ├── constants.py (static data)
  ├── models.py (Pydantic schemas)
  ├── services/data_loader.py → state, utils
  ├── routes/platform.py → state
  ├── routes/search.py → state, utils
  ├── routes/parcels.py → state, utils
  ├── routes/prediction.py → state, utils, constants
  ├── routes/zoning.py → state, utils, services.zoning_code, routes.prediction (for full_analysis)
  ├── routes/recommend.py → state, routes.prediction.build_features
  └── routes/market_intel.py → (already extracted, injected via init())
```

**Circular import risk:** `routes/zoning.py` calling `routes/prediction.py::analyze_proposal()`. Solutions:
- Option A: Move the core prediction logic to a service (`services/prediction_engine.py`) that both routers import
- Option B: Have `full_analysis` in zoning.py call the prediction endpoint via internal function import (lazy import inside the endpoint function body)
- Option C (simplest): Keep `full_analysis` in prediction.py since it's really a prediction endpoint with zoning context

**Recommended:** Option C. Move `POST /zoning/full_analysis` to `routes/prediction.py` since its primary purpose is prediction. Tag it `["Zoning Analysis", "Prediction"]` so it appears in both Swagger groups.

---

## 9. Risk Assessment

### High Risk
- **`normalize_address()` is called at startup** during `load_data()` to precompute `_addr_norm` columns. If the import path changes and `load_data()` can't find it, the app won't start.
- **`build_features()` depends on `model_package` and `zba_df`** (global state). Must ensure `state.model_package` is accessible from `routes/prediction.py`.
- **Pickle deserialization of `model_classes`** requires the `sys.path` hack at line 22-25. This must stay in main.py or move to `state.py`/`data_loader.py` before `joblib.load()`.
- **`routes/market_intel.py` injection pattern** (`init(zba_df, ...)`) works differently from other routers. Keep this pattern or migrate it to use `state.*` directly (breaking change to that module).

### Medium Risk
- **`_cached_search` uses `@lru_cache`** on a module-level function. Moving it to a router module means the cache lives on the router module, which is fine, but the cache key is based on `q_norm` string -- no risk of stale state references.
- **`recommend_parcels` has its own `_recommend_cache` dict** with TTL. Must move with the endpoint.
- **Import order in main.py matters** -- the market_intel router is included (line 3181) and then `market_init` is called in `load_data()` (line 326). After refactoring, ensure router inclusion happens before startup event fires.

### Low Risk
- **Pydantic models** (`ProposalInput`, response models) are only used by prediction endpoints. Straightforward to move.
- **`data_loader.py` already exists** but is an old, unused file (loads shapefiles directly -- obsolete). Will need to rename or replace it.
- **`models.py` already exists** with an old `ProposalRequest` class (unused by main.py). Will need to merge.

---

## 10. Code Smells and Issues Found

### Dead/Obsolete Code
1. **`api/data_loader.py`** (78 lines) — Loads shapefiles directly via geopandas spatial join. Completely unused by main.py (which loads GeoJSON instead). Safe to delete or rename.
2. **`api/schema.py`** (15 lines) — JSON schema validator, not imported by main.py. Unused.
3. **`api/config.py`** (11 lines) — Pydantic `Settings` class. Not imported by main.py. Unused.
4. **`api/zoning_rules.py`** — Not imported by main.py (uses `services/zoning_code.py` instead). Likely dead.
5. **`api/models.py`** (12 lines) — Defines `ProposalRequest` which is never imported. main.py has its own `ProposalInput`.
6. **`api/boston_parcel_zoning.schema.json`** — Used only by dead `schema.py`. Unused.

### Redundancies
7. **`VARIANCE_TYPES` and `PROJECT_TYPES` are defined in 3 places:** `main.py` (lines 1791-1801), `services/feature_builder.py` (lines 97-107), and passed to `market_intel.init()`. Should be single source of truth in `constants.py`.
8. **`build_features()` exists in both `main.py` (200 lines, full implementation) and `services/feature_builder.py` (just the FEATURE_COLS list, no actual function).** The name `feature_builder.py` implies it builds features, but it only defines the column list. Confusing.
9. **Disclaimer text** is duplicated in 3 places: `analyze_proposal` ML path (line 2866), heuristic fallback (line 2945), and `recommend_parcels` (line 3469). Should be a constant.
10. **Ward auto-detection from district** is duplicated in `analyze_proposal` (lines 2720-2725) and `compare_scenarios` (lines 3014-3019). Same 6 lines copy-pasted.

### Code Smells
11. **`analyze_proposal` accepts `dict` instead of `ProposalInput`** — It defines a Pydantic model but then accepts raw dict and manually falls back (lines 2680-2703). The Pydantic model should be the parameter type.
12. **`compare_scenarios` doesn't use `ProposalInput` at all** — It manually extracts every field from the dict.
13. **`zoning_compliance_check` accepts `dict`** with extensive manual field aliasing (lines 1104-1115). Should use a Pydantic model.
14. **`full_zoning_analysis` calls `analyze_proposal` as a regular function** (line 1398), not via HTTP. This works but creates a tight coupling between zoning and prediction code.
15. **`_precompute_timeline_stats`** is defined before `load_data()` but uses no shared state -- it's a pure function that takes a path. Good candidate for `services/data_loader.py`.
16. **`_build_case_coords`** reads from 3 globals and writes to 1 global. It's called once at startup. Should be in the data loading module.
17. **`import time` appears on line 102** (outside the normal import block at the top). Also `import datetime` on line 3298, `import json` on line 3329, `import shap` on line 2760, and `import time as _time` on line 3370. Scattered lazy imports.
18. **`_rate_buckets`** (in-memory rate limiter, line 108) will not work correctly with multiple workers (uvicorn with >1 worker). Not a refactoring issue, but worth noting.
19. **The `except Exception:` on Pydantic validation** (line 2682) silently swallows all errors and falls through. This masks bad input.

### Naming Issues
20. **`_get_variance_history`** and the `POST /variance_analysis` endpoint do very similar things (query variance combo rates). The endpoint is ~160 lines; the helper is ~80 lines. They compute overlapping statistics with different return formats. Could be unified.

---

## 11. Migration Checklist

For each module extraction:

- [ ] Create the new file with proper imports
- [ ] Use `from api.state import gdf, zba_df, ...` (or `from . import state`)
- [ ] Create an `APIRouter()` with appropriate `tags` and `prefix`
- [ ] Move functions + endpoints to the new module
- [ ] Update main.py to `include_router()`
- [ ] Remove moved code from main.py
- [ ] Run `make test` (75 tests must pass)
- [ ] Manually test: `curl localhost:8000/health`, `curl localhost:8000/search?q=tremont`
- [ ] Check Swagger docs at `/docs` still show all endpoints

---

## 12. Estimated Effort

| Step | Module | Complexity | Estimated Time |
|------|--------|------------|----------------|
| 1 | state.py + utils.py + constants.py | Low | 30 min |
| 2 | models.py (expand) | Low | 15 min |
| 3 | services/data_loader.py | Medium | 30 min |
| 4 | routes/platform.py | Low | 15 min |
| 5 | routes/search.py | Low | 20 min |
| 6 | routes/parcels.py | Medium | 30 min |
| 7 | routes/prediction.py | High | 45 min |
| 8 | routes/zoning.py | Medium | 30 min |
| 9 | routes/recommend.py | Low | 15 min |
| 10 | Cleanup dead files + test | Low | 20 min |
| **Total** | | | **~4 hours** |

---

## 13. Success Criteria

After the refactor:
- `main.py` is under 250 lines
- No module is over 700 lines
- `normalize_address` lives in `utils.py` -- a change there cannot break prediction or zoning
- All 75 existing tests pass
- Swagger docs at `/docs` show all endpoints with correct tags
- `make run` starts API + frontend without errors
- Each router can be tested in isolation by mocking `state.*`
