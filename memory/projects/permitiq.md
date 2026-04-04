# PermitIQ

## Status
Active development — 9 sessions completed (March 24 - April 1, 2026).

## Current State
- **Model:** AUC 0.7987, Honest CV 0.7910, Denial Recall 69.7%
- **API:** 33 endpoints, 8 route modules, fully refactored
- **Frontend:** Polished Streamlit UI with 8 major sections
- **Tests:** 96+ passing (14 compliance, 46 zoning/search/prediction)
- **Data:** 17,676 ZBA cases, 98,510 parcels, 175K property records
- **GitHub:** https://github.com/steviejoe23/PermitIQ (private)

## Key Milestones
1. Sessions 1-2: Core pipeline (OCR → features → ML → API → UI)
2. Session 3: PostGIS, feature leakage fix, GitHub
3. Session 4-5: Model improvements (XGBoost, stacking ensemble, honest CV)
4. Session 6: Data enrichment (denied variance data, BPDA subdistricts)
5. Session 7: Full audit, 119 tests, attorney intelligence
6. Session 8: API refactoring (3,550 → 200 line main.py)
7. Session 9: Auto-detect parcel issues, search enrichment, frontend polish

## Open Question
Product direction undefined — startup launch vs portfolio project vs continued R&D.
