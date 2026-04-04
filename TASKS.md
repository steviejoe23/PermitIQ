# PermitIQ — Task List

## In Progress
- [ ] Fix attorney/market_intel router import paths (lines 153, 162 in main.py — `routes.` → `api.routes.`) **[5 min, HIGH IMPACT]**

## Up Next
- [ ] Decide product direction: startup launch vs portfolio project vs keep building
- [ ] Create public README.md for GitHub (not CLAUDE.md — a proper public-facing README)
- [ ] Record 2-min demo video: 57 Centre Street end-to-end workflow
- [ ] Normalize decision labels in dataset (AppProv/Approved/GRANTED → single canonical form)
- [ ] Investigate 31% missing variance data (OCR issue or genuinely no variances?)

## Backlog — Technical
- [ ] Download parcel polygon boundaries from Boston GIS (enables lot frontage auto-detect + real map shapes)
- [ ] Load test: concurrent predictions, large batch jobs
- [ ] Expand to Cambridge/Somerville (prove model generalizes beyond Boston)
- [ ] Add test coverage for market_intel + attorney routers after import fix
- [ ] Audit 50 OCR cases manually vs original PDFs
- [ ] PostGIS optimization — profile queries for large result sets

## Backlog — Product / Business
- [ ] Landing page at permitiq.com (value prop, demo videos, waitlist)
- [ ] Define customer segments and pricing tiers
- [ ] Talk to 10 zoning attorneys — get real-world feedback
- [ ] API documentation for third-party integrations
- [ ] Boston TAM analysis + expansion strategy for venture scale

## Done (Session 9)
- [x] Split main.py from 3,550 → 200 lines (12 modules)
- [x] Auto-detect parcel-level zoning issues from public records
- [x] Separate parcel-level vs proposal-level variances in compliance response
- [x] Enrich search results with parcel_id (range address matching)
- [x] Frontend: parcel issues expander, compliance breakdown, auto-load parcel
- [x] 14/14 compliance tests pass, 46/46 zoning/search/prediction tests pass
