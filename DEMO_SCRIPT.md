# PermitIQ Wednesday Demo Script

## Pre-Demo Setup (Do This Tuesday Night)

```bash
cd ~/Desktop/Boston\ Zoning\ Project
source zoning-env/bin/activate

# 1. Train the ML model (takes ~2 min)
python3 train_model_v2.py

# 2. Start the API
cd api
uvicorn main:app --reload --port 8000

# 3. In a new terminal, start the frontend
cd ~/Desktop/Boston\ Zoning\ Project/frontend
streamlit run app.py --server.port 8501
```

Verify at: http://localhost:8501

**New features for the demo:**
- Stats dashboard at top (ZBA decisions, parcels, approval rate, wards, features)
- Sidebar has sample addresses and parcel IDs (no need to memorize)
- Health check shows API status at top
- "What-If" comparison — **now model-computed** (shows actual probability changes, not estimates)
- Downloadable analysis report (now includes What-If scenarios)
- Ward Insights section
- Project Details expander (units, stories)
- Confidence badges (HIGH/MEDIUM/LOW with color coding)

---

## Demo Flow (5-10 minutes)

### Opening (30 sec)
"PermitIQ predicts whether the Boston Zoning Board of Appeals will approve your project — before you file. It's trained on every ZBA decision from 2020 through March 2026 — over 7,500 real cases with 69 engineered features each."

---

### Act 1: Address Search (1 min)
**Search: "75 Tremont Street"**
- Shows 14 historical ZBA cases at this address
- 8 approved, 5 denied — 62% approval rate
- "This is real data from actual ZBA hearings. Every address in Boston with ZBA history is searchable."

**Search: "Burbank Street"**
- Shows cases across Burbank St — 44% approval rate
- "Tougher neighborhood for variances. You'd want to know that before spending $50K on an architect."

---

### Act 2: Parcel Lookup (1 min)
**Enter Parcel ID: 0100001000**
- Shows zoning code (3A-3C), East Boston Neighborhood district
- Map renders the actual parcel boundary
- "Every one of Boston's 98,000 parcels is in here with zoning, district, and article references."

---

### Act 3: The Prediction (2 min) — THE MONEY SHOT

**Configure a realistic project:**
- Parcel ID: 0100001000
- Proposed Use: Residential
- Project Type: Addition / Extension
- Variances: Height, Parking
- Ward: 1
- Attorney: ✅ checked

**Click "Predict Approval Likelihood"**

Walk them through the result:
- "The model says 73% likely to pass" (or whatever it shows)
- "HIGH confidence because we have hundreds of similar cases in this ward"
- Point out Key Factors: "Attorney representation adds ~18% — that's real data"
- Show Similar Cases: "Here are 5 actual ZBA decisions for similar projects"

**Now uncheck Attorney:**
- Watch the probability drop
- "That's the difference legal representation makes. Quantified."

**Now change to Multi-Family Development:**
- Watch it drop further
- "Multi-family projects have a 41% approval rate historically. The board scrutinizes them harder."

---

### Act 4: The What-If (1 min)
Scroll down to the **What-If** section below the prediction:
- Now shows **actual model-computed scenarios** (not estimates!)
- "With an attorney: 81% (+18%)" — real numbers from the ML model
- "With 1 variance instead of 3: 79% (+6%)" — actual probability changes
- "Best case (attorney + minimal variances): 85% (+12%)"
- "This is the kind of analysis that saves developers $50K in guesswork."

**Download the Report:**
- Click "Download Analysis Report"
- "Every analysis can be exported and attached to your permit application or shared with your attorney."

---

### Act 5: The Business Case (1 min)
"Right now, developers spend $30-100K on permitting with no idea if they'll get approved. PermitIQ tells you before you file. It's the difference between walking in blind and walking in prepared."

"No one else does this. The competitors — UrbanForm, Zoneomics — tell you the rules. We tell you if you'll win."

---

### NEW: Ward Insights
Expand the "Ward Insights" section at the bottom:
- Enter Ward 1 — shows approval rate vs Boston average
- "Every ward has different patterns. Ward 1 in East Boston has very different outcomes than Ward 5 in Back Bay."

---

## Demo Addresses to Search
| Address | Cases | Rate | Good For |
|---------|-------|------|----------|
| 75 Tremont Street | 14 | 62% | Mixed results, good story |
| 1081 River Street | 13 | 73% | High volume, mostly approved |
| 370 Vermont Street | 11 | 78% | Strong approval area |
| 58 Burbank Street | 10 | 44% | Tough area, shows risk |
| 226 Magnolia Street | 10 | 50% | Coin flip — shows model value |
| 354 E Street | 11 | 70% | South Boston |
| 900 Beacon Street | 10 | 70% | Beacon St name recognition |

## Sample Parcel IDs
| Parcel ID | Zoning | Neighborhood |
|-----------|--------|-------------|
| 0100001000 | 3A-3C | East Boston |
| 0302951010 | 1 | South Boston |
| 1000358010 | 6D | Jamaica Plain |
| 2100394000 | 7A-7D | Allston/Brighton |

---

## If Something Breaks
- **API won't start**: Check that `zba_model.pkl` exists in the `api/` folder
- **"No model loaded"**: Run `python3 train_model_v2.py` first
- **Map won't render**: Requires internet for Carto basemap tiles
- **Prediction shows "heuristic_baseline"**: Model file not loaded — retrain
- **Search returns nothing**: Make sure `zba_cases_cleaned.csv` is in the project root

## Key Stats to Drop
- 7,500+ unique ZBA case decisions (deduplicated, from 262 decision PDFs)
- 69 engineered features per prediction (no data leakage — only pre-hearing inputs)
- 98,510 Boston parcels with zoning data
- Every decision from 2020 through March 2026
- Integrated with City property assessments, building permits, and ZBA tracker
- Model AUC ~0.75+ (realistic — trained on clean features only)
