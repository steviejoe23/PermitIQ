#!/usr/bin/env python3
"""
Build a neighborhood opposition risk index from ZBA transcript data.

Reads parsed_cases.jsonl (transcript-level hearing data) and zba_cases_cleaned.csv
(case-level decision data), then produces data/opposition_index.json with per-neighborhood
opposition ratios, variance-specific scores, denial rates, and trend analysis.
"""

import json
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from typing import Optional, List, Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED_CASES = os.path.join(PROJECT_ROOT, "data", "zba_transcripts", "parsed_cases.jsonl")
ZBA_CSV = os.path.join(PROJECT_ROOT, "zba_cases_cleaned.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "opposition_index.json")

# ---------------------------------------------------------------------------
# Boston ward-to-neighborhood mapping
# Wards often span multiple neighborhoods; we use the primary/dominant one.
# ---------------------------------------------------------------------------
WARD_TO_NEIGHBORHOOD = {
    "1": "East Boston",
    "2": "Charlestown",
    "3": "Downtown",         # Downtown / Beacon Hill / North End
    "4": "Back Bay",         # Back Bay / South End
    "5": "Back Bay",         # South End / Bay Village
    "6": "South Boston",
    "7": "South Boston",
    "8": "Roxbury",
    "9": "Fenway",           # Fenway / Mission Hill
    "10": "Jamaica Plain",
    "11": "Jamaica Plain",
    "12": "Roxbury",         # Roxbury / Nubian Sq area
    "13": "Dorchester",
    "14": "Dorchester",
    "15": "Dorchester",
    "16": "Dorchester",
    "17": "Mattapan",
    "18": "Hyde Park",
    "19": "Roslindale",      # Roslindale / West Roxbury
    "20": "West Roxbury",
    "21": "Brighton",        # Allston-Brighton
    "22": "Brighton",        # Allston-Brighton
}


def risk_level(ratio: float) -> str:
    if ratio < 0.3:
        return "Low"
    elif ratio <= 0.5:
        return "Medium"
    return "High"


def parse_year_from_source(source_file: str) -> Optional[int]:
    """Extract a 4-digit year from source_file like 'ZBA_Hearing_2016_08_23_xxx'."""
    parts = source_file.split("_")
    for p in parts:
        if len(p) == 4 and p.isdigit():
            yr = int(p)
            if 2000 <= yr <= 2030:
                return yr
    return None


def load_parsed_cases(path: str) -> List[dict]:
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_csv_denial_rates(path: str) -> Dict:
    """Return {neighborhood: {"approved": n, "denied": n}} from the CSV."""
    counts = defaultdict(lambda: {"approved": 0, "denied": 0})
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ward_raw = row.get("ward", "").strip()
            decision = row.get("decision_clean", "").strip().upper()
            if not ward_raw or decision not in ("APPROVED", "DENIED"):
                continue
            # ward may be '1.0' -> normalize to '1'
            ward_key = ward_raw.split(".")[0]
            neighborhood = WARD_TO_NEIGHBORHOOD.get(ward_key)
            if not neighborhood:
                continue
            if "APPROVED" in decision or "GRANTED" in decision:
                counts[neighborhood]["approved"] += 1
            elif "DENIED" in decision:
                counts[neighborhood]["denied"] += 1
    return dict(counts)


def main():
    # ------------------------------------------------------------------
    # 1. Load transcript hearing data
    # ------------------------------------------------------------------
    hearings = load_parsed_cases(PARSED_CASES)
    print(f"Loaded {len(hearings)} hearings from parsed_cases.jsonl")

    # ------------------------------------------------------------------
    # 2-3. Compute opposition ratios and aggregate by neighborhood
    # ------------------------------------------------------------------
    # For each hearing, compute opposition_ratio, then fan out to each
    # neighborhood mentioned in that hearing.

    # Determine the "recent" cutoff: most recent 2 years vs older
    years = []
    for h in hearings:
        yr = parse_year_from_source(h.get("source_file", ""))
        if yr:
            years.append(yr)
    if years:
        max_year = max(years)
        recent_cutoff = max_year - 1  # recent = max_year and max_year-1
    else:
        recent_cutoff = 9999

    # Accumulators per neighborhood
    hood_ratios = defaultdict(list)            # all ratios
    hood_recent = defaultdict(list)            # ratios for recent hearings
    hood_old = defaultdict(list)               # ratios for older hearings
    hood_max = defaultdict(float)
    # Variance-specific: (neighborhood, variance) -> list of ratios
    var_ratios = defaultdict(list)

    for h in hearings:
        sup = h.get("support_mentions", 0)
        opp = h.get("opposition_mentions", 0)
        ratio = opp / (sup + opp + 1)

        yr = parse_year_from_source(h.get("source_file", ""))
        neighborhoods = h.get("neighborhoods", [])
        variances = h.get("variances_mentioned", [])

        for nb in neighborhoods:
            nb_clean = nb.strip()
            if not nb_clean:
                continue
            hood_ratios[nb_clean].append(ratio)
            if hood_max[nb_clean] < ratio:
                hood_max[nb_clean] = ratio

            # Trend buckets
            if yr is not None:
                if yr >= recent_cutoff:
                    hood_recent[nb_clean].append(ratio)
                else:
                    hood_old[nb_clean].append(ratio)

            # Variance-specific
            for v in variances:
                v_clean = v.strip().lower()
                if v_clean:
                    var_ratios[(nb_clean, v_clean)].append(ratio)

    # ------------------------------------------------------------------
    # 4. Cross-reference CSV for denial rates
    # ------------------------------------------------------------------
    if os.path.exists(ZBA_CSV):
        csv_counts = load_csv_denial_rates(ZBA_CSV)
        print(f"Loaded denial-rate data for {len(csv_counts)} neighborhoods from CSV")
    else:
        print(f"WARNING: {ZBA_CSV} not found; denial rates will be null")
        csv_counts = {}

    # ------------------------------------------------------------------
    # 5-6. Build the output
    # ------------------------------------------------------------------
    neighborhoods_out = {}
    for nb in sorted(hood_ratios.keys()):
        ratios = hood_ratios[nb]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        # Trend
        recent_avg = sum(hood_recent[nb]) / len(hood_recent[nb]) if hood_recent[nb] else None
        old_avg = sum(hood_old[nb]) / len(hood_old[nb]) if hood_old[nb] else None
        if recent_avg is not None and old_avg is not None:
            if recent_avg > old_avg + 0.03:
                trend = "increasing"
            elif recent_avg < old_avg - 0.03:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Denial rate from CSV
        csv_data = csv_counts.get(nb, {})
        total_decided = csv_data.get("approved", 0) + csv_data.get("denied", 0)
        denial_rate = round(csv_data["denied"] / total_decided, 4) if total_decided > 0 else None

        # Variance-specific opposition
        variance_opposition = {}
        for (nb2, var), vr_list in var_ratios.items():
            if nb2 != nb:
                continue
            v_avg = sum(vr_list) / len(vr_list) if vr_list else 0.0
            variance_opposition[var] = {
                "opposition_ratio": round(v_avg, 4),
                "hearings": len(vr_list),
                "risk_level": risk_level(v_avg),
            }
        # Sort variance_opposition by ratio descending
        variance_opposition = dict(
            sorted(variance_opposition.items(), key=lambda kv: -kv[1]["opposition_ratio"])
        )

        neighborhoods_out[nb] = {
            "avg_opposition_ratio": round(avg_ratio, 4),
            "max_opposition_ratio": round(hood_max[nb], 4),
            "hearings_analyzed": len(ratios),
            "risk_level": risk_level(avg_ratio),
            "denial_rate": denial_rate,
            "trend": trend,
            "variance_opposition": variance_opposition,
        }

    output = {
        "neighborhoods": neighborhoods_out,
        "generated": date.today().isoformat(),
        "total_hearings_analyzed": len(hearings),
        "recent_cutoff_year": recent_cutoff if years else None,
    }

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote opposition index to {OUTPUT_PATH}")
    print(f"  Neighborhoods: {len(neighborhoods_out)}")
    print(f"  Total hearings: {len(hearings)}")

    # Quick summary
    print("\n--- Risk Summary ---")
    for nb, data in sorted(neighborhoods_out.items(), key=lambda kv: -kv[1]["avg_opposition_ratio"]):
        dr = f"{data['denial_rate']:.1%}" if data['denial_rate'] is not None else "N/A"
        print(
            f"  {nb:25s}  opposition={data['avg_opposition_ratio']:.3f}  "
            f"risk={data['risk_level']:6s}  denial_rate={dr:>6s}  "
            f"hearings={data['hearings_analyzed']:3d}  trend={data['trend']}"
        )


if __name__ == "__main__":
    main()
