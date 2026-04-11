#!/usr/bin/env python3
"""Parse ZBA hearing SRT/VTT transcripts into structured case data.

Extracts from hearing transcripts:
- BOA case numbers
- Addresses
- Attorney/applicant names
- Variance types mentioned
- Community support/opposition signals
- Decision outcomes (approved/denied/deferred)
- Sentiment and discussion context

Output: JSON lines file suitable for enriching the ZBA training dataset.

Usage:
    python scripts/parse_zba_transcripts.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

SRT_DIR = Path(__file__).parent.parent / "data" / "zba_transcripts" / "raw_srt"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "zba_transcripts" / "parsed_cases.jsonl"
FULL_TEXT_DIR = Path(__file__).parent.parent / "data" / "zba_transcripts" / "full_text"


def parse_srt(filepath: str) -> str:
    """Parse an SRT file into plain text, stripping timestamps and sequence numbers."""
    lines = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Skip sequence numbers (just digits)
            if re.match(r"^\d+$", line):
                continue
            # Skip SRT timestamp lines
            if re.match(r"^\d{2}:\d{2}:\d{2}", line):
                continue
            # Skip VTT header
            if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            # Skip empty lines
            if not line:
                continue
            # Remove HTML tags from subtitles
            line = re.sub(r"<[^>]+>", "", line)
            if line:
                lines.append(line)
    return " ".join(lines)


def parse_vtt(filepath: str) -> str:
    """Parse a VTT file into plain text."""
    return parse_srt(filepath)  # Same logic works for both


def extract_date_from_filename(filename: str) -> str:
    """Extract hearing date from the archive.org identifier filename."""
    # Patterns like: cobma-Zoning_Board_of_Appeal_Hearings_4_7_26
    # or: cobma-Zoning_Board_of_Appeal_Hearings_3-10-26
    # or: cobma-Zoning_Board_of_Appeal_Hearings_12_3_2024_Part_1_of_3

    # Try full year patterns first (e.g., 12_3_2024)
    m = re.search(r"(\d{1,2})[_-](\d{1,2})[_-](20\d{2})", filename)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try short year patterns (e.g., 4_7_26 or 3-10-26)
    m = re.search(r"(\d{1,2})[_-](\d{1,2})[_-](\d{2})(?:_|$|\.|Part)", filename)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year += 2000
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try month-name patterns for subcommittee hearings
    # e.g., Boston_Zoning_Board_of_Appeal_February_12_2026
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)_(\d{1,2})_(\d{4})", filename)
    if m:
        month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        month = datetime.strptime(month_name, "%B").month
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return "unknown"


# Patterns for extracting structured data from transcript text
BOA_PATTERN = re.compile(
    r"(?:BOA|B\.O\.A\.?|board of appeal|case)\s*(?:number|no\.?|#)?\s*"
    r"(\d{2,4}[-–]\d{3,5})",
    re.IGNORECASE
)

ADDRESS_PATTERN = re.compile(
    r"(\d{1,5}(?:\s*[-–]\s*\d{1,5})?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+"
    r"(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
    r"Drive|Dr\.?|Lane|Ln\.?|Way|Place|Pl\.?|Court|Ct\.?|"
    r"Terrace|Ter\.?|Circle|Cir\.?)"
    r"(?:\s|,|$)",
    re.IGNORECASE
)

ATTORNEY_PATTERN = re.compile(
    r"(?:attorney|counsel|lawyer|on behalf of)\s+(?:is\s+|named?\s+)?"
    r"([A-Z][a-z]{2,15}\s+[A-Z][a-z]{2,15})",
    re.IGNORECASE
)

VARIANCE_TYPES = [
    "use", "height", "far", "floor area ratio", "open space",
    "parking", "setback", "lot area", "frontage", "density",
    "rear yard", "side yard", "front yard", "stories",
    "lot coverage", "building height", "conditional use"
]

DECISION_PATTERNS = {
    "approved": re.compile(
        r"(?:motion\s+(?:to\s+)?(?:approve|grant)|approved|granted|"
        r"vote\s+(?:is\s+)?(?:to\s+)?approve|unanimous(?:ly)?\s+approved)",
        re.IGNORECASE
    ),
    "denied": re.compile(
        r"(?:motion\s+(?:to\s+)?deny|denied|rejected|"
        r"vote\s+(?:is\s+)?(?:to\s+)?deny)",
        re.IGNORECASE
    ),
    "deferred": re.compile(
        r"(?:defer(?:red)?|continued|postpone[d]?|table[d]?|"
        r"(?:put|held)\s+(?:over|off))",
        re.IGNORECASE
    ),
}

SUPPORT_PATTERNS = re.compile(
    r"(?:in\s+(?:support|favor)|support(?:s|ing)?|"
    r"(?:councilor|council\s*member)\s+\w+\s+(?:supports?|in\s+favor))",
    re.IGNORECASE
)

OPPOSITION_PATTERNS = re.compile(
    r"(?:in\s+opposition|oppos(?:e[ds]?|ition|ing)|"
    r"against|concern(?:s|ed)?|object(?:s|ion)?)",
    re.IGNORECASE
)

NEIGHBORHOOD_MAP = {
    "allston": "Allston", "brighton": "Brighton",
    "back bay": "Back Bay", "beacon hill": "Beacon Hill",
    "charlestown": "Charlestown", "chinatown": "Chinatown",
    "dorchester": "Dorchester", "downtown": "Downtown",
    "east boston": "East Boston", "fenway": "Fenway",
    "hyde park": "Hyde Park", "jamaica plain": "Jamaica Plain",
    "mattapan": "Mattapan", "mission hill": "Mission Hill",
    "north end": "North End", "roslindale": "Roslindale",
    "roxbury": "Roxbury", "south boston": "South Boston",
    "south end": "South End", "west end": "West End",
    "west roxbury": "West Roxbury",
}


def extract_cases_from_text(text: str, hearing_date: str, source_file: str) -> list:
    """Extract individual case data from transcript text.

    Uses a sliding window approach to segment the transcript into
    approximate case discussions and extract structured data from each.
    """
    cases = []
    text_lower = text.lower()

    # Find all BOA case numbers
    boa_matches = list(BOA_PATTERN.finditer(text))

    # Find all addresses — clean and deduplicate
    addr_matches = list(ADDRESS_PATTERN.finditer(text))
    clean_addresses = set()
    for m in addr_matches:
        num = m.group(1).strip()
        street = m.group(2).strip()
        suffix = m.group(3).strip()
        addr = f"{num} {street} {suffix}"
        # Skip addresses that are too long (ASR noise) or too short
        if len(addr) < 50 and len(num) <= 5:
            clean_addresses.add(addr)

    # Find all attorney mentions — filter noise
    attorney_matches = list(ATTORNEY_PATTERN.finditer(text))
    clean_attorneys = set()
    noise_words = {"the", "is", "with", "name", "mayor", "that", "this", "from", "city"}
    for m in attorney_matches:
        name = m.group(1).strip()
        parts = name.lower().split()
        if not any(p in noise_words for p in parts):
            clean_attorneys.add(name)

    # Extract variance types mentioned
    variances_found = []
    for vt in VARIANCE_TYPES:
        if vt.lower() in text_lower:
            variances_found.append(vt)

    # Count support/opposition mentions
    support_count = len(SUPPORT_PATTERNS.findall(text))
    opposition_count = len(OPPOSITION_PATTERNS.findall(text))

    # Count decisions
    decisions = {}
    for dec_type, pattern in DECISION_PATTERNS.items():
        count = len(pattern.findall(text))
        if count > 0:
            decisions[dec_type] = count

    # Detect neighborhoods mentioned
    neighborhoods = []
    for key, name in NEIGHBORHOOD_MAP.items():
        if key in text_lower:
            neighborhoods.append(name)

    # Build a summary record for this hearing
    hearing_record = {
        "hearing_date": hearing_date,
        "source_file": source_file,
        "boa_cases": [m.group(1) for m in boa_matches],
        "addresses": sorted(clean_addresses),
        "attorneys": sorted(clean_attorneys),
        "variances_mentioned": variances_found,
        "neighborhoods": list(set(neighborhoods)),
        "support_mentions": support_count,
        "opposition_mentions": opposition_count,
        "decisions": decisions,
        "total_approved": decisions.get("approved", 0),
        "total_denied": decisions.get("denied", 0),
        "total_deferred": decisions.get("deferred", 0),
        "text_length": len(text),
        "word_count": len(text.split()),
    }
    cases.append(hearing_record)

    return cases


def main():
    FULL_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    srt_files = sorted(SRT_DIR.glob("*.*"))
    if not srt_files:
        print("No SRT/VTT files found. Run download_zba_transcripts.py first.")
        return

    print(f"Found {len(srt_files)} transcript files to parse")

    all_cases = []
    total_words = 0

    for i, filepath in enumerate(srt_files):
        filename = filepath.stem  # without extension
        hearing_date = extract_date_from_filename(filename)

        # Parse the subtitle file
        if filepath.suffix == ".vtt":
            text = parse_vtt(str(filepath))
        else:
            text = parse_srt(str(filepath))

        if not text or len(text) < 100:
            print(f"  [SKIP] {filename}: too short ({len(text)} chars)")
            continue

        word_count = len(text.split())
        total_words += word_count

        # Save full text for reference
        text_file = FULL_TEXT_DIR / f"{filename}.txt"
        with open(text_file, "w") as f:
            f.write(f"# {filename}\n# Date: {hearing_date}\n# Words: {word_count}\n\n")
            f.write(text)

        # Extract structured case data
        cases = extract_cases_from_text(text, hearing_date, filename)
        all_cases.extend(cases)

        if (i + 1) % 50 == 0:
            print(f"  Parsed {i+1}/{len(srt_files)} files ({total_words:,} words so far)")

    # Write all cases to JSONL
    with open(OUTPUT_FILE, "w") as f:
        for case in all_cases:
            f.write(json.dumps(case) + "\n")

    print(f"\nDone!")
    print(f"  Parsed {len(srt_files)} transcript files")
    print(f"  Extracted {len(all_cases)} hearing records")
    print(f"  Total words: {total_words:,}")
    print(f"  Output: {OUTPUT_FILE}")

    # Summary stats
    total_approved = sum(c.get("total_approved", 0) for c in all_cases)
    total_denied = sum(c.get("total_denied", 0) for c in all_cases)
    total_deferred = sum(c.get("total_deferred", 0) for c in all_cases)
    all_neighborhoods = set()
    for c in all_cases:
        all_neighborhoods.update(c.get("neighborhoods", []))
    all_variances = set()
    for c in all_cases:
        all_variances.update(c.get("variances_mentioned", []))

    print(f"\n  Decision mentions: {total_approved} approved, {total_denied} denied, {total_deferred} deferred")
    print(f"  Neighborhoods covered: {sorted(all_neighborhoods)}")
    print(f"  Variance types found: {sorted(all_variances)}")


if __name__ == "__main__":
    main()
