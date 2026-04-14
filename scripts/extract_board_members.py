#!/usr/bin/env python3
"""
Extract board member names and voting patterns from ZBA hearing transcripts.

Reads transcripts from raw_srt/ and full_text/, extracts board member mentions
using regex patterns, fuzzy-matches name variants, cross-references with case
outcome data, and outputs per-member statistics to data/board_member_profiles.json.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "zba_transcripts"
RAW_SRT_DIR = TRANSCRIPTS_DIR / "raw_srt"
FULL_TEXT_DIR = TRANSCRIPTS_DIR / "full_text"
PARSED_CASES = TRANSCRIPTS_DIR / "parsed_cases.jsonl"
TRACKER_CSV = PROJECT_ROOT / "zba_tracker.csv"
CASES_CSV = PROJECT_ROOT / "zba_cases_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "board_member_profiles.json"


# ── Date extraction from filenames ─────────────────────────────────────────
DATE_PATTERNS = [
    # ZBA_Hearing_2024_04_09_xxx.txt
    re.compile(r'ZBA_Hearing_(\d{4})_(\d{2})_(\d{2})'),
    # cobma-..._1-23-24_... or ..._1_23_24
    re.compile(r'(\d{1,2})[-_](\d{1,2})[-_](\d{2,4})'),
    # cobma-..._January_14_2025_...
    re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)_(\d{1,2})_(\d{4})'),
    # Zoning_Board_of_Appeal_Hearings_08-29-17
    re.compile(r'Hearings?_(\d{2})-(\d{2})-(\d{2,4})'),
]

MONTH_MAP = {m: i for i, m in enumerate([
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], 0)}


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Try to extract a YYYY-MM-DD date from a transcript filename."""
    basename = Path(filename).stem

    # Pattern 1: ZBA_Hearing_YYYY_MM_DD
    m = re.search(r'ZBA_Hearing_(\d{4})_(\d{2})_(\d{2})', basename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Pattern 2: Month_DD_YYYY (e.g. June_12_2025)
    m = re.search(
        r'(January|February|March|April|May|June|July|August|September|'
        r'October|November|December)_(\d{1,2})_(\d{4})', basename)
    if m:
        month = MONTH_MAP.get(m.group(1), 0)
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"

    # Pattern 3: M-DD-YY or M_DD_YY at end of name
    m = re.search(r'(\d{1,2})[-_](\d{1,2})[-_](\d{2,4})(?:_|$)', basename)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        if 1 <= month <= 12 and 1 <= day <= 31 and 2014 <= year <= 2027:
            return f"{year}-{month:02d}-{day:02d}"

    return None


# ── Name extraction patterns ──────────────────────────────────────────────
# These are tuned for Whisper-transcribed Boston ZBA hearings.
# We capture the title context to infer role.

TITLE_PATTERNS = [
    # "Chairman Erlich", "Chair Collins", "Chairwoman Robinson"
    (re.compile(r'\b[Cc]hair(?:man|woman|person)?\s+([A-Z][a-z]{2,})'), 'Chair'),
    # "Member Valencia", "Board Member Stembridge"
    (re.compile(r'\b(?:[Bb]oard\s+)?[Mm]ember\s+([A-Z][a-z]{2,})'), 'Member'),
    # "Commissioner Williams"
    (re.compile(r'\b[Cc]ommissioner\s+([A-Z][a-z]{2,})'), 'Commissioner'),
]

# Mr./Ms./Mrs. mentions — most common way members are referenced
MR_MS_PATTERN = re.compile(r'\b(?:Mr|Ms|Mrs)\.\s+([A-Z][a-z]{2,})')

# Vote patterns: "Mr. Collins yes", "Mr. Valencia aye", etc.
VOTE_PATTERN = re.compile(
    r'(?:Mr|Ms|Mrs)\.\s+([A-Z][a-z]{2,})\s*[,:]?\s*'
    r'(yes|no|aye|nay|in\s+favor|against|abstain)',
    re.IGNORECASE
)

# "all in favor" / "motion carried" — board-wide approval signals
BOARD_VOTE_PATTERN = re.compile(
    r'\b(all\s+in\s+favor|motion\s+(?:carried|passes|approved)|'
    r'unanimously\s+(?:approved|granted|denied))\b',
    re.IGNORECASE
)

# Names to exclude — these are false positives from Whisper transcription
EXCLUDE_NAMES = {
    'Chair', 'Chairman', 'Chairwoman', 'President', 'Secretary',
    'Ambassador', 'Commissioner', 'Board', 'Members', 'Thank',
    'Madam', 'Miss', 'Broadway', 'May', 'June', 'March', 'April',
    'Zero', 'Beta', 'Council', 'Association', 'Civic', 'Better',
    'Street', 'Avenue', 'Road', 'Place', 'Drive', 'Court',
    'Building', 'Floor', 'Room', 'Office', 'Department',
    'North', 'South', 'East', 'West', 'City', 'State', 'County',
    'Services', 'Commission', 'Ways', 'And', 'The', 'This',
    'Good', 'Morning', 'Afternoon', 'Evening', 'Today',
    'Article', 'Section', 'Zoning', 'Code', 'Hearing',
    'Motion', 'Second', 'Favor', 'Opposed', 'Approved', 'Denied',
    'Joseph', 'Christopher', 'Christian', 'John', 'Joe', 'Paul',
    'Mary', 'James', 'George', 'David', 'Michael', 'Jeff',
    'Chris', 'Jose', 'Marie', 'Jean', 'Caroline', 'Ford',
    'Oliver', 'Olivier', 'Del', 'Alec', 'Oleg', 'Dian',
    'Sor', 'Sigy', 'Bever', 'Olic', 'Beta', 'Better',
    'Barza', 'Bazer', 'Speaker', 'Public', 'Attorney',
    'Applicant', 'Resident', 'Counselor', 'Representative',
    'Stephanie', 'Boston', 'Suffolk', 'Record', 'Counsel',
    'Ward', 'Neighborhood', 'Developer', 'Architect',
    'Inspector', 'Director', 'Planner', 'Manager',
}

# Minimum mentions to be considered a board member (not just an applicant)
MIN_MENTIONS_THRESHOLD = 15

# Known Whisper-mangled name pairs that don't meet the 0.85 fuzzy threshold
# but are clearly the same person. Format: {variant: canonical}
KNOWN_MERGES = {
    'Sambridge': 'Stembridge',
    'Sanbridge': 'Stembridge',
    'Panado': 'Panato',
    'Pazani': 'Panato',
    'Pazzani': 'Panato',
    'Langum': 'Langham',
    'Langam': 'Langham',
    'Regiro': 'Ruggiero',
    'Rugiro': 'Ruggiero',
    'Marancy': 'Morancy',
    'Ligress': 'Legress',
    'Erlich': 'Ehrlich',
    'Bazzani': 'Pazani',
    'Shepard': 'Shephard',
    'Dregov': 'Drego',
}


def is_valid_board_name(name: str) -> bool:
    """Filter out names that are likely false positives."""
    if name in EXCLUDE_NAMES:
        return False
    if len(name) < 3:
        return False
    return True


# ── Fuzzy matching ─────────────────────────────────────────────────────────
def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_name_variants(name_data: dict, threshold: float = 0.85) -> list[dict]:
    """
    Merge name variants using fuzzy matching.
    name_data: {name: {hearings: set, mentions: int, votes: Counter, role: str}}
    Returns list of merged member dicts.
    """
    # First, apply known merges
    for variant, canonical in KNOWN_MERGES.items():
        if variant in name_data and variant != canonical:
            if canonical not in name_data:
                # Rename variant to canonical
                name_data[canonical] = name_data.pop(variant)
            else:
                # Merge variant into canonical
                name_data[canonical]['hearings'] |= name_data[variant]['hearings']
                name_data[canonical]['mentions'] += name_data[variant]['mentions']
                name_data[canonical]['votes'] += name_data[variant]['votes']
                if name_data[variant]['role'] == 'Chair':
                    name_data[canonical]['role'] = 'Chair'
                del name_data[variant]

    names = sorted(name_data.keys(), key=lambda n: -name_data[n]['mentions'])
    merged = []
    used = set()

    for name in names:
        if name in used:
            continue
        # This is the canonical (most-mentioned) variant
        group = {
            'canonical': name,
            'aliases': set(),
            'hearings': set(name_data[name]['hearings']),
            'mentions': name_data[name]['mentions'],
            'votes': Counter(name_data[name]['votes']),
            'role': name_data[name]['role'],
        }
        used.add(name)

        # Find similar names
        for other in names:
            if other in used:
                continue
            if fuzzy_ratio(name, other) >= threshold:
                group['aliases'].add(other)
                group['hearings'] |= name_data[other]['hearings']
                group['mentions'] += name_data[other]['mentions']
                group['votes'] += name_data[other]['votes']
                # Prefer Chair role
                if name_data[other]['role'] == 'Chair':
                    group['role'] = 'Chair'
                used.add(other)

        merged.append(group)

    return merged


# ── Date extraction from transcript text ────────────────────────────────────
TEXT_DATE_PATTERN = re.compile(
    r'(?:hearing\s+for|board\s+of\s+appeal\s+(?:hearing\s+)?for)\s+'
    r'(?:today[,.]?\s+)?'
    r'(?:(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|'
    r'(?:\w+day,?\s+)?(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4})',
    re.IGNORECASE
)

TEXT_DATE_SIMPLE = re.compile(
    r'(January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})',
    re.IGNORECASE
)


def extract_date_from_text(text: str) -> Optional[str]:
    """Try to extract hearing date from transcript text (first 2000 chars)."""
    snippet = text[:3000]
    m = TEXT_DATE_SIMPLE.search(snippet)
    if m:
        month_name = m.group(1).capitalize()
        month = MONTH_MAP.get(month_name, 0)
        if month > 0:
            day = int(m.group(2))
            year = int(m.group(3))
            if 2014 <= year <= 2027:
                return f"{year}-{month:02d}-{day:02d}"
    return None


# ── Main extraction ────────────────────────────────────────────────────────
def load_transcripts() -> list[tuple[str, str, Optional[str]]]:
    """Load all transcripts. Returns list of (filename, text, date)."""
    transcripts = []
    seen_keys = set()

    # Directories to scan: full_text, raw_srt, and top-level .txt files
    dirs_to_scan = [FULL_TEXT_DIR, RAW_SRT_DIR, TRANSCRIPTS_DIR]

    for directory in dirs_to_scan:
        if not directory.exists():
            print(f"  Warning: {directory} not found, skipping")
            continue
        for fn in sorted(os.listdir(directory)):
            if not (fn.endswith('.txt') or fn.endswith('.vtt')):
                continue
            filepath = directory / fn
            if not filepath.is_file():
                continue

            date = extract_date_from_filename(fn)

            # Deduplicate by date (full_text and raw_srt have the same content)
            dedup_key = date or fn
            if dedup_key in seen_keys:
                continue

            try:
                text = filepath.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"  Warning: could not read {filepath}: {e}")
                continue

            # If no date from filename, try extracting from text
            if not date:
                date = extract_date_from_text(text)
                if date and date in seen_keys:
                    continue  # Already have this date
                if date:
                    seen_keys.add(date)

            seen_keys.add(dedup_key)
            transcripts.append((fn, text, date))

    return transcripts


def extract_members_from_transcripts(transcripts):
    """
    Extract board member names and voting data from transcripts.
    Returns name_data dict keyed by name.
    """
    name_data = defaultdict(lambda: {
        'hearings': set(),
        'mentions': 0,
        'votes': Counter(),  # {'yes': N, 'no': N}
        'role': 'Member',
    })

    total_board_votes = defaultdict(int)  # date -> count of "all in favor" etc.

    for filename, text, date in transcripts:
        date_key = date or filename

        # Extract titled names (Chairman X, Member X, etc.)
        for pattern, role in TITLE_PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1)
                if not is_valid_board_name(name):
                    continue
                name_data[name]['hearings'].add(date_key)
                name_data[name]['mentions'] += 1
                if role == 'Chair':
                    name_data[name]['role'] = 'Chair'

        # Extract Mr./Ms. mentions
        for m in MR_MS_PATTERN.finditer(text):
            name = m.group(1)
            if not is_valid_board_name(name):
                continue
            name_data[name]['hearings'].add(date_key)
            name_data[name]['mentions'] += 1

        # Extract individual votes
        for m in VOTE_PATTERN.finditer(text):
            name = m.group(1)
            vote = m.group(2).lower().strip()
            if not is_valid_board_name(name):
                continue
            # Normalize vote
            if vote in ('yes', 'aye', 'in favor'):
                vote_norm = 'yes'
            elif vote in ('no', 'nay', 'against'):
                vote_norm = 'no'
            else:
                vote_norm = 'abstain'
            name_data[name]['votes'][vote_norm] += 1
            name_data[name]['hearings'].add(date_key)

        # Count board-wide votes
        for m in BOARD_VOTE_PATTERN.finditer(text):
            total_board_votes[date_key] += 1

    return name_data, total_board_votes


def load_case_outcomes() -> dict:
    """
    Load case outcomes from tracker and cleaned CSV.
    Returns {hearing_date: {total, approved, denied, variance_stats}}.
    """
    date_outcomes = defaultdict(lambda: {
        'total': 0, 'approved': 0, 'denied': 0,
        'variance_stats': defaultdict(lambda: {'total': 0, 'approved': 0})
    })

    # Load tracker for hearing_date -> decision mapping
    if TRACKER_CSV.exists():
        print(f"  Loading tracker: {TRACKER_CSV}")
        with open(TRACKER_CSV, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hdate = row.get('hearing_date', '').strip()
                decision = row.get('decision', '').strip().lower()
                if not hdate or len(hdate) < 8:
                    continue
                date_outcomes[hdate]['total'] += 1
                if 'approv' in decision or 'grant' in decision:
                    date_outcomes[hdate]['approved'] += 1
                elif 'deni' in decision or 'dismiss' in decision:
                    date_outcomes[hdate]['denied'] += 1
    else:
        print(f"  Warning: {TRACKER_CSV} not found")

    # Build case_number -> hearing_date mapping from tracker
    case_to_hearing_date = {}
    if TRACKER_CSV.exists():
        with open(TRACKER_CSV, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                boa_apno = row.get('boa_apno', '').strip()
                hdate = row.get('hearing_date', '').strip()
                if boa_apno and hdate and len(hdate) >= 8:
                    case_to_hearing_date[boa_apno] = hdate

    # Load cleaned cases for variance type breakdown, joining on hearing_date
    if CASES_CSV.exists():
        print(f"  Loading cases for variance stats: {CASES_CSV}")
        variance_types_map = {
            'height': ['height', 'building height', 'stories'],
            'parking': ['parking'],
            'far': ['far', 'floor area ratio'],
            'use': ['use', 'conditional use'],
            'setback': ['setback', 'front yard', 'rear yard', 'side yard'],
            'density': ['density', 'lot area', 'units'],
            'open_space': ['open space'],
        }
        matched = 0
        with open(CASES_CSV, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_num = row.get('case_number', '').strip()
                decision = row.get('decision_clean', row.get('decision', '')).strip().lower()
                variances = row.get('variance_types', '').lower()
                if not variances:
                    continue

                # Try to find hearing_date via tracker mapping
                hdate = case_to_hearing_date.get(case_num, '')
                if not hdate:
                    # Fallback: use filing_date if available
                    hdate = row.get('filing_date', '').strip()
                if not hdate or len(hdate) < 8:
                    continue

                is_approved = 'grant' in decision or 'approv' in decision
                for vtype, keywords in variance_types_map.items():
                    if any(kw in variances for kw in keywords):
                        date_outcomes[hdate]['variance_stats'][vtype]['total'] += 1
                        if is_approved:
                            date_outcomes[hdate]['variance_stats'][vtype]['approved'] += 1
                        matched += 1
        print(f"  Matched {matched} variance entries to hearing dates")
    else:
        print(f"  Warning: {CASES_CSV} not found")

    return date_outcomes


def compute_member_stats(merged_members, date_outcomes) -> list[dict]:
    """Compute per-member statistics."""
    all_dates = sorted(date_outcomes.keys())
    results = []

    for member in merged_members:
        hearing_dates = sorted([
            d for d in member['hearings']
            if re.match(r'\d{4}-\d{2}-\d{2}', d)
        ])
        if not hearing_dates:
            # Skip members with no parseable dates
            hearing_dates_display = sorted(member['hearings'])
            date_range = [hearing_dates_display[0], hearing_dates_display[-1]] if hearing_dates_display else []
        else:
            date_range = [hearing_dates[0], hearing_dates[-1]]

        # Cases during tenure: all cases on dates this member attended
        cases_during = 0
        approved_during = 0
        denied_during = 0
        variance_agg = defaultdict(lambda: {'cases': 0, 'approved': 0})

        for d in hearing_dates:
            if d in date_outcomes:
                outcomes = date_outcomes[d]
                cases_during += outcomes['total']
                approved_during += outcomes['approved']
                denied_during += outcomes['denied']
                for vtype, vstats in outcomes['variance_stats'].items():
                    variance_agg[vtype]['cases'] += vstats['total']
                    variance_agg[vtype]['approved'] += vstats['approved']

        total_decided = approved_during + denied_during
        approval_rate = round(approved_during / total_decided, 3) if total_decided > 0 else None
        denial_rate = round(denied_during / total_decided, 3) if total_decided > 0 else None

        # Build variance_stats with approval rates
        variance_stats = {}
        for vtype, vstats in sorted(variance_agg.items()):
            if vstats['cases'] > 0:
                variance_stats[vtype] = {
                    'cases': vstats['cases'],
                    'approval_rate': round(vstats['approved'] / vstats['cases'], 3)
                        if vstats['cases'] > 0 else None
                }

        # Build aliases list
        aliases = sorted(member['aliases'])
        if member['role'] == 'Chair':
            aliases.append(f"Chairman {member['canonical']}")
            aliases.append(f"Chair {member['canonical']}")
        aliases = sorted(set(aliases))

        votes = dict(member['votes']) if member['votes'] else None

        entry = {
            'name': member['canonical'],
            'aliases': aliases,
            'role': member['role'],
            'hearings_attended': len(member['hearings']),
            'total_mentions': member['mentions'],
            'date_range': date_range,
            'cases_during_tenure': cases_during,
            'approval_rate': approval_rate,
            'denial_rate': denial_rate,
            'individual_votes': votes,
            'variance_stats': variance_stats if variance_stats else None,
        }
        results.append(entry)

    # Sort by hearings attended descending
    results.sort(key=lambda x: -x['hearings_attended'])
    return results


def main():
    print("=" * 60)
    print("Board Member Extraction from ZBA Hearing Transcripts")
    print("=" * 60)

    # Step 1: Load transcripts
    print("\n[1/5] Loading transcripts...")
    transcripts = load_transcripts()
    print(f"  Loaded {len(transcripts)} transcripts")

    # Step 2: Extract names
    print("\n[2/5] Extracting board member names and votes...")
    name_data, board_votes = extract_members_from_transcripts(transcripts)
    print(f"  Found {len(name_data)} unique name strings")

    # Filter to likely board members (enough mentions)
    filtered = {
        name: data for name, data in name_data.items()
        if data['mentions'] >= MIN_MENTIONS_THRESHOLD
    }
    print(f"  After filtering (>={MIN_MENTIONS_THRESHOLD} mentions): {len(filtered)} candidates")

    # Step 3: Fuzzy merge
    print("\n[3/5] Fuzzy-merging name variants (threshold=0.85)...")
    merged = merge_name_variants(filtered, threshold=0.85)
    print(f"  Merged into {len(merged)} distinct members")
    for m in sorted(merged, key=lambda x: -x['mentions'])[:15]:
        aliases_str = ', '.join(sorted(m['aliases']))[:60]
        print(f"    {m['canonical']:20s}  mentions={m['mentions']:5d}  "
              f"hearings={len(m['hearings']):3d}  role={m['role']:8s}  "
              f"aliases=[{aliases_str}]")

    # Step 4: Load case outcomes
    print("\n[4/5] Loading case outcome data...")
    date_outcomes = load_case_outcomes()
    print(f"  Loaded outcomes for {len(date_outcomes)} hearing dates")

    # Step 5: Compute stats and output
    print("\n[5/5] Computing per-member statistics...")
    member_profiles = compute_member_stats(merged, date_outcomes)

    output = {
        'members': member_profiles,
        'generated': datetime.now().strftime('%Y-%m-%d'),
        'total_transcripts_analyzed': len(transcripts),
        'extraction_notes': (
            'Names extracted via regex from Whisper-transcribed audio. '
            'Spelling variations merged with fuzzy matching (ratio >= 0.85). '
            'Vote counts are approximate due to transcription noise.'
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Output written to: {OUTPUT_PATH}")
    print(f"  Total members: {len(member_profiles)}")
    print(f"  Members with vote data: "
          f"{sum(1 for m in member_profiles if m['individual_votes'])}")

    # Print summary
    print("\n" + "=" * 60)
    print("Top 10 Board Members by Hearing Attendance")
    print("=" * 60)
    for m in member_profiles[:10]:
        votes_str = ""
        if m['individual_votes']:
            v = m['individual_votes']
            votes_str = f"  votes: yes={v.get('yes',0)} no={v.get('no',0)}"
        rate_str = f"  approval={m['approval_rate']:.1%}" if m['approval_rate'] else ""
        print(f"  {m['name']:20s}  hearings={m['hearings_attended']:3d}  "
              f"cases={m['cases_during_tenure']:5d}{rate_str}{votes_str}")


if __name__ == '__main__':
    main()
