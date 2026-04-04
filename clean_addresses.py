"""
PermitIQ Address Cleaner
Cleans the address_clean column in zba_cases_cleaned.csv to fix OCR artifacts,
strip non-address text, normalize formatting, and null out garbage entries.

Run: python3 clean_addresses.py              # Dry run (report only)
     python3 clean_addresses.py --apply      # Overwrite zba_cases_cleaned.csv
"""

import pandas as pd
import re
import os
import shutil
import argparse
from datetime import datetime


# ── Street suffixes for Boston addresses ──
STREET_SUFFIXES = (
    r'(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Boulevard|Blvd\.?|'
    r'Way|Lane|Ln\.?|Court|Ct\.?|Place|Pl\.?|Terrace|Ter\.?|Circle|Cir\.?|'
    r'Square|Sq\.?|Parkway|Pkwy\.?|Highway|Hwy\.?|Broadway|Walk|Path|Row|'
    r'Park|Wharf|Turnpike|Tpke\.?|Pier)'
)

STREET_SUFFIX_SET = {
    'ST', 'STREET', 'AV', 'AVE', 'AVENUE', 'RD', 'ROAD', 'DR', 'DRIVE',
    'PL', 'PLACE', 'TER', 'TERRACE', 'CT', 'COURT', 'LN', 'LANE',
    'BLVD', 'BOULEVARD', 'CIR', 'CIRCLE', 'SQ', 'SQUARE', 'PK', 'PARK',
    'PKWY', 'PARKWAY', 'HWY', 'HIGHWAY', 'WAY', 'PATH', 'ALY', 'ALLEY',
    'ROW', 'WHARF', 'WF', 'PIER', 'BROADWAY', 'WALK', 'TURNPIKE', 'TPKE',
}

# Full street address regex: number + street name + suffix
STREET_ADDR_RE = re.compile(
    rf'^(\d[\d\-]*(?:[A-Z])?\s+[\w\s\.\-]+?{STREET_SUFFIXES})\b',
    re.IGNORECASE
)

# Boston neighborhoods for stripping
NEIGHBORHOODS = [
    'South Boston', 'East Boston', 'West Roxbury', 'North End',
    'Downtown', 'Back Bay', 'Beacon Hill', 'Jamaica Plain',
    'Roxbury', 'Dorchester', 'Brighton', 'Allston', 'Mattapan',
    'Hyde Park', 'Roslindale', 'Charlestown', 'Fenway', 'Mission Hill',
    'Chinatown', 'South End', 'Midtown', 'Seaport', 'Leather District',
    'Bay Village', 'West End',
]

# Words that indicate OCR-captured non-address text
GARBAGE_WORDS = {
    'appeal', 'appellant', 'board', 'reviewed', 'conformity', 'hearing',
    'petition', 'relief', 'pursuant', 'section', 'article', 'chapter',
    'ordinance', 'regulation', 'requirement', 'insufficient', 'excessive',
    'violation', 'penalty', 'dwelling', 'occupancy', 'structure',
    'construct', 'demolish', 'renovate', 'convert', 'alter', 'modify',
    'install', 'remove', 'replace', 'repair', 'erect', 'maintain',
    'consisting', 'residential', 'units', 'mixed', 'stories',
    'january', 'february', 'march', 'april', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'proposed', 'project', 'applicant', 'property', 'zoning',
    'staircase', 'gymnasium', 'sprinkler', 'plumbing', 'electrical',
    'beauty', 'salon', 'health', 'club', 'tenant',
    'absent', 'requested', 'variance', 'tue', 'wed', 'thu', 'fri',
    'mon', 'sat', 'sun', 'again', 'ists',
}


def is_garbage(addr):
    """Return True if this string is clearly not a real street address."""
    if not addr or pd.isna(addr):
        return True
    a = str(addr).strip()
    if len(a) < 4:
        return True
    # No letters at all
    if not re.search(r'[a-zA-Z]{2,}', a):
        return True
    # Starts with 6+ digits (parcel ID or case number, not house number)
    if re.match(r'^\d{6,}', a):
        return True
    # Contains boilerplate legal/decision text
    lower = a.lower()
    for phrase in ['made a part', 'this appeal', 'appeal seeks', 'the board',
                   'public hearing', 'in conformity', 'zoning code',
                   'off street parking', 'rear yard', 'side yard', 'front yard',
                   'lot area', 'floor area', 'building height', 'dwelling unit',
                   'occupancy from', 'change of occ', 'fire protection',
                   'parking requirement', 'plans modified', 'cost of',
                   'beauty salon', 'health club', 'plumbing and',
                   'reflected on', 'staircase', 'gymnasium', 'fire escape',
                   'deck at rear', 'construct new', 'off street',
                   'absent the', 'requested variance', 'and again on']:
        if phrase in lower:
            return True
    # Multi-line text (>2 newlines before cleanup)
    if '\n' in a and a.count('\n') > 1:
        return True
    # Starts with year + day-of-week or month (OCR date text)
    if re.match(r'^20\d{2}\s+(?:tue|wed|thu|fri|mon|sat|sun|the|and|january|february|march|april|may|june|july|august|september|october|november|december)\b', lower):
        return True
    # More than 40% garbage words (skip house number)
    words = lower.split()
    # Very short + garbage word: "1 ists", "09 Off Street"
    if len(words) <= 3 and len(a) < 20:
        if any(w in GARBAGE_WORDS for w in words[1:]):
            return True
    if len(words) > 3:
        check = words[1:]  # skip number
        garbage_count = sum(1 for w in check if w in GARBAGE_WORDS)
        if garbage_count > len(check) * 0.4:
            return True
    # Very long with no street suffix
    if len(a) > 100:
        upper = a.upper()
        if not any(f' {s}' in upper or upper.endswith(f' {s}') for s in STREET_SUFFIX_SET):
            return True
    return False


def clean_one_address(addr):
    """Clean a single address string. Returns cleaned string or None if garbage."""
    if not addr or pd.isna(addr):
        return None
    a = str(addr).strip()

    # ── Whitespace / newline normalization ──
    a = a.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    a = re.sub(r'\s+', ' ', a).strip()

    # ── Remove leading OCR artifacts: |, _, -, leading Z before digit, A0→40 ──
    a = re.sub(r'^[|_]\s*', '', a)
    a = re.sub(r'^-(\d)', r'\1', a)       # -4172 → 4172
    a = re.sub(r'^Z(\d)', r'\1', a)       # Z26R → 26R (OCR Z for 2)
    a = re.sub(r'^A(\d)', r'4\1', a)      # A0 → 40 (OCR A for 4)

    # ── Null out clear garbage early ──
    if is_garbage(a):
        # Try to extract a street address from the garbage
        m = STREET_ADDR_RE.search(a)
        if m:
            extracted = m.group(1).strip()
            if len(extracted) > 8 and not is_garbage(extracted):
                a = extracted
            else:
                return None
        else:
            return None

    # ── Normalize dashes ──
    # Em/en-dash with spaces: "1526 — 1530" → "1526-1530"
    a = re.sub(r'(\d)\s*[—–]\s*(\d)', r'\1-\2', a)
    # Spaced hyphens in ranges: "1 - 17" → "1-17"
    a = re.sub(r'(\d)\s+-\s+(\d)', r'\1-\2', a)

    # ── Remove parenthetical text: (the "Property"), (the "Project") ──
    a = re.sub(r'\s*\(.*$', '', a).strip()
    a = re.sub(r'\s*\).*$', '', a).strip()

    # ── Strip ", Ward - XX" / ", Ward XX" / ", 'WfJ.rd 17" suffixes ──
    # Broad pattern catches OCR-mangled "Ward" like "WfJ.rd", "W.rd", etc.
    a = re.sub(r""",?\s*['"]?W[a-zA-Z.]*r?d\.?\s*[-–—]?\s*\d+['"]?\s*$""", '', a, flags=re.IGNORECASE).strip()
    a = re.sub(r',?\s*Ward\s*[-–—]?\s*\d+\s*$', '', a, flags=re.IGNORECASE).strip()

    # ── Strip city/state/zip ──
    # "423 William F. McClellan Highway, East Boston, MA, Parcel ID" →
    # "423 William F. McClellan Highway"
    a = re.sub(r',?\s*Parcel\s*(?:ID|#)?\s*[\d\-]*\s*$', '', a, flags=re.IGNORECASE).strip()
    a = re.sub(r',?\s*Boston\s*,?\s*(?:MA|Massachusetts)?\s*,?\s*(?:0\d{4})?\s*$', '', a, flags=re.IGNORECASE).strip()
    a = re.sub(r',?\s*MA\s*,?\s*(?:0\d{4})?\s*$', '', a, flags=re.IGNORECASE).strip()
    a = re.sub(r',?\s*0\d{4}\s*$', '', a).strip()
    # Neighborhood suffixes: "259 Gold ST South Boston 02127"
    for nbhd in sorted(NEIGHBORHOODS, key=len, reverse=True):
        pat = re.compile(
            rf'(.*?\b(?:{"|".join(STREET_SUFFIX_SET)})\.?)\s+{re.escape(nbhd)}\s*,?\s*(?:MA)?\s*(?:0\d{{4}})?\s*$',
            re.IGNORECASE
        )
        m = pat.match(a)
        if m:
            a = m.group(1).strip()
            break
    # Just "Boston" at end
    a = re.sub(r',?\s+Boston\s*$', '', a, flags=re.IGNORECASE).strip()
    # Trailing ", East" or ", South" etc. (leftover from city strip)
    a = re.sub(r',\s+(?:East|West|South|North)\s*$', '', a, flags=re.IGNORECASE).strip()
    # Just zip at end
    a = re.sub(r'\s+0\d{4}\s*$', '', a).strip()

    # ── Strip "in [neighborhood]" / "in the [neighborhood]" after street ──
    # Also handles "in South Boston" where "South Boston" is the neighborhood
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s*,?\s+in\s+(?:the\s+)?(?:'
        + '|'.join(re.escape(n) for n in sorted(NEIGHBORHOODS, key=len, reverse=True))
        + r')\b.*$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()
    # Also: "NNN Avenue in South" (partial neighborhood — OCR cut off)
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s*,?\s+in\s+(?:the\s+)?(?:South|East|West|North)\s*$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()

    # ── Strip trailing description text ──
    # "NNN Street to erect a...", "NNN Street from Offices to Restaurant"
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s+(?:to\s+(?:erect|construct|demolish|renovate|convert|alter|'
        r'change|install|build|add|extend|maintain|use|raze|subdivide|operate|relocate|'
        r'establish|create|remove|rehab|legalize|combine|correct)\b).*$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s+from\s+\w+.*$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()
    # "into two lots...", "and construction to..."
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s+(?:into\s+|and\s+(?:construction|made)\b).*$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()
    # Generic: street suffix + connector word + text
    a = re.sub(
        r'(\b' + STREET_SUFFIXES + r'\.?)\s+(?:and|the|a|an|for|with|as|that|which|this|said|is|was|will)(?:\s+.*)$',
        r'\1', a, flags=re.IGNORECASE
    ).strip()

    # ── Strip "& PARCELID" ──
    a = re.sub(r'\s*&\s*\d{7,}\b.*$', '', a).strip()

    # ── Fix malformed range: "4-0 Liberty Square" → "4 Liberty Square" ──
    a = re.sub(r'^(\d+)-0\s+', r'\1 ', a)

    # ── Fix spaced number ranges: "7 4 - 76 Rowe St" → "74-76 Rowe St" ──
    m = re.match(r'^(\d)\s(\d+)\s*-\s*(\d+)\s+(.*)', a)
    if m:
        a = f"{m.group(1)}{m.group(2)}-{m.group(3)} {m.group(4)}"

    # ── Fix OCR slash in number: "2/0 Norfolk" → "20 Norfolk" ──
    a = re.sub(r'^(\d)/(\d)', r'\1\2', a)

    # ── Remove leading period: "59A. Strathmore" → "59A Strathmore" ──
    a = re.sub(r'^(\d+[A-Z]?)\.(\s)', r'\1\2', a)

    # ── Strip trailing punctuation/quotes ──
    a = re.sub(r"""[.;:,'""\u201c\u201d\u2018\u2019()\[\]\\]+\s*$""", '', a).strip()

    # ── Final whitespace cleanup ──
    a = re.sub(r'\s+', ' ', a).strip()

    # ── Final validation ──
    if not a or len(a) < 5:
        return None
    if not re.match(r'^\d', a):
        return None
    if not re.search(r'[a-zA-Z]', a):
        return None
    # Still too long after all cleaning — last resort extraction
    if len(a) > 80:
        m = STREET_ADDR_RE.match(a)
        if m:
            a = m.group(1).strip()
        else:
            # Truncate at first delimiter after 10 chars
            for delim in [',', '.', ';', ' from ', ' for ', ' per ']:
                idx = a.find(delim, 10)
                if 10 < idx < 80:
                    a = a[:idx].strip()
                    break
            else:
                a = a[:80].rsplit(' ', 1)[0].strip()
    # Final garbage check on result
    if is_garbage(a):
        return None

    return a


def main():
    parser = argparse.ArgumentParser(description='Clean address_clean column in ZBA dataset')
    parser.add_argument('--apply', action='store_true', help='Save changes to zba_cases_cleaned.csv')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'zba_cases_cleaned.csv')

    print("=" * 70)
    print("  PermitIQ Address Cleaner")
    print("=" * 70)

    # Load
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    original_addrs = df['address_clean'].copy()
    orig_non_null = original_addrs.notna().sum()
    print(f"  address_clean non-null: {orig_non_null}")
    print(f"  address_clean unique:   {original_addrs.nunique()}")

    # Backup before modifying
    if args.apply:
        backup = os.path.join(base_dir, f'zba_cases_cleaned.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        print(f"\n  Creating backup: {os.path.basename(backup)}")
        shutil.copy2(csv_path, backup)

    # Clean
    print("\n  Cleaning addresses...")
    df['address_clean'] = df['address_clean'].apply(clean_one_address)

    # Stats
    new_non_null = df['address_clean'].notna().sum()
    nulled = orig_non_null - new_non_null

    changed_mask = (original_addrs.fillna('__NA__') != df['address_clean'].fillna('__NA__'))
    cleaned_not_nulled = changed_mask & df['address_clean'].notna() & original_addrs.notna()
    cleaned_count = cleaned_not_nulled.sum()
    total_changed = changed_mask.sum()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total rows:                     {len(df)}")
    print(f"  Addresses cleaned (modified):   {cleaned_count}")
    print(f"  Addresses nulled (garbage):     {nulled}")
    print(f"  Total changes:                  {total_changed}")
    print(f"  Unchanged:                      {len(df) - total_changed}")
    print(f"  Non-null before: {orig_non_null}  ->  after: {new_non_null}")
    print(f"  Unique addresses after:         {df['address_clean'].nunique()}")

    # Examples of cleaned
    print(f"\n  --- Examples of CLEANED addresses ({cleaned_count} total) ---")
    shown = 0
    for idx in df.loc[cleaned_not_nulled].index:
        old = str(original_addrs.iloc[idx])
        new = df['address_clean'].iloc[idx]
        if len(old) < 120:
            print(f"    BEFORE: [{old}]")
            print(f"    AFTER:  [{new}]")
            print()
            shown += 1
            if shown >= 20:
                break

    # Examples of nulled
    nulled_mask = original_addrs.notna() & df['address_clean'].isna()
    print(f"  --- Examples of NULLED addresses ({nulled} total) ---")
    shown = 0
    for idx in df.loc[nulled_mask].index:
        old = str(original_addrs.iloc[idx])
        if len(old) < 120:
            print(f"    NULLED: [{old}]")
            shown += 1
            if shown >= 15:
                break

    # Save
    if args.apply:
        print(f"\n  Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print(f"  Done. Backup saved.")
    else:
        print(f"\n  DRY RUN -- use --apply to save changes")


if __name__ == '__main__':
    main()
