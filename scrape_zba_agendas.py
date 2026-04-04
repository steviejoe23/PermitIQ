"""
PermitIQ — Scrape ZBA Hearing Agendas from boston.gov
Extracts case numbers and specific variances/relief requested from:
  1. Public notice pages (HTML) at boston.gov/public-notices/{id}
  2. Agenda PDFs at content.boston.gov/sites/default/files/file/{year}/{month}/ZBA...pdf

The goal: get pre-hearing variance data for cases where OCR couldn't extract it.

Run:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 scrape_zba_agendas.py
"""

import requests
import re
import os
import csv
import time
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

print("=" * 60)
print("  PermitIQ — Scrape ZBA Hearing Agendas")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ========================================
# CONFIG
# ========================================

BASE_URL = "https://www.boston.gov"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}
OUTPUT_CSV = "zba_agendas.csv"
CHECKPOINT_FILE = "agenda_scrape_checkpoint.json"
RATE_LIMIT_SECONDS = 1.5  # Be polite to boston.gov

session = requests.Session()
session.headers.update(HEADERS)

# ========================================
# KNOWN PUBLIC NOTICE IDs (discovered via search)
# These are ZBA hearing public notices with case details
# ========================================

# Collected from web search and boston.gov exploration
# Format: (notice_id, approximate_hearing_date_str)
KNOWN_NOTICE_IDS = [
    # 2026
    (16571496, "2026-04-07"),
    (16562461, "2026-02-03"),
    (16561661, "2026-01-27"),
    # 2025
    (16489231, "2025-11-06"),  # Advisory
    (16268596, "2025-11-18"),
    (16275096, "2025-11-00"),
    (16369041, "2025-04-17"),  # Advisory
    (16363221, "2025-04-08"),
    (16390426, "2025-00-00"),  # Advisory
    # 2024
    (16237131, "2024-10-08"),
    (16223606, "2024-09-10"),
    (16178781, "2024-07-25"),  # Advisory
    (16171306, "2024-06-13"),  # Advisory
    (16160776, "2024-04-30"),
    (16108411, "2024-04-09"),
    (16106661, "2024-03-26"),
    (16099466, "2024-02-27"),
    (16089736, "2024-02-06"),
    # 2023
    (15952686, "2023-00-00"),
    (15904396, "2023-00-00"),  # Advisory
    (15891771, "2023-00-00"),
    (15829146, "2023-00-00"),
    # Older
    (15772026, "2022-00-00"),
    (15768221, "2022-00-00"),
    (15759091, "2022-00-00"),
    (53196, "2022-00-00"),
    (40401, "2021-00-00"),
    (35631, "2021-00-00"),
    (18256, "2020-00-00"),
    (16871, "2020-00-00"),
    (13675526, "2020-00-00"),
]

# ZBA hearing dates from boston.gov main page (2024-2026)
# We'll use these to try to discover additional public notice pages
ZBA_HEARING_DATES_2024 = [
    "2024-01-09", "2024-01-23", "2024-02-06", "2024-02-27",
    "2024-03-12", "2024-03-26", "2024-04-09", "2024-04-30",
    "2024-05-07", "2024-05-21", "2024-06-04", "2024-06-25",
    "2024-07-16", "2024-07-30", "2024-08-13", "2024-08-27",
    "2024-09-10", "2024-09-24", "2024-10-08", "2024-10-29",
    "2024-11-19", "2024-11-26", "2024-12-03", "2024-12-10",
]

ZBA_HEARING_DATES_2025 = [
    "2025-01-14", "2025-01-28", "2025-02-04", "2025-02-25",
    "2025-03-04", "2025-03-25", "2025-04-08", "2025-04-29",
    "2025-05-06", "2025-05-20", "2025-06-03", "2025-06-24",
    "2025-07-08", "2025-07-29", "2025-08-12", "2025-08-26",
    "2025-09-09", "2025-09-23", "2025-10-07", "2025-10-28",
    "2025-11-18", "2025-11-25", "2025-12-09", "2025-12-16",
]

ZBA_HEARING_DATES_2026 = [
    "2026-01-13", "2026-01-27", "2026-02-03", "2026-02-24",
    "2026-03-10", "2026-03-24", "2026-04-07", "2026-04-28",
    "2026-05-05", "2026-05-19", "2026-06-02", "2026-06-16",
    "2026-07-14", "2026-07-28", "2026-08-11", "2026-08-25",
    "2026-09-08", "2026-09-22", "2026-10-06", "2026-10-27",
    "2026-11-10", "2026-11-17", "2026-12-08", "2026-12-15",
]

# Advisory subcommittee dates (Thursdays)
ZBA_ADVISORY_DATES_2025 = [
    "2025-01-23", "2025-02-13", "2025-03-20", "2025-04-17",
    "2025-05-15", "2025-06-12", "2025-07-24", "2025-08-21",
    "2025-09-18", "2025-10-23", "2025-11-06", "2025-12-04",
]

ZBA_ADVISORY_DATES_2026 = [
    "2026-01-22", "2026-02-12", "2026-03-19", "2026-04-16",
    "2026-05-14", "2026-06-11", "2026-07-23", "2026-08-20",
    "2026-09-17", "2026-10-22", "2026-11-05", "2026-12-03",
]


# ========================================
# VARIANCE NORMALIZATION
# Maps raw text to our standard variance types
# ========================================

VARIANCE_MAP = {
    # Height
    r'building\s*height': 'height',
    r'excessive\s*(?:building\s*)?height': 'height',
    r'height\s*(?:in\s*feet|variance)': 'height',
    r'number\s*of\s*stories': 'height',
    r'excessive\s*(?:number\s*of\s*)?stories': 'height',
    # FAR
    r'floor\s*area\s*ratio': 'FAR',
    r'excessive\s*floor\s*area': 'FAR',
    r'(?:excessive\s*)?f\.?a\.?r': 'FAR',
    # Lot area
    r'(?:insufficient\s*)?lot\s*area': 'lot_area',
    r'(?:insufficient\s*)?additional\s*lot\s*area': 'lot_area',
    r'(?:insufficient\s*)?lot\s*size': 'lot_area',
    # Setbacks (combined)
    r'(?:insufficient\s*)?front\s*yard': 'setbacks',
    r'(?:insufficient\s*)?rear\s*yard': 'setbacks',
    r'(?:insufficient\s*)?side\s*yard': 'setbacks',
    r'(?:minimum\s*)?(?:front|rear|side)\s*yard\s*(?:setback|depth|minimum)': 'setbacks',
    r'projection\s*into\s*required\s*(?:rear|side|front)\s*yard': 'setbacks',
    # Parking
    r'(?:insufficient\s*)?(?:off[\s-]*street\s*)?parking': 'parking',
    r'parking\s*(?:space\s*)?design': 'parking',
    r'restricted\s*parking\s*district': 'parking',
    # Conditional use
    r'conditional\s*(?:use|approval)': 'conditional_use',
    r'conditional\s*(?:cannabis|restaurant|fitness|bank|retail|multifamily|artist)': 'conditional_use',
    # Use
    r'forbidden\s*(?:use|dwelling)': 'use',
    r'(?:business\s*)?use\s*(?:forbidden|conditional)': 'use',
    r'change\s*(?:of\s*)?(?:legal\s*)?occupancy': 'use',
    # Open space
    r'(?:insufficient\s*)?(?:usable\s*)?open\s*space': 'open_space',
    r'(?:insufficient\s*)?permeable\s*open\s*space': 'open_space',
    # Lot dimensions
    r'(?:insufficient\s*)?lot\s*width': 'lot_width',
    r'(?:insufficient\s*)?lot\s*frontage': 'lot_frontage',
    # Nonconforming
    r'(?:extension|reconstruction)\s*(?:of\s*)?(?:a\s*)?non[\s-]*conform': 'nonconforming',
    r'non[\s-]*conform(?:ing|ance)?\s*(?:extension|building|use|structure)': 'nonconforming',
    # Roof / structure
    r'roof\s*structure\s*restriction': 'roof_structure',
    r'roof\s*deck': 'roof_structure',
    # Signs
    r'(?:on[\s-]*premise\s*)?sign': 'signage',
    # Accessory
    r'accessory\s*(?:building|structure|use)': 'accessory',
    # Groundwater
    r'groundwater\s*conservation': 'GCOD',
    r'g\.?c\.?o\.?d': 'GCOD',
    # Flood
    r'flood\s*hazard': 'flood_district',
    # Multiple dwellings
    r'(?:two|multiple)\s*(?:or\s*more\s*)?dwell': 'multiple_dwellings',
    r'multiple\s*dwellings?\s*(?:on\s*)?(?:same\s*)?lot': 'multiple_dwellings',
}


def normalize_case_number(raw):
    """Normalize case number to BOA-XXXXXXX or BOAXXXXXXX format."""
    if not raw:
        return None
    # Remove whitespace
    raw = raw.strip()
    # Extract digits after BOA
    m = re.search(r'BOA[\s-]*(\d+)', raw, re.IGNORECASE)
    if m:
        digits = m.group(1)
        return f"BOA-{digits}"
    return None


def extract_variances_from_text(text):
    """Extract normalized variance types from relief description text."""
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for pattern, var_type in VARIANCE_MAP.items():
        if re.search(pattern, text_lower):
            found.add(var_type)
    return sorted(found)


def load_checkpoint():
    """Load progress checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"scraped_notices": [], "scraped_pdfs": []}


def save_checkpoint(checkpoint):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


# ========================================
# SCRAPE PUBLIC NOTICE PAGES (HTML)
# ========================================

def scrape_public_notice(notice_id):
    """Scrape a single public notice page for ZBA case details.

    Returns list of dicts with case info, or empty list on failure.
    """
    url = f"{BASE_URL}/public-notices/{notice_id}"

    try:
        resp = session.get(url, timeout=30, allow_redirects=True)

        # Check for SAML redirect (older pages require login)
        if 'saml' in resp.url.lower() or 'sso.boston.gov' in resp.url:
            print(f"    [skip] Notice {notice_id} requires authentication (SAML redirect)")
            return []

        if resp.status_code != 200:
            print(f"    [skip] Notice {notice_id} returned status {resp.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"    [error] Notice {notice_id}: {e}")
        return []

    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    # Extract hearing date from page title or body
    hearing_date = extract_hearing_date(soup, html)

    # Extract cases using regex on full HTML text
    # The pages have structured text with BOA numbers followed by addresses and relief descriptions
    text = soup.get_text(separator='\n')

    cases = extract_cases_from_text(text, hearing_date, notice_id)

    return cases


def extract_hearing_date(soup, html):
    """Extract hearing date from the public notice page."""
    # Look for date patterns in text
    text = soup.get_text()

    # Pattern: "Appeals will be heard on January 27, 2026"
    # Pattern: "January 27, 2026, hearing"
    # Pattern: "April 7, 2026 at 9:30"
    date_patterns = [
        r'(?:heard|scheduled)\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\w+\s+\d{1,2},?\s+\d{4})\s*(?:,?\s*hearing|,?\s*at\s+\d)',
        r'(?:Tuesday|Thursday),?\s+(\w+\s+\d{1,2},?\s+\d{4})',
    ]

    for pattern in date_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            date_str = m.group(1).strip().rstrip(',')
            try:
                dt = datetime.strptime(date_str, "%B %d %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
            try:
                dt = datetime.strptime(date_str, "%B %d, %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def extract_cases_from_text(text, hearing_date, source_id):
    """Extract case records from the text content of an agenda page.

    Looks for BOA case numbers and associated relief/variance descriptions.
    """
    cases = []
    lines = text.split('\n')

    # Strategy: find each BOA number, then look at surrounding text for
    # address and relief information

    # First, find all BOA numbers and their positions in the text
    boa_pattern = re.compile(r'BOA[\s-]*(\d{5,8})', re.IGNORECASE)

    # Build chunks: split text around BOA numbers
    # Each chunk = text from one BOA number to the next
    boa_positions = []
    for m in boa_pattern.finditer(text):
        boa_positions.append((m.start(), m.group(0), m.group(1)))

    if not boa_positions:
        return cases

    for i, (pos, raw_boa, digits) in enumerate(boa_positions):
        case_number = f"BOA-{digits}"

        # Get the text chunk for this case (up to next BOA or 2000 chars)
        end_pos = boa_positions[i + 1][0] if i + 1 < len(boa_positions) else pos + 2000
        chunk = text[pos:end_pos]

        # Extract address - typically right after the BOA number
        # Pattern: "at 123 Main Street" or "123 Main Street, Ward X"
        addr_match = re.search(
            r'(?:at\s+)?(\d+[A-Za-z]?(?:\s*[-–]\s*\d+[A-Za-z]?)?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Way|Place|Pl|Lane|Ln|Circle|Ct|Court|Terrace|Ter)\.?)',
            chunk, re.IGNORECASE
        )
        address = addr_match.group(1).strip() if addr_match else None

        # Also try simpler address after comma
        if not address:
            addr_match2 = re.search(
                r'(?:at\s+)?(\d+[A-Za-z]?(?:\s*[-–]\s*\d+[A-Za-z]?)?\s+\S+(?:\s+\S+){0,4}),\s*Ward\s+\d+',
                chunk, re.IGNORECASE
            )
            if addr_match2:
                address = addr_match2.group(1).strip()

        # Extract ward
        ward_match = re.search(r'Ward\s+(\d+)', chunk, re.IGNORECASE)
        ward = ward_match.group(1) if ward_match else None

        # Extract applicant
        applicant_match = re.search(
            r'(?:filed\s+by|applicant:?|by)\s+([A-Z][a-zA-Z\s,\.]+?)(?:\s*[–—-]\s*|\s*(?:seeking|relief|purpose|$))',
            chunk, re.IGNORECASE
        )
        applicant = applicant_match.group(1).strip().rstrip('.,- ') if applicant_match else None

        # Extract relief/variances from the chunk
        variances = extract_variances_from_text(chunk)

        # Also capture the raw relief text for inspection
        # Look for relief descriptions: lines mentioning specific zoning terms
        relief_lines = []
        for line in chunk.split('\n'):
            line = line.strip()
            if not line:
                continue
            line_lower = line.lower()
            # Check if line describes relief
            relief_keywords = [
                'insufficient', 'excessive', 'forbidden', 'conditional',
                'variance', 'relief', 'setback', 'parking', 'floor area',
                'lot area', 'lot width', 'frontage', 'open space',
                'height', 'stories', 'roof structure', 'nonconform',
                'non-conform', 'use forbidden', 'dwelling', 'groundwater',
                'gcod', 'sign', 'accessory', 'projection', 'multiple dwell'
            ]
            if any(kw in line_lower for kw in relief_keywords):
                # Skip very long lines (probably paragraph text, not relief items)
                if len(line) < 200:
                    relief_lines.append(line)

        raw_relief = '; '.join(relief_lines[:15])  # Cap at 15 items

        # Build the variances_requested string
        variances_str = ', '.join(variances) if variances else ''

        cases.append({
            'case_number': case_number,
            'address': address,
            'ward': ward,
            'applicant': applicant,
            'variances_requested': variances_str,
            'hearing_date': hearing_date,
            'raw_text': raw_relief[:2000],  # Cap raw text
            'source': f'public_notice_{source_id}',
        })

    return cases


# ========================================
# SCRAPE AGENDA PDFs
# ========================================

def build_agenda_pdf_urls():
    """Build candidate agenda PDF URLs based on known URL patterns.

    Pattern: content.boston.gov/sites/default/files/file/{year}/{month:02d}/ZBA%20{M.D.YYYY}%20Agenda...pdf
    """
    urls = []

    all_dates = ZBA_HEARING_DATES_2024 + ZBA_HEARING_DATES_2025 + ZBA_HEARING_DATES_2026
    all_dates += ZBA_ADVISORY_DATES_2025 + ZBA_ADVISORY_DATES_2026

    for date_str in all_dates:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        year = dt.year
        month = dt.month
        # Format: M.D.YYYY (e.g., 4.8.2025, 9.18.2025)
        date_part = f"{dt.month}.{dt.day}.{year}"

        # Try several naming patterns observed in the wild
        patterns = [
            f"ZBA%20{date_part}%20Agenda%20w.%20Advisory.pdf",
            f"ZBA%20{date_part}%20Agenda%20w.%20Advisory%20Subcommittee.pdf",
            f"ZBA%20{date_part}%20Subcommittee%20Agenda%20w.%20Advisory.pdf",
            f"ZBA%20{date_part}%20Agenda.pdf",
            f"ZBA%20{date_part}%20Agenda%20w.%20Advisory-Revised.pdf",
            f"ZBA%20{date_part}%20Agenda%20w.%20Advisory-Revised%20(2).pdf",
        ]

        for pattern in patterns:
            url = f"https://content.boston.gov/sites/default/files/file/{year}/{month:02d}/{pattern}"
            urls.append((url, date_str))

    return urls


def scrape_agenda_pdf(url, hearing_date):
    """Download and extract case information from an agenda PDF.

    Uses PyMuPDF (fitz) if available for text extraction.
    """
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        if len(resp.content) < 1000:
            return []
        # Check it's actually a PDF
        if not resp.content[:5] == b'%PDF-':
            return []
    except requests.exceptions.RequestException:
        return []

    # Save temporarily and extract text
    tmp_path = "/tmp/zba_agenda_temp.pdf"
    with open(tmp_path, 'wb') as f:
        f.write(resp.content)

    text = ""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except ImportError:
        print("    [warn] PyMuPDF not available, skipping PDF extraction")
        return []
    except Exception as e:
        print(f"    [error] PDF extraction failed: {e}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text.strip():
        return []

    # Parse using the same text extraction logic
    source = os.path.basename(url).replace('%20', ' ')
    cases = extract_cases_from_text(text, hearing_date, f"pdf_{source}")

    return cases


# ========================================
# DISCOVER ADDITIONAL PUBLIC NOTICES
# ========================================

def discover_notice_ids_from_search():
    """Try to find additional public notice IDs by searching boston.gov."""
    discovered = []

    # Search the public notices listing with various page offsets
    search_urls = [
        f"{BASE_URL}/public-notices?title=zoning+board+of+appeal&page={p}"
        for p in range(5)
    ] + [
        f"{BASE_URL}/public-notices?title=board+of+appeal&page={p}"
        for p in range(5)
    ]

    seen_ids = {nid for nid, _ in KNOWN_NOTICE_IDS}

    for url in search_urls:
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                continue

            # Find notice IDs in the HTML
            ids_found = re.findall(r'/public-notices/(\d+)', resp.text)
            for nid_str in ids_found:
                nid = int(nid_str)
                if nid not in seen_ids:
                    # Check if it's actually a ZBA notice by looking at the link text
                    if re.search(
                        rf'(?:zoning|board\s+of\s+appeal).*?/public-notices/{nid}|/public-notices/{nid}.*?(?:zoning|board\s+of\s+appeal)',
                        resp.text, re.IGNORECASE | re.DOTALL
                    ):
                        seen_ids.add(nid)
                        discovered.append((nid, "unknown"))

            time.sleep(0.5)
        except Exception as e:
            print(f"  [warn] Search failed for {url}: {e}")
            continue

    return discovered


# ========================================
# DEDUPLICATE AND MERGE
# ========================================

def deduplicate_cases(cases):
    """Deduplicate cases by case_number, keeping the one with most variance info."""
    by_case = {}
    for case in cases:
        cn = case['case_number']
        if cn not in by_case:
            by_case[cn] = case
        else:
            # Keep the one with more variance data
            existing_vars = len(by_case[cn].get('variances_requested', ''))
            new_vars = len(case.get('variances_requested', ''))
            if new_vars > existing_vars:
                by_case[cn] = case
            elif new_vars == existing_vars:
                # Keep the one with more raw_text
                if len(case.get('raw_text', '')) > len(by_case[cn].get('raw_text', '')):
                    by_case[cn] = case
    return list(by_case.values())


# ========================================
# MAIN
# ========================================

checkpoint = load_checkpoint()
all_cases = []

# ------------------------------------------
# STEP 1: Scrape known public notice pages
# ------------------------------------------
print(f"\n--- Step 1: Scrape Public Notice Pages ---")
print(f"  Known notice IDs: {len(KNOWN_NOTICE_IDS)}")

# Try to discover additional IDs
print("  Searching for additional notice IDs...")
extra_ids = discover_notice_ids_from_search()
if extra_ids:
    print(f"  Discovered {len(extra_ids)} additional notice IDs")

all_notice_ids = KNOWN_NOTICE_IDS + extra_ids
already_scraped = set(checkpoint.get("scraped_notices", []))

notices_to_scrape = [
    (nid, d) for nid, d in all_notice_ids
    if nid not in already_scraped
]
print(f"  Notices to scrape: {len(notices_to_scrape)} (already done: {len(already_scraped)})")

notice_cases = 0
for i, (notice_id, approx_date) in enumerate(notices_to_scrape):
    print(f"  [{i+1}/{len(notices_to_scrape)}] Notice {notice_id} (approx {approx_date})...", end="", flush=True)

    cases = scrape_public_notice(notice_id)
    if cases:
        all_cases.extend(cases)
        notice_cases += len(cases)
        print(f" {len(cases)} cases")
    else:
        print(f" no cases found")

    checkpoint["scraped_notices"] = list(already_scraped | {notice_id})
    save_checkpoint(checkpoint)

    time.sleep(RATE_LIMIT_SECONDS)

print(f"  Total cases from public notices: {notice_cases}")

# ------------------------------------------
# STEP 2: Scrape agenda PDFs
# ------------------------------------------
print(f"\n--- Step 2: Scrape Agenda PDFs ---")
pdf_urls = build_agenda_pdf_urls()
print(f"  Candidate PDF URLs to try: {len(pdf_urls)}")

already_scraped_pdfs = set(checkpoint.get("scraped_pdfs", []))
pdf_cases_count = 0
pdfs_found = 0

for i, (url, date_str) in enumerate(pdf_urls):
    if url in already_scraped_pdfs:
        continue

    # Only print progress every 10th URL to avoid spam
    if i % 20 == 0:
        print(f"  Checking PDFs... [{i}/{len(pdf_urls)}] ({pdfs_found} found so far)")

    cases = scrape_agenda_pdf(url, date_str)
    if cases:
        all_cases.extend(cases)
        pdf_cases_count += len(cases)
        pdfs_found += 1
        short_name = url.split('/')[-1][:60]
        print(f"    Found: {short_name} -> {len(cases)} cases")

    checkpoint["scraped_pdfs"] = list(already_scraped_pdfs | {url})
    already_scraped_pdfs.add(url)
    save_checkpoint(checkpoint)

    # Only rate-limit on actual hits; HEAD-like quick failures are fine
    if cases:
        time.sleep(RATE_LIMIT_SECONDS)
    else:
        time.sleep(0.3)  # Brief pause even on misses

print(f"  PDFs found: {pdfs_found}")
print(f"  Total cases from PDFs: {pdf_cases_count}")

# ------------------------------------------
# STEP 3: Deduplicate and save
# ------------------------------------------
print(f"\n--- Step 3: Deduplicate and Save ---")
print(f"  Total raw cases: {len(all_cases)}")

# Filter out cases with no case number
all_cases = [c for c in all_cases if c.get('case_number')]
print(f"  Cases with valid BOA number: {len(all_cases)}")

# Deduplicate
all_cases = deduplicate_cases(all_cases)
print(f"  After deduplication: {len(all_cases)}")

# Count cases with variance data
with_variances = sum(1 for c in all_cases if c.get('variances_requested'))
print(f"  Cases with variance data: {with_variances}")

# Save to CSV
fieldnames = ['case_number', 'address', 'ward', 'applicant',
              'variances_requested', 'hearing_date', 'raw_text', 'source']

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for case in sorted(all_cases, key=lambda x: x.get('hearing_date') or ''):
        writer.writerow(case)

print(f"\n  Saved to {OUTPUT_CSV}")

# ------------------------------------------
# STEP 4: Summary statistics
# ------------------------------------------
print(f"\n{'='*60}")
print(f"  SCRAPE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
print(f"  Unique cases found:     {len(all_cases)}")
print(f"  Cases with variances:   {with_variances}")
print(f"  Cases with addresses:   {sum(1 for c in all_cases if c.get('address'))}")
print(f"  Cases with hearing date:{sum(1 for c in all_cases if c.get('hearing_date'))}")

# Show variance type distribution
variance_counts = {}
for case in all_cases:
    for v in (case.get('variances_requested') or '').split(', '):
        v = v.strip()
        if v:
            variance_counts[v] = variance_counts.get(v, 0) + 1

if variance_counts:
    print(f"\n  Variance type distribution:")
    for vtype, count in sorted(variance_counts.items(), key=lambda x: -x[1]):
        print(f"    {vtype:25s} {count:5d}")

# Show hearing date range
dates = [c['hearing_date'] for c in all_cases if c.get('hearing_date')]
if dates:
    print(f"\n  Hearing date range: {min(dates)} to {max(dates)}")

# Check overlap with existing dataset
try:
    import pandas as pd
    existing = pd.read_csv('zba_cases_cleaned.csv')
    if 'case_number' in existing.columns:
        existing_cases = set(existing['case_number'].dropna().astype(str))
        agenda_cases = set(c['case_number'] for c in all_cases)
        overlap = existing_cases & agenda_cases
        new_only = agenda_cases - existing_cases
        print(f"\n  Overlap with existing dataset:")
        print(f"    Cases in both:      {len(overlap)}")
        print(f"    Agenda-only cases:  {len(new_only)}")

        # How many of the overlapping cases are missing variance data in existing?
        if 'variance_types' in existing.columns:
            missing_variances = existing[
                (existing['case_number'].isin(overlap)) &
                (existing['variance_types'].isna() | (existing['variance_types'] == ''))
            ]
            print(f"    Overlap cases missing variances in existing: {len(missing_variances)}")
except Exception as e:
    print(f"\n  [info] Could not check existing dataset: {e}")

# Clean up checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"\n  Cleaned up checkpoint file")

print(f"\n  Next step: Run the merge script to integrate agenda data into the main dataset")
print(f"  python3 merge_agenda_variances.py  (if created)")
