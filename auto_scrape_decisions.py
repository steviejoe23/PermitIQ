"""
PermitIQ — Auto-Scrape ZBA Decision PDFs from boston.gov
Scrapes the decisions page, finds new Google Drive PDF links,
downloads them, then runs the OCR pipeline.

Run manually:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 auto_scrape_decisions.py

Requires: pip install requests beautifulsoup4
"""

import requests
import re
import os
import glob
import time
from datetime import datetime

print("=" * 60)
print("  PermitIQ — Auto-Scrape ZBA Decisions")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ========================================
# CONFIG
# ========================================

DECISIONS_URL = "https://www.boston.gov/departments/inspectional-services/zoning-board-appeal-decisions"
PDF_DIR = "pdfs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def get_existing_pdfs():
    """Get set of existing PDF filenames (lowercased for fuzzy matching)."""
    existing = set()
    for f in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        existing.add(os.path.basename(f).lower())
    return existing


def normalize_date_text(text):
    """Normalize a date string for filename matching.
    'March 20, 2026' -> various possible filename patterns
    """
    text = text.strip()
    # Parse the date
    for fmt in ["%B %d, %Y", "%B %dst, %Y", "%B %dnd, %Y", "%B %drd, %Y", "%B %dth, %Y"]:
        try:
            dt = datetime.strptime(text, fmt)
            return dt
        except ValueError:
            continue

    # Try removing ordinal suffixes
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)
    try:
        return datetime.strptime(cleaned, "%B %d, %Y")
    except ValueError:
        return None


def date_to_filename_patterns(dt):
    """Generate possible filename patterns for a given date."""
    if dt is None:
        return []

    month = dt.strftime("%B").lower()
    day = dt.day
    year = dt.year

    # Add ordinal suffix
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    patterns = [
        f"decision detail filed {month} {day}{suffix}, {year}",
        f"decision details filed {month} {day}{suffix}, {year}",
        f"decision detail filed {month} {day}, {year}",
        f"decision details filed {month} {day}, {year}",
        f"decision details signed {month} {day}, {year}",
        f"decisions signed {month} {day}, {year}",
        # Sometimes month is abbreviated
        f"decision detail filed {month[:3]} {day}{suffix}, {year}",
        f"decision detail filed {month} {day}{suffix} {year}",
    ]
    return [p.lower() for p in patterns]


def already_have_pdf(date_text, existing_pdfs, debug=False):
    """Check if we already have a PDF for this decision date."""
    dt = normalize_date_text(date_text)
    if dt is None:
        if debug:
            print(f"    [debug] Could not parse date: '{date_text}'")
        return False

    patterns = date_to_filename_patterns(dt)
    for existing in existing_pdfs:
        for pattern in patterns:
            # Must match pattern + .pdf to avoid partial matches across years
            if pattern in existing and str(dt.year) in existing:
                if debug:
                    print(f"    [debug] MATCH: '{pattern}' found in '{existing}'")
                return True
    if debug:
        print(f"    [debug] NO MATCH for '{date_text}' (year={dt.year})")
    return False


def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive given its ID."""
    # First try direct download
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    resp = session.get(url, headers=HEADERS, stream=True, timeout=60)

    # Check if we got a virus scan warning (large files)
    if "confirm" in resp.url or b"virus scan" in resp.content[:5000]:
        # Need to confirm download
        confirm_token = None
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break

        if confirm_token:
            url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
            resp = session.get(url, headers=HEADERS, stream=True, timeout=60)

    # Check content type
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type and len(resp.content) < 10000:
        # Might be an error page or confirmation page
        # Try the direct download API
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        resp = session.get(url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)

    if resp.status_code == 200 and len(resp.content) > 1000:
        with open(output_path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"    ✅ Downloaded ({size_kb:.0f} KB)")
        return True
    else:
        print(f"    ❌ Download failed (status={resp.status_code}, size={len(resp.content)})")
        return False


def scrape_decision_links():
    """Scrape the boston.gov decisions page for Google Drive PDF links."""
    print("\nFetching decisions page...")

    try:
        resp = requests.get(DECISIONS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch page: {e}")
        return []

    html = resp.text

    # The page uses dynamic year tabs — the initial HTML may only contain
    # the default year (2026). We use a broad regex to find ALL Google Drive
    # links anywhere in the HTML, regardless of surrounding tags.

    # Pattern 1: Google Drive links with decision text nearby
    # Match href first, then find associated text
    drive_pattern = r'href="https://drive\.google\.com/file/d/([^"\/]+)/[^"]*"'
    drive_ids = re.findall(drive_pattern, html, re.IGNORECASE)

    # Pattern 2: Find the full anchor tags with text
    anchor_pattern = r'<a[^>]*href="https://drive\.google\.com/file/d/([^"\/]+)/[^"]*"[^>]*>(.*?)</a>'
    anchor_matches = re.findall(anchor_pattern, html, re.IGNORECASE | re.DOTALL)

    # Pattern 3: boston.gov hosted PDFs
    pdf_pattern = r'<a[^>]*href="(https://www\.boston\.gov/[^"]*\.pdf)"[^>]*>(.*?)</a>'
    pdf_matches = re.findall(pdf_pattern, html, re.IGNORECASE | re.DOTALL)

    decisions = []
    seen_ids = set()

    for drive_id, raw_text in anchor_matches:
        # Clean HTML from text
        text = re.sub(r'<[^>]+>', '', raw_text).strip()
        if not text or drive_id in seen_ids:
            continue
        seen_ids.add(drive_id)

        # Extract date from text like "Decision Detail Filed March 20, 2026"
        date_text = re.sub(r'Decision\s+Details?\s+(?:Filed|Signed|filed|signed)\s*', '', text, flags=re.IGNORECASE).strip()
        # Remove trailing whitespace/periods
        date_text = date_text.strip('. ')

        if date_text:
            decisions.append({
                "date": date_text,
                "drive_id": drive_id,
                "source": "gdrive",
                "link_text": text
            })

    for url, raw_text in pdf_matches:
        text = re.sub(r'<[^>]+>', '', raw_text).strip()
        if not text:
            continue
        date_text = re.sub(r'Decision\s+(?:details?|Details?):?\s*', '', text, flags=re.IGNORECASE).strip()
        date_text = date_text.strip('. ')
        if date_text:
            decisions.append({
                "date": date_text,
                "url": url,
                "source": "boston_gov",
                "link_text": text
            })

    print(f"  Found {len(decisions)} decision links on page")
    if decisions:
        print(f"  Date range: {decisions[-1]['date']} — {decisions[0]['date']}")
        print(f"  Sample dates: {', '.join(d['date'] for d in decisions[:5])}")
    return decisions


# ========================================
# MAIN
# ========================================

os.makedirs(PDF_DIR, exist_ok=True)
existing_pdfs = get_existing_pdfs()
print(f"\nExisting PDFs: {len(existing_pdfs)}")

# Scrape the page
decisions = scrape_decision_links()

# Find new ones
new_decisions = []
for d in decisions:
    is_existing = already_have_pdf(d["date"], existing_pdfs, debug=True)
    if not is_existing:
        new_decisions.append(d)

print(f"\n🆕 New decisions to download: {len(new_decisions)}")
if new_decisions:
    for d in new_decisions:
        print(f"  • {d['date']}")

# Download new PDFs
downloaded = 0
for d in new_decisions:
    date_text = d["date"]
    dt = normalize_date_text(date_text)

    if dt:
        day = dt.day
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        filename = f"Decision Detail Filed {dt.strftime('%B')} {day}{suffix}, {dt.year}.pdf"
    else:
        filename = f"Decision Detail Filed {date_text}.pdf"

    output_path = os.path.join(PDF_DIR, filename)

    print(f"\n  Downloading: {filename}")

    if d["source"] == "gdrive":
        success = download_from_gdrive(d["drive_id"], output_path)
    elif d["source"] == "boston_gov":
        try:
            resp = requests.get(d["url"], headers=HEADERS, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"    ✅ Downloaded ({len(resp.content)/1024:.0f} KB)")
                success = True
            else:
                print(f"    ❌ Failed")
                success = False
        except Exception as e:
            print(f"    ❌ Error: {e}")
            success = False

    if success:
        downloaded += 1
        time.sleep(1)  # Be polite

# ========================================
# RUN OCR PIPELINE ON NEW PDFs
# ========================================

if downloaded > 0:
    print(f"\n{'='*60}")
    print(f"  Downloaded {downloaded} new PDFs")
    print(f"  Run the OCR pipeline to process them:")
    print(f"{'='*60}")
    print(f"  python3 -c \"")
    print(f"  from zba_pipeline.build_dataset import main")
    print(f"  main()\"")
    print(f"\n  Or run the full re-extraction pipeline:")
    print(f"    python3 reextract_features.py")
    print(f"    python3 integrate_external_data.py")
    print(f"    python3 fuzzy_match_properties.py")
    print(f"    python3 train_model_v2.py")

    # Auto-run the full pipeline (reextract → integrate → fuzzy match)
    try:
        import subprocess

        steps = [
            ("Re-extracting features", ["python3", "reextract_features.py"]),
            ("Integrating external data", ["python3", "integrate_external_data.py"]),
            ("Fuzzy matching properties", ["python3", "fuzzy_match_properties.py"]),
        ]

        for step_name, cmd in steps:
            print(f"\n  {step_name}...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                # Print last few lines of output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:
                    print(f"    {line}")
            else:
                print(f"  ⚠️ {step_name} had issues:")
                print(result.stderr[-300:])
                break

        print(f"\n  ✅ Pipeline completed. Now retrain the model:")
        print(f"    python3 train_model_v2.py")
    except Exception as e:
        print(f"  ❌ Pipeline failed: {e}")
        print(f"  Run steps manually:")
        print(f"    python3 reextract_features.py")
        print(f"    python3 integrate_external_data.py")
        print(f"    python3 fuzzy_match_properties.py")
        print(f"    python3 train_model_v2.py")
else:
    print(f"\n✅ All PDFs are up to date!")

print(f"\n{'='*60}")
print(f"  SCRAPE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total PDFs: {len(existing_pdfs) + downloaded}")
print(f"{'='*60}")
