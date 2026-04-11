#!/usr/bin/env python3
"""Download ZBA hearing transcripts (SRT subtitle files) from Internet Archive.

Internet Archive hosts 339 Boston City TV recordings of ZBA hearings.
Each has auto-generated English subtitles (.en.asr.srt) that we can
download instead of the full video — ~1MB vs ~1GB per hearing.

Usage:
    python scripts/download_zba_transcripts.py
"""

import json
import os
import time
import urllib.request
import urllib.error

# Directory for raw SRT files
SRT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "zba_transcripts", "raw_srt")
os.makedirs(SRT_DIR, exist_ok=True)

# Internet Archive API to get all ZBA hearing identifiers
SEARCH_URL = (
    "https://archive.org/advancedsearch.php?"
    "q=creator%3A%22Boston+City+TV%22+AND+title%3A%22Zoning+Board+of+Appeal%22"
    "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date"
    "&sort%5B%5D=-date&rows=500&output=json"
)

def get_srt_url(identifier: str) -> str:
    """Construct the SRT download URL for an Internet Archive item.

    The file naming convention is: {identifier_without_prefix}.en.asr.srt
    But the actual filename varies, so we check metadata first.
    """
    # The SRT filename is typically the item title with underscores + .en.asr.srt
    # We can get it from the metadata API
    return f"https://archive.org/download/{identifier}"


def get_srt_filename(identifier: str) -> str:
    """Get the actual SRT filename from item metadata."""
    meta_url = f"https://archive.org/metadata/{identifier}/files"
    try:
        req = urllib.request.Request(meta_url, headers={"User-Agent": "PermitIQ/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            for f in data.get("result", []):
                name = f.get("name", "")
                if name.endswith(".en.asr.srt"):
                    return name
                # Fallback to VTT if no SRT
            for f in data.get("result", []):
                name = f.get("name", "")
                if name.endswith(".en.vtt"):
                    return name
    except Exception as e:
        print(f"  [WARN] Could not fetch metadata for {identifier}: {e}")
    return None


def download_file(url: str, dest: str) -> bool:
    """Download a file with retry logic."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PermitIQ/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
            return True
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [FAIL] {url}: {e}")
    return False


def main():
    # Step 1: Get all identifiers
    print("Fetching list of ZBA hearings from Internet Archive...")
    req = urllib.request.Request(SEARCH_URL, headers={"User-Agent": "PermitIQ/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        search_data = json.loads(resp.read().decode())

    docs = search_data.get("response", {}).get("docs", [])
    print(f"Found {len(docs)} ZBA hearing recordings")

    # Step 2: Download SRT for each
    downloaded = 0
    skipped = 0
    failed = 0

    for i, doc in enumerate(docs):
        identifier = doc["identifier"]
        title = doc.get("title", identifier)
        date = doc.get("date", "unknown")[:10]

        # Check if already downloaded
        existing = [f for f in os.listdir(SRT_DIR) if f.startswith(identifier)]
        if existing:
            skipped += 1
            continue

        print(f"[{i+1}/{len(docs)}] {title} ({date})")

        # Get the SRT filename from metadata
        srt_name = get_srt_filename(identifier)
        if not srt_name:
            print(f"  [SKIP] No subtitle file found")
            failed += 1
            continue

        url = f"https://archive.org/download/{identifier}/{srt_name}"
        ext = ".srt" if srt_name.endswith(".srt") else ".vtt"
        dest = os.path.join(SRT_DIR, f"{identifier}{ext}")

        if download_file(url, dest):
            size_kb = os.path.getsize(dest) / 1024
            print(f"  [OK] {size_kb:.0f} KB")
            downloaded += 1
        else:
            failed += 1

        # Be polite to Internet Archive
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\nDone! Downloaded: {downloaded}, Skipped (already had): {skipped}, Failed: {failed}")
    print(f"Total SRT files: {len(os.listdir(SRT_DIR))}")


if __name__ == "__main__":
    main()
