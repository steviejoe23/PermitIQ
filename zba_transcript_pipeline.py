"""
PermitIQ — ZBA Transcript Pipeline
Downloads ZBA hearing recordings from YouTube, transcribes with Whisper,
matches transcripts to PDF decision files, and validates OCR data accuracy.

Usage:
    python3 zba_transcript_pipeline.py discover      # Find YouTube ZBA hearing URLs
    python3 zba_transcript_pipeline.py download       # Download audio from discovered URLs
    python3 zba_transcript_pipeline.py transcribe     # Transcribe downloaded audio with Whisper
    python3 zba_transcript_pipeline.py match          # Match transcripts to PDF cases
    python3 zba_transcript_pipeline.py validate       # Cross-validate OCR vs transcript data
    python3 zba_transcript_pipeline.py status         # Show pipeline coverage status
    python3 zba_transcript_pipeline.py all            # Run full pipeline

Requires: pip install yt-dlp openai-whisper
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).parent
TRANSCRIPT_DIR = BASE_DIR / "data" / "zba_transcripts"
AUDIO_DIR = BASE_DIR / "data" / "zba_audio"
MANIFEST_FILE = BASE_DIR / "data" / "transcript_manifest.json"
VALIDATION_REPORT = BASE_DIR / "data" / "transcript_validation_report.json"
CLEANED_CSV = BASE_DIR / "zba_cases_cleaned.csv"
TRACKER_CSV = BASE_DIR / "zba_tracker.csv"

YOUTUBE_CHANNEL = "@BostonCityTV"
YOUTUBE_SEARCH_TERMS = [
    "Zoning Board of Appeal",
    "ZBA hearing",
    "Zoning Board Appeal Boston",
]

# Whisper model — "base" is fast, "small" is more accurate, "medium" for best quality
WHISPER_MODEL = "base"

# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"hearings": {}, "youtube_urls": {}, "last_discovery": None}


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def normalize_case_number(cn):
    """Normalize case number to alphanumeric uppercase: BOA1234567"""
    return "".join(c for c in str(cn) if c.isalnum()).upper()


def parse_date_from_filename(filename):
    """Extract date from PDF filename like 'Decision Detail Filed March 20th, 2026.pdf'"""
    patterns = [
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:st|nd|rd|th)?,?\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, filename, re.I)
        if m:
            month, day, year = m.group(1), m.group(2), m.group(3)
            try:
                return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def load_hearing_map():
    """Build mapping: hearing_date -> [{case_number, address, decision, source_pdf}]"""
    # Load tracker for hearing dates
    tracker_hearing = {}
    tracker_addr = {}
    tracker_decision = {}
    with open(TRACKER_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            for col in ["parent_apno", "boa_apno"]:
                cn = row.get(col, "")
                if cn and str(cn).strip():
                    norm = normalize_case_number(cn)
                    tracker_hearing[norm] = row.get("hearing_date", "").strip()
                    tracker_addr[norm] = row.get("address", "").strip()
                    tracker_decision[norm] = row.get("decision", "").strip()

    # Load cleaned dataset for OCR data
    hearing_cases = defaultdict(list)
    with open(CLEANED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            cn = row.get("case_number", "").strip()
            if not cn:
                continue
            norm = normalize_case_number(cn)
            hd = tracker_hearing.get(norm, "")
            if hd:
                hearing_cases[hd].append({
                    "case_number": cn,
                    "norm_case": norm,
                    "address_ocr": row.get("address", ""),
                    "address_tracker": tracker_addr.get(norm, ""),
                    "decision_ocr": row.get("decision", ""),
                    "decision_tracker": tracker_decision.get(norm, ""),
                    "source_pdf": row.get("source_pdf", ""),
                    "variance_types": row.get("variance_types", ""),
                })

    return hearing_cases


# ============================================================
# STEP 1: DISCOVER YouTube ZBA Hearing URLs
# ============================================================

def discover_youtube_urls():
    """Search YouTube for Boston ZBA hearing recordings."""
    ensure_dirs()
    manifest = load_manifest()

    print("=" * 60)
    print("  Step 1: Discovering ZBA Hearing Videos on YouTube")
    print("=" * 60)

    all_urls = {}

    for term in YOUTUBE_SEARCH_TERMS:
        search_query = f"{term} site:youtube.com"
        print(f"\n  Searching: '{term}'...")

        try:
            # Use yt-dlp to search YouTube
            cmd = [
                sys.executable, "-m", "yt_dlp",
                f"ytsearch50:{term} Boston Zoning Board",
                "--flat-playlist", "--dump-json",
                "--no-download", "--quiet",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    try:
                        info = json.loads(line)
                        title = info.get("title", "")
                        url = info.get("url", "") or info.get("webpage_url", "")
                        video_id = info.get("id", "")

                        # Filter for ZBA-related content
                        title_lower = title.lower()
                        if any(kw in title_lower for kw in [
                            "zoning board", "zba", "zoning appeal",
                            "board of appeal", "zoning hearing"
                        ]):
                            if video_id:
                                full_url = f"https://www.youtube.com/watch?v={video_id}"
                                all_urls[video_id] = {
                                    "url": full_url,
                                    "title": title,
                                    "duration": info.get("duration"),
                                    "upload_date": info.get("upload_date"),
                                }
                                print(f"    Found: {title[:70]}")
                    except json.JSONDecodeError:
                        continue
        except subprocess.TimeoutExpired:
            print(f"    Timeout searching for '{term}'")
        except Exception as e:
            print(f"    Error: {e}")

    # Also search the Boston City TV channel directly
    print(f"\n  Searching channel: {YOUTUBE_CHANNEL}...")
    try:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            f"https://www.youtube.com/{YOUTUBE_CHANNEL}/search?query=zoning+board+appeal",
            "--flat-playlist", "--dump-json",
            "--no-download", "--quiet",
            "--playlist-end", "100",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split("\n"):
                try:
                    info = json.loads(line)
                    title = info.get("title", "")
                    video_id = info.get("id", "")
                    if video_id:
                        full_url = f"https://www.youtube.com/watch?v={video_id}"
                        all_urls[video_id] = {
                            "url": full_url,
                            "title": title,
                            "duration": info.get("duration"),
                            "upload_date": info.get("upload_date"),
                        }
                        print(f"    Found: {title[:70]}")
                except json.JSONDecodeError:
                    continue
    except subprocess.TimeoutExpired:
        print("    Timeout searching channel")
    except Exception as e:
        print(f"    Error: {e}")

    manifest["youtube_urls"] = all_urls
    manifest["last_discovery"] = datetime.now().isoformat()
    save_manifest(manifest)

    print(f"\n  Total ZBA videos found: {len(all_urls)}")
    print(f"  Manifest saved: {MANIFEST_FILE}")

    # Try to match videos to hearing dates by parsing titles
    matched = match_videos_to_dates(all_urls)
    manifest["video_date_map"] = matched
    save_manifest(manifest)

    return all_urls


def match_videos_to_dates(urls):
    """Try to extract hearing dates from video titles."""
    date_patterns = [
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    ]

    matched = {}
    for vid_id, info in urls.items():
        title = info.get("title", "")
        upload_date = info.get("upload_date", "")

        # Try title patterns
        for pat in date_patterns:
            m = re.search(pat, title, re.I)
            if m:
                groups = m.groups()
                try:
                    if groups[0].isdigit() and len(groups) == 3:
                        if len(groups[2]) == 2:
                            date_str = f"{groups[0]}/{groups[1]}/20{groups[2]}"
                        else:
                            date_str = f"{groups[0]}/{groups[1]}/{groups[2]}"
                        dt = datetime.strptime(date_str, "%m/%d/%Y")
                    else:
                        # Month name format
                        date_str = " ".join(groups)
                        for fmt in ["%B %d %Y", "%d %B %Y"]:
                            try:
                                dt = datetime.strptime(date_str.replace(",", ""), fmt)
                                break
                            except ValueError:
                                continue
                    matched[vid_id] = dt.strftime("%Y-%m-%d")
                except (ValueError, UnboundLocalError):
                    pass
                break

        # Fall back to upload date (usually same day or day after hearing)
        if vid_id not in matched and upload_date:
            try:
                dt = datetime.strptime(upload_date, "%Y%m%d")
                matched[vid_id] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return matched


# ============================================================
# STEP 2: DOWNLOAD Audio from YouTube
# ============================================================

def download_audio(max_videos=None):
    """Download audio from discovered YouTube URLs.

    NOTE: YouTube may block yt-dlp with 403 errors (SABR streaming).
    If downloads fail, alternatives:
      1. Use --cookies-from-browser chrome (requires logged-in YouTube)
      2. Manually download via browser to data/zba_audio/
      3. Use 'python3 zba_transcript_pipeline.py import_audio <path>' to add manually downloaded files
    """
    ensure_dirs()
    manifest = load_manifest()
    urls = manifest.get("youtube_urls", {})

    if not urls:
        print("No YouTube URLs found. Run 'discover' first.")
        return

    print("=" * 60)
    print(f"  Step 2: Downloading Audio ({len(urls)} videos)")
    print("  NOTE: If YouTube blocks downloads (403), download manually")
    print(f"  and place files in: {AUDIO_DIR}")
    print("=" * 60)

    downloaded = 0
    skipped = 0

    items = list(urls.items())
    if max_videos:
        items = items[:max_videos]

    for vid_id, info in items:
        # Check for any audio format
        audio_path = AUDIO_DIR / f"{vid_id}.m4a"
        existing = list(AUDIO_DIR.glob(f"{vid_id}.*"))
        existing = [f for f in existing if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav')]
        if existing:
            skipped += 1
            continue

        url = info["url"]
        title = info.get("title", vid_id)
        print(f"\n  [{downloaded + skipped + 1}/{len(items)}] {title[:60]}...")

        try:
            # Use pytubefix (handles YouTube's anti-bot better than yt-dlp)
            from pytubefix import YouTube as YT
            yt = YT(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                audio_stream = yt.streams.first()
            if audio_stream:
                out_name = f"{vid_id}.mp4"
                audio_stream.download(output_path=str(AUDIO_DIR), filename=out_name)
                actual_path = AUDIO_DIR / out_name
                if actual_path.exists():
                    size_mb = actual_path.stat().st_size / (1024 * 1024)
                    print(f"    Downloaded: {size_mb:.1f} MB ({actual_path.name})")
                    downloaded += 1
                    manifest["youtube_urls"][vid_id]["audio_file"] = str(actual_path)
                    save_manifest(manifest)
                else:
                    print(f"    Failed: file not found after download")
            else:
                print(f"    Failed: no audio stream available")
        except subprocess.TimeoutExpired:
            print("    Timeout (>10 min)")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n  Downloaded: {downloaded}, Skipped (existing): {skipped}")


# ============================================================
# STEP 3: TRANSCRIBE Audio with Whisper
# ============================================================

def transcribe_audio():
    """Transcribe downloaded audio files with Whisper."""
    ensure_dirs()
    manifest = load_manifest()

    audio_files = sorted(
        f for f in AUDIO_DIR.iterdir()
        if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav')
    )
    if not audio_files:
        print("No audio files found. Run 'download' first.")
        return

    print("=" * 60)
    print(f"  Step 3: Transcribing {len(audio_files)} Audio Files")
    print(f"  Model: {WHISPER_MODEL}")
    print("=" * 60)

    transcribed = 0
    skipped = 0

    for audio_path in audio_files:
        vid_id = audio_path.stem
        transcript_path = TRANSCRIPT_DIR / f"{vid_id}.txt"

        if transcript_path.exists() and transcript_path.stat().st_size > 100:
            skipped += 1
            continue

        print(f"\n  [{transcribed + skipped + 1}/{len(audio_files)}] {vid_id}...")
        start = time.time()

        try:
            cmd = [
                sys.executable, "-c",
                f"""
import whisper
model = whisper.load_model("{WHISPER_MODEL}")
result = model.transcribe("{audio_path}", language="en")
with open("{transcript_path}", "w") as f:
    f.write(result["text"])
print(f"OK: {{len(result['text'])}} chars")
"""
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            elapsed = time.time() - start
            if result.returncode == 0 and transcript_path.exists():
                chars = transcript_path.stat().st_size
                print(f"    Transcribed: {chars:,} chars in {elapsed:.0f}s")
                transcribed += 1

                # Update manifest
                if vid_id in manifest.get("youtube_urls", {}):
                    manifest["youtube_urls"][vid_id]["transcript_file"] = str(transcript_path)
                    save_manifest(manifest)
            else:
                print(f"    Failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("    Timeout (>1 hour)")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n  Transcribed: {transcribed}, Skipped (existing): {skipped}")


# ============================================================
# STEP 4: MATCH Transcripts to PDF Cases
# ============================================================

def extract_cases_from_transcript(text):
    """Extract BOA case numbers and associated data from transcript text."""
    cases = []

    # Find all BOA case number patterns — capture full number including dashes/spaces
    boa_patterns = [
        r"BOA[\s\-]*(\d[\d\s\-]{4,8}\d)",  # e.g. BOA 180-5958, BOA 1800750, BOA-1716753
        r"B\.?O\.?A\.?\s*[\-#]?\s*(\d[\d\s\-]{4,8}\d)",
        r"case\s+(?:number\s+)?(?:BOA[\s\-]*)?(\d{5,7})",
    ]

    found_numbers = set()
    for pat in boa_patterns:
        for m in re.finditer(pat, text, re.I):
            raw = m.group(1)
            # Strip to digits only
            num = re.sub(r"[^\d]", "", raw)
            if len(num) >= 6:
                found_numbers.add(num)

    # For each case number, try to extract surrounding context
    for num in found_numbers:
        norm = f"BOA{num}"

        # Build regex that matches this number with any spacing/dashes
        # e.g. num=1805958 should match "BOA 180-5958" or "BOA1805958" or "BOA-1805958"
        num_flex = r"[\s\-]*".join(list(num))
        pattern = re.compile(rf"BOA[\s\-]*{num_flex}", re.I)

        for m in pattern.finditer(text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 800)
            context = text[start:end]

            # Try to extract address — handle markdown "### BOA xxx | 123 Main Street, Neighborhood"
            address = ""
            addr_match = re.search(r"\|\s*(.+?)(?:\n|$)", context)
            if addr_match:
                address = addr_match.group(1).strip().rstrip(",")
            if not address:
                addr_match = re.search(r"(\d{1,5}(?:\s*-\s*\d{1,5})?\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Place|Pl|Court|Ct|Boulevard|Blvd|Terrace|Ter)))", context, re.I)
                address = addr_match.group(1).strip() if addr_match else ""

            # Try to extract decision
            decision = ""
            if re.search(r"\*\*Decision:\*\*\s*APPROVED", context, re.I):
                decision = "APPROVED"
            elif re.search(r"\*\*Decision:\*\*\s*DENIED", context, re.I):
                decision = "DENIED"
            elif re.search(r"\b(?:approved|granted|approval)\b", context, re.I):
                decision = "APPROVED"
            elif re.search(r"\b(?:denied|refused|rejected|denial)\b", context, re.I):
                decision = "DENIED"

            # Extract variances if present
            variances = ""
            var_match = re.search(r"\*\*Variances?:\*\*\s*(.+?)(?:\n|$)", context)
            if var_match:
                variances = var_match.group(1).strip()

            cases.append({
                "case_number": norm,
                "address": address,
                "decision": decision,
                "variances": variances,
                "context": context[:300],
            })
            break  # Only need one match per case number

    return cases


def match_transcripts():
    """Match transcript data to PDF case data for validation."""
    ensure_dirs()
    manifest = load_manifest()
    hearing_map = load_hearing_map()

    # Also load tracker for cases that don't have PDFs yet
    tracker_data = {}
    with open(TRACKER_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            for col in ["parent_apno", "boa_apno"]:
                cn = row.get(col, "")
                if cn and str(cn).strip():
                    norm = normalize_case_number(cn)
                    tracker_data[norm] = {
                        "address": row.get("address", ""),
                        "decision": row.get("decision", ""),
                        "hearing_date": row.get("hearing_date", ""),
                        "status": row.get("status", ""),
                    }

    print("=" * 60)
    print("  Step 4: Matching Transcripts to PDF Cases")
    print("=" * 60)

    # Load all transcripts
    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt")) + sorted(TRANSCRIPT_DIR.glob("*.md"))

    if not transcript_files:
        print("No transcript files found. Run 'transcribe' first.")
        return

    all_matches = []
    total_transcript_cases = 0
    total_matched = 0
    total_tracker_only = 0

    for tf in transcript_files:
        text = tf.read_text(errors="replace")
        transcript_cases = extract_cases_from_transcript(text)
        total_transcript_cases += len(transcript_cases)

        print(f"\n  {tf.name}: {len(transcript_cases)} cases found")

        for tc in transcript_cases:
            norm = normalize_case_number(tc["case_number"])

            # Find this case in the hearing map (PDF-backed cases)
            matched_hearing = None
            matched_pdf_case = None
            for hd, cases in hearing_map.items():
                for case in cases:
                    if case["norm_case"] == norm:
                        matched_hearing = hd
                        matched_pdf_case = case
                        break
                if matched_pdf_case:
                    break

            match_record = {
                "transcript_file": tf.name,
                "transcript_case": tc["case_number"],
                "transcript_address": tc.get("address", ""),
                "transcript_decision": tc.get("decision", ""),
                "transcript_variances": tc.get("variances", ""),
                "matched": matched_pdf_case is not None,
            }

            if matched_pdf_case:
                total_matched += 1
                match_record.update({
                    "hearing_date": matched_hearing,
                    "pdf_case_number": matched_pdf_case["case_number"],
                    "pdf_address_ocr": matched_pdf_case["address_ocr"],
                    "pdf_address_tracker": matched_pdf_case["address_tracker"],
                    "pdf_decision": matched_pdf_case["decision_ocr"],
                    "source_pdf": matched_pdf_case["source_pdf"],
                    "variance_types": matched_pdf_case.get("variance_types", ""),
                })

                # Check if data matches
                addr_match = _fuzzy_address_match(
                    tc.get("address", ""),
                    matched_pdf_case["address_tracker"]
                )
                dec_match = _decision_match(
                    tc.get("decision", ""),
                    matched_pdf_case["decision_ocr"]
                )
                match_record["address_validated"] = addr_match
                match_record["decision_validated"] = dec_match

                status = "MATCH" if addr_match and dec_match else "PARTIAL" if addr_match or dec_match else "MISMATCH"
                print(f"    {tc['case_number']}: {status} — PDF (addr={'OK' if addr_match else 'X'}, dec={'OK' if dec_match else 'X'})")

            elif norm in tracker_data:
                # Case exists in tracker but no PDF decision yet
                total_tracker_only += 1
                tr = tracker_data[norm]
                match_record["matched"] = True
                match_record["match_source"] = "tracker_only"
                match_record["hearing_date"] = tr["hearing_date"]
                match_record["tracker_address"] = tr["address"]
                match_record["tracker_decision"] = tr["decision"]
                match_record["tracker_status"] = tr["status"]

                addr_match = _fuzzy_address_match(tc.get("address", ""), tr["address"])
                dec_match = _decision_match(tc.get("decision", ""), tr["decision"])
                match_record["address_validated"] = addr_match
                match_record["decision_validated"] = dec_match

                status = "TRACKER" if addr_match else "TRACKER (addr X)"
                print(f"    {tc['case_number']}: {status} — no PDF yet (tracker: {tr['status']})")
            else:
                print(f"    {tc['case_number']}: NOT FOUND in dataset or tracker")

            all_matches.append(match_record)

    # Save results
    total_any_match = total_matched + total_tracker_only
    report = {
        "generated": datetime.now().isoformat(),
        "total_transcript_cases": total_transcript_cases,
        "total_matched_to_pdf": total_matched,
        "total_matched_tracker_only": total_tracker_only,
        "total_matched_any": total_any_match,
        "match_rate": round(total_any_match / total_transcript_cases, 3) if total_transcript_cases > 0 else 0,
        "matches": all_matches,
    }

    with open(VALIDATION_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Transcript cases:    {total_transcript_cases}")
    print(f"  Matched to PDFs:     {total_matched}")
    print(f"  Matched tracker only: {total_tracker_only} (PDF decisions not yet filed)")
    print(f"  Total matched:       {total_any_match}")
    print(f"  Match rate:          {report['match_rate']:.1%}")
    print(f"  Report saved:        {VALIDATION_REPORT}")


def _normalize_address(addr):
    """Normalize address for comparison."""
    a = addr.lower().strip()
    # Remove zip codes
    a = re.sub(r"\b\d{5}(-\d{4})?\b", "", a)
    # Normalize street suffixes
    replacements = {
        " street": " st", " st.": " st", " avenue": " av", " ave": " av", " ave.": " av",
        " road": " rd", " rd.": " rd", " drive": " dr", " dr.": " dr",
        " lane": " ln", " ln.": " ln", " place": " pl", " pl.": " pl",
        " court": " ct", " ct.": " ct", " boulevard": " blvd", " blvd.": " blvd",
        " terrace": " ter", " ter.": " ter", " circle": " cir", " cir.": " cir",
    }
    for old, new in replacements.items():
        a = a.replace(old, new)
    # Normalize "to" ranges: "81 to 85A" -> "81-85a"
    a = re.sub(r"(\d+)\s+to\s+(\d+)", r"\1-\2", a)
    # Remove neighborhood names and extra words
    a = re.sub(r"\b(boston|allston|brighton|dorchester|roxbury|hyde park|jamaica plain|mattapan|roslindale|west roxbury|south boston|charlestown|east boston|south end|back bay|beacon hill|fenway|mission hill|north end)\b", "", a)
    # Strip non-alphanumeric except dash
    a = re.sub(r"[^a-z0-9\-]", "", a)
    return a

def _fuzzy_address_match(addr1, addr2):
    """Check if two addresses are likely the same."""
    if not addr1 or not addr2:
        return False
    a1 = _normalize_address(addr1)
    a2 = _normalize_address(addr2)
    if not a1 or not a2:
        return False
    # Extract street number for quick check
    num1 = re.match(r"(\d+)", a1)
    num2 = re.match(r"(\d+)", a2)
    if num1 and num2 and num1.group(1) != num2.group(1):
        # Different street numbers — check if they overlap in a range
        n1, n2 = int(num1.group(1)), int(num2.group(1))
        if abs(n1 - n2) > 10:
            return False
    # Check if one contains the other or high overlap
    return a1 in a2 or a2 in a1 or _jaccard(a1, a2) > 0.5


def _jaccard(s1, s2):
    """Jaccard similarity of character bigrams."""
    bg1 = set(s1[i:i + 2] for i in range(len(s1) - 1))
    bg2 = set(s2[i:i + 2] for i in range(len(s2) - 1))
    if not bg1 or not bg2:
        return 0
    return len(bg1 & bg2) / len(bg1 | bg2)


def _decision_match(dec1, dec2):
    """Check if two decisions match."""
    if not dec1 or not dec2:
        return False
    d1 = dec1.upper().strip()
    d2 = dec2.upper().strip()
    # Normalize: GRANTED = APPROVED
    approvals = {"GRANTED", "APPROVED", "GRANT"}
    denials = {"DENIED", "REFUSED", "REJECTED", "DENY"}
    return (d1 in approvals and d2 in approvals) or (d1 in denials and d2 in denials)


# ============================================================
# STEP 5: VALIDATE — Cross-Check OCR vs Transcript
# ============================================================

def validate():
    """Generate validation report comparing OCR data accuracy against transcripts."""
    if not VALIDATION_REPORT.exists():
        print("No validation report found. Run 'match' first.")
        return

    with open(VALIDATION_REPORT) as f:
        report = json.load(f)

    matches = report.get("matches", [])
    matched = [m for m in matches if m.get("matched")]

    print("=" * 60)
    print("  Step 5: Data Validation Report")
    print("=" * 60)

    if not matched:
        print("  No matched cases to validate.")
        return

    addr_ok = sum(1 for m in matched if m.get("address_validated"))
    dec_ok = sum(1 for m in matched if m.get("decision_validated"))
    both_ok = sum(1 for m in matched if m.get("address_validated") and m.get("decision_validated"))

    print(f"\n  Matched cases:       {len(matched)}")
    print(f"  Address validated:   {addr_ok}/{len(matched)} ({addr_ok / len(matched):.0%})")
    print(f"  Decision validated:  {dec_ok}/{len(matched)} ({dec_ok / len(matched):.0%})")
    print(f"  Both validated:      {both_ok}/{len(matched)} ({both_ok / len(matched):.0%})")

    # Show mismatches
    mismatches = [m for m in matched if not m.get("address_validated") or not m.get("decision_validated")]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for m in mismatches[:20]:
            issues = []
            if not m.get("address_validated"):
                issues.append(f"addr: '{m.get('transcript_address', '')}' vs OCR '{m.get('pdf_address_ocr', '')}'")
            if not m.get("decision_validated"):
                issues.append(f"dec: '{m.get('transcript_decision', '')}' vs OCR '{m.get('pdf_decision', '')}'")
            print(f"    {m['transcript_case']}: {'; '.join(issues)}")

    # Coverage stats
    hearing_map = load_hearing_map()
    total_hearings = len(hearing_map)
    total_cases = sum(len(v) for v in hearing_map.values())
    transcript_hearings = set()
    for m in matched:
        hd = m.get("hearing_date")
        if hd:
            transcript_hearings.add(hd)

    print(f"\n  Pipeline Coverage:")
    print(f"    Total hearing dates:     {total_hearings}")
    print(f"    Hearings with transcript: {len(transcript_hearings)} ({len(transcript_hearings) / total_hearings:.0%})")
    print(f"    Total cases in dataset:  {total_cases}")
    print(f"    Cases with transcript:   {len(matched)} ({len(matched) / total_cases:.1%})")

    # Show which source PDFs are covered
    covered_pdfs = set()
    for m in matched:
        sp = m.get("source_pdf", "")
        if sp:
            covered_pdfs.add(sp)

    all_pdfs = set()
    with open(CLEANED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            sp = row.get("source_pdf", "")
            if sp and sp != "zba_tracker":
                all_pdfs.add(sp)

    print(f"    Total source PDFs:       {len(all_pdfs)}")
    print(f"    PDFs with transcript:    {len(covered_pdfs)} ({len(covered_pdfs) / len(all_pdfs):.0%})" if all_pdfs else "")


# ============================================================
# STATUS — Show pipeline coverage
# ============================================================

def show_status():
    """Show current state of the transcript pipeline."""
    ensure_dirs()
    manifest = load_manifest()
    hearing_map = load_hearing_map()

    print("=" * 60)
    print("  PermitIQ — Transcript Pipeline Status")
    print("=" * 60)

    # Dataset stats
    total_hearings = len(hearing_map)
    total_cases = sum(len(v) for v in hearing_map.values())
    print(f"\n  Dataset:")
    print(f"    Hearing dates:     {total_hearings}")
    print(f"    Cases with dates:  {total_cases}")

    # YouTube discovery
    urls = manifest.get("youtube_urls", {})
    print(f"\n  YouTube Discovery:")
    print(f"    Videos found:      {len(urls)}")
    print(f"    Last discovery:    {manifest.get('last_discovery', 'never')}")

    # Audio downloads
    audio_files = [f for f in AUDIO_DIR.iterdir() if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav')]
    print(f"\n  Audio Downloads:")
    print(f"    Files downloaded:  {len(audio_files)}")
    if audio_files:
        total_size = sum(f.stat().st_size for f in audio_files) / (1024 * 1024)
        print(f"    Total size:        {total_size:.0f} MB")

    # Transcripts
    txt_files = list(TRANSCRIPT_DIR.glob("*.txt"))
    md_files = list(TRANSCRIPT_DIR.glob("*.md"))
    print(f"\n  Transcripts:")
    print(f"    Text files:        {len(txt_files)}")
    print(f"    Markdown files:    {len(md_files)}")

    # Validation
    if VALIDATION_REPORT.exists():
        with open(VALIDATION_REPORT) as f:
            report = json.load(f)
        print(f"\n  Validation:")
        print(f"    Transcript cases:  {report.get('total_transcript_cases', 0)}")
        print(f"    Matched to PDFs:   {report.get('total_matched_to_pdf', 0)}")
        print(f"    Match rate:        {report.get('match_rate', 0):.1%}")

    # Coverage gap
    all_pdfs = set()
    with open(CLEANED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            sp = row.get("source_pdf", "")
            if sp and sp != "zba_tracker":
                all_pdfs.add(sp)

    print(f"\n  Coverage Gap:")
    print(f"    Source PDFs needing transcripts: {len(all_pdfs)}")
    print(f"    Transcripts available:           {len(txt_files) + len(md_files)}")
    needed = len(all_pdfs) - len(txt_files) - len(md_files)
    print(f"    Still needed:                    {max(0, needed)}")


# ============================================================
# MAIN
# ============================================================

def generate_download_list():
    """Generate a list of YouTube URLs that need downloading for manual download."""
    manifest = load_manifest()
    urls = manifest.get("youtube_urls", {})

    if not urls:
        print("No YouTube URLs found. Run 'discover' first.")
        return

    # Check which already have audio
    downloaded = set()
    if AUDIO_DIR.exists():
        for f in AUDIO_DIR.iterdir():
            if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav'):
                downloaded.add(f.stem)

    dl_list = BASE_DIR / "data" / "youtube_download_list.txt"
    with open(dl_list, "w") as f:
        f.write("# ZBA Hearing YouTube URLs for Manual Download\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Save downloaded audio to: {AUDIO_DIR}\n")
        f.write(f"# Name files as: <video_id>.mp3 (e.g., TXLYWAmaPAI.mp3)\n\n")

        count = 0
        for vid_id, info in urls.items():
            if vid_id in downloaded:
                continue
            f.write(f"# {info.get('title', 'Unknown')}\n")
            f.write(f"{info['url']}\n\n")
            count += 1

    print(f"Generated download list: {dl_list}")
    print(f"  {count} videos need downloading ({len(downloaded)} already have audio)")
    print(f"\n  To batch-download with yt-dlp:")
    print(f"  yt-dlp -a {dl_list} -f bestaudio -x --audio-format mp3 -o '{AUDIO_DIR}/%(id)s.%(ext)s'")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "discover":
        discover_youtube_urls()
    elif command == "download":
        max_v = int(sys.argv[2]) if len(sys.argv) > 2 else None
        download_audio(max_videos=max_v)
    elif command == "transcribe":
        transcribe_audio()
    elif command == "match":
        match_transcripts()
    elif command == "validate":
        validate()
    elif command == "status":
        show_status()
    elif command == "download_list":
        generate_download_list()
    elif command == "all":
        discover_youtube_urls()
        download_audio()
        transcribe_audio()
        match_transcripts()
        validate()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
