"""
PermitIQ — Auto-Pull ZBA Hearing Transcripts from YouTube
Discovers new ZBA hearing videos on Boston City TV's YouTube channel,
downloads audio, transcribes with Whisper, and integrates into the dataset.

Run manually:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 auto_pull_transcripts.py

Or set up a cron job to run weekly (Sunday at 3 AM, after auto_update_data):
    crontab -e
    0 3 * * 0 cd ~/Desktop/Boston\ Zoning\ Project && source zoning-env/bin/activate && python3 auto_pull_transcripts.py >> auto_pull_transcripts.log 2>&1

Requires: pip install pytubefix openai-whisper
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

print("=" * 60)
print("  PermitIQ — Auto-Pull ZBA Transcripts")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ========================================
# CONFIG
# ========================================

BASE_DIR = Path(__file__).parent
TRANSCRIPT_DIR = BASE_DIR / "data" / "zba_transcripts"
AUDIO_DIR = BASE_DIR / "data" / "zba_audio"
MANIFEST_FILE = BASE_DIR / "data" / "transcript_manifest.json"

YOUTUBE_SEARCH_TERMS = [
    "Zoning Board of Appeal",
    "ZBA hearing",
    "Zoning Board Appeal Boston",
]

WHISPER_MODEL = "base"

# ========================================
# HELPERS
# ========================================

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


def get_existing_audio():
    """Get set of video IDs that already have audio downloaded."""
    existing = set()
    if AUDIO_DIR.exists():
        for f in AUDIO_DIR.iterdir():
            if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav'):
                existing.add(f.stem)
    return existing


def get_existing_transcripts():
    """Get set of video IDs that already have transcripts."""
    existing = set()
    if TRANSCRIPT_DIR.exists():
        for f in TRANSCRIPT_DIR.iterdir():
            if f.suffix in ('.txt',) and f.stem != 'README':
                existing.add(f.stem)
    return existing


# ========================================
# STEP 1: DISCOVER new YouTube videos
# ========================================

def discover_new_videos():
    """Search YouTube for new ZBA hearing videos not yet in manifest."""
    manifest = load_manifest()
    known_ids = set(manifest.get("youtube_urls", {}).keys())

    print(f"\n📡 Discovering new ZBA videos...")
    print(f"   Known videos: {len(known_ids)}")

    new_urls = {}

    for term in YOUTUBE_SEARCH_TERMS:
        print(f"   Searching: '{term}'...")
        try:
            cmd = [
                sys.executable, "-m", "yt_dlp",
                f"ytsearch30:{term} Boston Zoning Board",
                "--flat-playlist", "--dump-json",
                "--no-download", "--quiet",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    try:
                        info = json.loads(line)
                        title = info.get("title", "")
                        video_id = info.get("id", "")
                        title_lower = title.lower()

                        if video_id and video_id not in known_ids:
                            if any(kw in title_lower for kw in [
                                "zoning board", "zba", "zoning appeal",
                                "board of appeal", "zoning hearing"
                            ]):
                                full_url = f"https://www.youtube.com/watch?v={video_id}"
                                new_urls[video_id] = {
                                    "url": full_url,
                                    "title": title,
                                    "duration": info.get("duration"),
                                    "upload_date": info.get("upload_date"),
                                }
                                print(f"   🆕 {title[:60]}")
                    except json.JSONDecodeError:
                        continue
        except subprocess.TimeoutExpired:
            print(f"   Timeout searching for '{term}'")
        except Exception as e:
            print(f"   Error: {e}")

    if new_urls:
        # Add to manifest
        manifest.setdefault("youtube_urls", {}).update(new_urls)
        manifest["last_discovery"] = datetime.now().isoformat()
        save_manifest(manifest)
        print(f"\n   Found {len(new_urls)} new video(s)")
    else:
        print(f"\n   No new videos found")
        manifest["last_discovery"] = datetime.now().isoformat()
        save_manifest(manifest)

    return new_urls


# ========================================
# STEP 2: DOWNLOAD audio for new videos
# ========================================

def download_new_audio(video_ids=None):
    """Download audio for videos that don't have audio yet."""
    manifest = load_manifest()
    urls = manifest.get("youtube_urls", {})
    existing_audio = get_existing_audio()

    if video_ids:
        to_download = {k: v for k, v in urls.items() if k in video_ids and k not in existing_audio}
    else:
        to_download = {k: v for k, v in urls.items() if k not in existing_audio}

    if not to_download:
        print(f"\n🎵 No new audio to download")
        return []

    print(f"\n🎵 Downloading {len(to_download)} audio file(s)...")

    downloaded = []
    try:
        from pytubefix import YouTube
    except ImportError:
        print("   ❌ pytubefix not installed. Run: pip install pytubefix")
        return []

    for i, (vid_id, info) in enumerate(to_download.items()):
        title = info.get("title", vid_id)
        url = info["url"]
        print(f"   [{i + 1}/{len(to_download)}] {title[:55]}...")

        try:
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                audio_stream = yt.streams.first()

            if audio_stream:
                out_name = f"{vid_id}.mp4"
                audio_stream.download(output_path=str(AUDIO_DIR), filename=out_name)
                actual_path = AUDIO_DIR / out_name

                if actual_path.exists():
                    size_mb = actual_path.stat().st_size / (1024 * 1024)
                    print(f"      ✅ {size_mb:.1f} MB")
                    downloaded.append(vid_id)
                    manifest["youtube_urls"][vid_id]["audio_file"] = str(actual_path)
                    save_manifest(manifest)
                else:
                    print(f"      ❌ File not found after download")
            else:
                print(f"      ❌ No audio stream available")
        except Exception as e:
            print(f"      ❌ {str(e)[:80]}")

        time.sleep(2)  # Be polite to YouTube

    print(f"\n   Downloaded: {len(downloaded)} file(s)")
    return downloaded


# ========================================
# STEP 3: TRANSCRIBE new audio with Whisper
# ========================================

def transcribe_new_audio(video_ids=None):
    """Transcribe audio files that don't have transcripts yet."""
    existing_transcripts = get_existing_transcripts()

    if video_ids:
        audio_files = [AUDIO_DIR / f"{vid}.mp4" for vid in video_ids if vid not in existing_transcripts]
        audio_files = [f for f in audio_files if f.exists()]
    else:
        audio_files = []
        for f in sorted(AUDIO_DIR.iterdir()):
            if f.suffix in ('.m4a', '.mp3', '.opus', '.webm', '.mp4', '.wav'):
                if f.stem not in existing_transcripts:
                    audio_files.append(f)

    if not audio_files:
        print(f"\n📝 No new audio to transcribe")
        return []

    print(f"\n📝 Transcribing {len(audio_files)} file(s) with Whisper ({WHISPER_MODEL})...")

    transcribed = []

    for i, audio_path in enumerate(audio_files):
        vid_id = audio_path.stem
        transcript_path = TRANSCRIPT_DIR / f"{vid_id}.txt"

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"   [{i + 1}/{len(audio_files)}] {vid_id} ({size_mb:.0f} MB)...")

        start = time.time()
        try:
            # Run Whisper in a subprocess to isolate memory
            cmd = [
                sys.executable, "-c",
                f"""
import whisper, sys
model = whisper.load_model("{WHISPER_MODEL}")
result = model.transcribe("{audio_path}", language="en")
with open("{transcript_path}", "w") as f:
    f.write(result["text"])
print(len(result["text"]))
"""
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            elapsed = time.time() - start
            if result.returncode == 0 and transcript_path.exists():
                chars = transcript_path.stat().st_size
                print(f"      ✅ {chars:,} chars in {elapsed / 60:.1f} min")
                transcribed.append(vid_id)
            else:
                err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
                print(f"      ❌ Failed ({elapsed:.0f}s): {err[:80]}")
        except subprocess.TimeoutExpired:
            print(f"      ❌ Timeout (>2 hours)")
        except Exception as e:
            print(f"      ❌ Error: {e}")

    print(f"\n   Transcribed: {len(transcribed)} file(s)")
    return transcribed


# ========================================
# STEP 4: PARSE transcripts for case data
# ========================================

def parse_new_transcripts(video_ids=None):
    """Parse new transcripts to extract BOA case numbers, addresses, decisions."""
    manifest = load_manifest()

    if video_ids:
        transcript_files = [TRANSCRIPT_DIR / f"{vid}.txt" for vid in video_ids]
        transcript_files = [f for f in transcript_files if f.exists()]
    else:
        transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))

    if not transcript_files:
        print(f"\n🔍 No transcripts to parse")
        return

    print(f"\n🔍 Parsing {len(transcript_files)} transcript(s) for case data...")

    total_cases = 0
    for tf in transcript_files:
        text = tf.read_text(errors="replace")

        # Extract BOA case numbers
        boa_pattern = r"BOA[\s\-]*(\d[\d\s\-]{4,8}\d)"
        raw_matches = re.findall(boa_pattern, text, re.I)
        case_numbers = set()
        for raw in raw_matches:
            num = re.sub(r"[^\d]", "", raw)
            if len(num) >= 6:
                case_numbers.add(f"BOA{num}")

        # Extract addresses
        addr_pattern = r"(\d{1,5}(?:\s*-\s*\d{1,5})?\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Place|Pl|Court|Ct|Boulevard|Blvd|Terrace|Ter)))"
        addresses = set(m.strip() for m in re.findall(addr_pattern, text, re.I))

        total_cases += len(case_numbers)
        if case_numbers:
            print(f"   {tf.name}: {len(case_numbers)} cases, {len(addresses)} addresses")

    print(f"\n   Total cases extracted: {total_cases}")


# ========================================
# MAIN — FULL AUTO-PULL PIPELINE
# ========================================

ensure_dirs()

# Step 1: Find new videos
new_videos = discover_new_videos()
new_ids = list(new_videos.keys()) if new_videos else None

# Step 2: Download audio (new videos + any previously undowned)
downloaded_ids = download_new_audio(video_ids=None)  # Process all undownloaded

# Step 3: Transcribe (new downloads + any previously untranscribed)
transcribed_ids = transcribe_new_audio(video_ids=None)  # Process all untranscribed

# Step 4: Parse new transcripts
if transcribed_ids:
    parse_new_transcripts(video_ids=transcribed_ids)

# ========================================
# SUMMARY
# ========================================

existing_audio = get_existing_audio()
existing_transcripts = get_existing_transcripts()
manifest = load_manifest()
total_videos = len(manifest.get("youtube_urls", {}))

print(f"\n{'=' * 60}")
print(f"  AUTO-PULL COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
print(f"  New videos discovered:  {len(new_videos) if new_videos else 0}")
print(f"  Audio downloaded:       {len(downloaded_ids)}")
print(f"  Transcripts created:    {len(transcribed_ids)}")
print(f"  ---")
print(f"  Total known videos:     {total_videos}")
print(f"  Total audio files:      {len(existing_audio)}")
print(f"  Total transcripts:      {len(existing_transcripts)}")
print(f"  Coverage:               {len(existing_transcripts)}/{total_videos} ({len(existing_transcripts) / total_videos * 100:.0f}%)" if total_videos else "")
print(f"{'=' * 60}")

# Suggest next steps if there are untranscribed files
untranscribed = existing_audio - existing_transcripts
if untranscribed:
    print(f"\n  ⚠️  {len(untranscribed)} audio files still need transcription")
    print(f"  Run again or use: python3 zba_transcript_pipeline.py transcribe")
