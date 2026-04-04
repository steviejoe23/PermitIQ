"""
PermitIQ — Overnight Full Rebuild
Runs the complete pipeline: OCR all PDFs → build base dataset → clean → retrain.

Run this before bed with Caffeine on:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    python3 overnight_rebuild.py --fresh 2>&1 | tee overnight_log.txt

To RESUME a failed run (skips already-processed PDFs):
    python3 overnight_rebuild.py 2>&1 | tee -a overnight_log.txt
"""

import os
import sys
import glob
import shutil
import subprocess
import time
from datetime import datetime

print("=" * 60)
print("  PermitIQ — Overnight Full Rebuild")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
sys.stdout.flush()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working directory: {os.getcwd()}")
sys.stdout.flush()

# Pass --fresh through to build_dataset if provided
fresh_flag = "--fresh" in sys.argv

# ========================================
# STEP 1: Count PDFs
# ========================================
pdf_files = glob.glob("pdfs/*.pdf")
print(f"\n--- Step 1: Found {len(pdf_files)} PDFs ---")
sys.stdout.flush()

# ========================================
# STEP 2: Re-run OCR on ALL PDFs → zba_cases_dataset.csv
# ========================================
mode = "FRESH (from scratch)" if fresh_flag else "RESUME (checkpoint)"
print(f"\n--- Step 2: Running OCR pipeline [{mode}] ---")
print(f"  Processing {len(pdf_files)} PDFs (output streams live below)...")
print("  " + "-" * 50)
sys.stdout.flush()
start_time = time.time()

cmd = [sys.executable, "-m", "zba_pipeline.build_dataset"]
if fresh_flag:
    cmd.append("--fresh")

result = subprocess.run(
    cmd,
    timeout=None  # No timeout — runs until complete
)

elapsed = time.time() - start_time
print("  " + "-" * 50)
print(f"  OCR completed in {elapsed/60:.1f} minutes (exit code: {result.returncode})")
sys.stdout.flush()

if result.returncode != 0:
    print(f"  ⚠️ OCR had issues (exit code {result.returncode})")
    if not os.path.exists('zba_cases_dataset.csv'):
        print("  ❌ No base dataset produced. Exiting.")
        sys.exit(1)
    else:
        print("  Continuing with existing zba_cases_dataset.csv...")
sys.stdout.flush()

# ========================================
# STEP 3: Run rebuild (dedup + pipeline)
# ========================================
print("\n--- Step 3: Running clean rebuild (dedup + reextract + integrate + fuzzy match) ---")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, "rebuild_dataset.py"],
    timeout=None  # No timeout
)

if result.returncode != 0:
    print(f"  ⚠️ Rebuild had issues (exit code {result.returncode})")
sys.stdout.flush()

# ========================================
# STEP 4: Clean OCR artifacts
# ========================================
print("\n--- Step 4: Cleaning OCR artifacts ---")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, "cleanup_ocr.py"],
    timeout=None
)
if result.returncode != 0:
    print(f"  ⚠️ Cleanup had issues (exit code {result.returncode})")
sys.stdout.flush()

# ========================================
# STEP 4b: OCR Quality Audit
# ========================================
print("\n--- Step 4b: Running OCR quality audit ---")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, "audit_ocr_quality.py"],
    timeout=None
)
sys.stdout.flush()

# ========================================
# STEP 5: Retrain model
# ========================================
print("\n--- Step 5: Retraining model ---")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, "train_model_v2.py"],
    timeout=None  # No timeout
)

if result.returncode != 0:
    print(f"  ⚠️ Training had issues (exit code {result.returncode})")
sys.stdout.flush()

# ========================================
# STEP 6: Copy model to API
# ========================================
print("\n--- Step 6: Deploying model to API ---")
model_src = 'zba_model_v2.pkl'
model_dst = 'api/zba_model.pkl'

if os.path.exists(model_src):
    shutil.copy2(model_src, model_dst)
    size_mb = os.path.getsize(model_dst) / (1024 * 1024)
    print(f"  ✅ Copied {model_src} → {model_dst} ({size_mb:.1f} MB)")
else:
    print(f"  ⚠️ {model_src} not found — check training output above")
sys.stdout.flush()

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 60)
print("  OVERNIGHT REBUILD COMPLETE")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

if os.path.exists('zba_cases_cleaned.csv'):
    import pandas as pd
    df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
    print(f"  Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Unique cases: {df['case_number'].nunique()}")
    if 'decision_clean' in df.columns:
        print(f"  With decisions: {df['decision_clean'].notna().sum()}")
        print(f"  Approval rate: {(df['decision_clean'] == 'APPROVED').mean():.1%}")

if os.path.exists(model_dst):
    print(f"  Model: {model_dst} ({os.path.getsize(model_dst)/1024/1024:.1f} MB)")
    print(f"\n  ✅ Ready for demo! Start with:")
    print(f"    cd api && uvicorn main:app --reload --port 8000")
else:
    print(f"\n  ⚠️ Model not deployed — check logs above")

print(f"\n  Full log saved to: overnight_log.txt")
sys.stdout.flush()
