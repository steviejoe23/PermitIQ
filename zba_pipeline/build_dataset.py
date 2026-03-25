import glob
import json
import os
import sys
import time
import pandas as pd

from zba_pipeline.extract_text import extract_pdf_text
from zba_pipeline.parse_cases import parse_cases

CHECKPOINT_FILE = "ocr_checkpoint.json"


def recover_cases(text):
    """
    Fallback extraction if primary parsing misses cases
    """
    import re

    pattern = r"(BOA[\s\-]?\d{6,7}.*?)((?=BOA[\s\-]?\d{6,7})|$)"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    return [m[0].strip() for m in matches if len(m[0]) > 150]


def load_checkpoint():
    """Load checkpoint of already-processed PDFs"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "cases": []}


def save_checkpoint(checkpoint):
    """Save checkpoint after each PDF"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def main():
    # Allow --fresh flag to ignore checkpoint
    fresh = "--fresh" in sys.argv

    print("Working directory:", os.getcwd())
    sys.stdout.flush()

    pdf_files = sorted(glob.glob("pdfs/*.pdf"))
    print(f"PDF files found: {len(pdf_files)}")
    sys.stdout.flush()

    # Load checkpoint (unless --fresh)
    if fresh:
        print("🔄 Fresh run — ignoring checkpoint")
        checkpoint = {"completed": [], "cases": []}
    else:
        checkpoint = load_checkpoint()
        if checkpoint["completed"]:
            print(f"📋 Resuming from checkpoint: {len(checkpoint['completed'])}/{len(pdf_files)} already done")
        sys.stdout.flush()

    all_cases = checkpoint["cases"]
    completed_set = set(checkpoint["completed"])
    remaining = [p for p in pdf_files if p not in completed_set]

    print(f"📄 PDFs to process: {len(remaining)}")
    sys.stdout.flush()

    start_time = time.time()

    for i, pdf in enumerate(remaining):
        pdf_start = time.time()
        print(f"\n[{len(completed_set) + i + 1}/{len(pdf_files)}] Processing: {pdf}")
        sys.stdout.flush()

        try:
            text = extract_pdf_text(pdf)
        except Exception as e:
            print(f"  ❌ Error reading PDF: {e}")
            sys.stdout.flush()
            # Still mark as completed so we don't retry a broken PDF forever
            checkpoint["completed"].append(pdf)
            save_checkpoint(checkpoint)
            continue

        cases = parse_cases(text, pdf)

        # Fallback if parser fails
        if len(cases) == 0:
            print(f"  ⚠️ Using fallback extraction")
            fallback_cases = recover_cases(text)

            for fc in fallback_cases:
                cases.append({
                    "case_number": None,
                    "address": None,
                    "zoning": None,
                    "decision": None,
                    "raw_text": fc,
                    "source_pdf": pdf
                })

        all_cases.extend(cases)
        elapsed = time.time() - pdf_start
        print(f"  ✅ {len(cases)} cases extracted ({elapsed:.0f}s)")
        sys.stdout.flush()

        # Save checkpoint after each PDF
        checkpoint["completed"].append(pdf)
        checkpoint["cases"] = all_cases
        save_checkpoint(checkpoint)

    total_elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"OCR complete: {len(pdf_files)} PDFs in {total_elapsed/60:.1f} minutes")
    print(f"Total raw cases: {len(all_cases)}")
    sys.stdout.flush()

    df = pd.DataFrame(all_cases)

    print(f"Initial rows: {len(df)}")

    # Drop empty rows
    df = df[df["raw_text"].notnull()]

    # Deduplicate
    df = df.drop_duplicates(subset=["case_number", "source_pdf"])

    print(f"After dedup: {len(df)}")

    # Normalize decisions
    df["decision"] = df["decision"].replace({
        "APPROVED": "GRANTED"
    })

    # Save
    df.to_csv("zba_cases_dataset.csv", index=False)

    print(f"✅ Dataset saved: zba_cases_dataset.csv ({len(df)} rows)")
    sys.stdout.flush()

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🧹 Checkpoint cleared (full run complete)")


if __name__ == "__main__":
    main()
