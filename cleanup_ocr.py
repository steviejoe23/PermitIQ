"""
PermitIQ OCR Artifact Cleanup
Post-processes zba_cases_cleaned.csv to fix common OCR errors, recover missing
decisions from raw_text, normalize addresses, and deduplicate cases.

Run: python3 cleanup_ocr.py
"""

import pandas as pd
import re
import sys


def clean_address(addr):
    """Fix common OCR artifacts in addresses."""
    if not addr or not isinstance(addr, str):
        return addr

    # Remove control characters
    addr = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', addr)

    # Remove newlines
    addr = addr.replace('\n', ' ').replace('\r', ' ')

    # Fix common OCR number substitutions
    addr = re.sub(r'\bO(\d)', r'0\1', addr)
    addr = re.sub(r'(\d)O\b', r'\g<1>0', addr)
    addr = re.sub(r'\bl\b(?=\d)', '1', addr)

    # Remove stray OCR fragments
    addr = re.sub(r'\s[|/\\]\s', ' ', addr)

    # Normalize whitespace
    addr = re.sub(r'\s+', ' ', addr).strip()

    # Remove trailing garbage
    addr = re.sub(r'[^\w\s]+$', '', addr).strip()

    # If address is too long, try to extract just the street address
    if len(addr) > 80:
        # Pattern: number + street name + street type
        m = re.match(
            r'(\d+[\-\w]*\s+[\w\s\.\-]+?'
            r'(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|'
            r'Way|Lane|Ln|Court|Ct|Place|Pl|Terrace|Ter|Circle|Cir|'
            r'Square|Sq|Parkway|Pkwy|Highway|Hwy|Broadway))'
            r'(?:\s|,|$)',
            addr, re.IGNORECASE
        )
        if m:
            addr = m.group(1).strip()
        else:
            # Fallback: take first 80 chars up to a comma or sentence boundary
            short = addr[:80]
            for delim in [',', '.', ' in the ', ' from ']:
                idx = short.find(delim)
                if idx > 10:
                    addr = short[:idx].strip()
                    break

    # Normalize common abbreviations
    addr = re.sub(r'\bStreet\b', 'St', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bAvenue\b', 'Ave', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bRoad\b', 'Rd', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bDrive\b', 'Dr', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bBoulevard\b', 'Blvd', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bLane\b', 'Ln', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bCourt\b', 'Ct', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bPlace\b', 'Pl', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bTerrace\b', 'Ter', addr, flags=re.IGNORECASE)

    return addr


def clean_case_number(case_num):
    """Normalize BOA case numbers."""
    if not case_num or not isinstance(case_num, str):
        return case_num

    case_num = case_num.strip()
    # Normalize BOA prefix variants: BOA-, BOA , B0A, etc.
    case_num = re.sub(r'^B[O0]A[\s\-]*', 'BOA-', case_num)

    # Ensure consistent format: BOA-1234567
    m = re.match(r'^BOA[\-]?(\d+)', case_num)
    if m:
        return f"BOA-{m.group(1)}"

    return case_num


def recover_decision_from_text(row):
    """Try to extract decision from raw_text for cases with missing decision_clean."""
    if pd.notna(row.get('decision_clean')):
        return row['decision_clean']

    text = str(row.get('raw_text', '')).upper()
    if not text:
        return None

    # Look for explicit board decision language
    decision_patterns = [
        # "voted to approve" / "voted to deny"
        (r'VOTED\s+TO\s+(?:GRANT|APPROVE|ALLOW|SUSTAIN)', 'APPROVED'),
        (r'VOTED\s+TO\s+(?:DENY|REFUSE|REJECT)', 'DENIED'),
        # "appeal is granted" / "appeal is denied"
        (r'APPEAL\s+(?:IS\s+)?(?:HEREBY\s+)?(?:GRANTED|SUSTAINED|APPROVED)', 'APPROVED'),
        (r'APPEAL\s+(?:IS\s+)?(?:HEREBY\s+)?(?:DENIED|DISMISSED|REJECTED)', 'DENIED'),
        # "the board voted X-Y to approve/deny"
        (r'BOARD\s+VOTED\s+\d+[\-–]\d+\s+TO\s+(?:GRANT|APPROVE)', 'APPROVED'),
        (r'BOARD\s+VOTED\s+\d+[\-–]\d+\s+TO\s+(?:DENY|REFUSE)', 'DENIED'),
        # Generic at end of document
        (r'(?:DECISION|RULING|ORDER)\s*:\s*(?:GRANTED|APPROVED)', 'APPROVED'),
        (r'(?:DECISION|RULING|ORDER)\s*:\s*(?:DENIED|REJECTED)', 'DENIED'),
    ]

    for pattern, decision in decision_patterns:
        if re.search(pattern, text):
            return decision

    return None


def main():
    input_file = 'zba_cases_cleaned.csv'
    output_file = 'zba_cases_cleaned.csv'

    print("=" * 60)
    print("  PermitIQ OCR Artifact Cleanup (Enhanced)")
    print("=" * 60)

    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    original_len = len(df)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    stats = {}

    # 1. Recover missing decisions from raw_text
    if 'decision_clean' in df.columns and 'raw_text' in df.columns:
        missing_before = df['decision_clean'].isna().sum()
        df['decision_clean'] = df.apply(recover_decision_from_text, axis=1)
        missing_after = df['decision_clean'].isna().sum()
        recovered = missing_before - missing_after
        stats['decisions_recovered'] = recovered
        print(f"\n  Decisions recovered from raw text: {recovered}")
        print(f"    Missing before: {missing_before} → Missing after: {missing_after}")

    # 2. Clean addresses
    if 'address_clean' in df.columns:
        before = df['address_clean'].copy()
        df['address_clean'] = df['address_clean'].apply(clean_address)
        changed = (before.fillna('') != df['address_clean'].fillna('')).sum()
        stats['addresses_cleaned'] = changed
        print(f"  Addresses cleaned: {changed}")

        # Nullify garbage addresses
        garbage_mask = df['address_clean'].notna() & (
            (df['address_clean'].str.len() < 3) |
            (~df['address_clean'].str.contains(r'\d', regex=True, na=False))
        )
        garbage_count = garbage_mask.sum()
        if garbage_count > 0:
            df.loc[garbage_mask, 'address_clean'] = pd.NA
            stats['garbage_addresses'] = garbage_count
            print(f"  Garbage addresses nullified: {garbage_count}")

    # 3. Clean case numbers
    if 'case_number' in df.columns:
        before = df['case_number'].copy()
        df['case_number'] = df['case_number'].apply(clean_case_number)
        changed = (before.fillna('') != df['case_number'].fillna('')).sum()
        stats['case_numbers_cleaned'] = changed
        print(f"  Case numbers cleaned: {changed}")

    # 4. Decision normalization
    if 'decision_clean' in df.columns:
        before = df['decision_clean'].copy()
        df['decision_clean'] = df['decision_clean'].apply(
            lambda d: 'APPROVED' if pd.notna(d) and 'APPROV' in str(d).upper()
            else ('DENIED' if pd.notna(d) and 'DENI' in str(d).upper()
            else d)
        )
        changed = (before.fillna('') != df['decision_clean'].fillna('')).sum()
        stats['decisions_normalized'] = changed
        print(f"  Decisions normalized: {changed}")

    # 5. Deduplicate by case number
    if 'case_number' in df.columns:
        before_len = len(df)
        df = df.drop_duplicates(subset='case_number', keep='first')
        dupes = before_len - len(df)
        stats['duplicates_removed'] = dupes
        print(f"  Duplicates removed: {dupes}")

    # 6. Fix has_attorney extraction (broader pattern)
    if 'raw_text' in df.columns:
        rt = df['raw_text'].fillna('').str.lower()
        old_attorney = df['has_attorney'].sum() if 'has_attorney' in df.columns else 0
        df['has_attorney'] = rt.str.contains(
            r'attorney|counsel|esq\.?|law\s*office|represented\s*by|on\s*behalf\s*of.*(?:llc|inc|esq)',
            regex=True
        ).astype(int)
        new_attorney = df['has_attorney'].sum()
        stats['attorney_fixed'] = int(new_attorney - old_attorney)
        print(f"  has_attorney: {old_attorney} → {new_attorney} (+{new_attorney - old_attorney})")

    # 7. Clean raw_text whitespace
    if 'raw_text' in df.columns:
        df['raw_text'] = df['raw_text'].fillna('').str.replace(r'\s{3,}', '  ', regex=True)
        if 'text_length' in df.columns:
            df['text_length'] = df['raw_text'].str.len()

    # Summary
    total_changes = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"  CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"  Original rows: {original_len}")
    print(f"  Final rows: {len(df)} ({original_len - len(df)} removed)")
    print(f"  Total modifications: {total_changes}")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {name}: {count}")

    # Decision stats after cleanup
    if 'decision_clean' in df.columns:
        print(f"\n  Decision distribution (after cleanup):")
        dec = df['decision_clean'].value_counts()
        for d, count in dec.items():
            print(f"    {d}: {count} ({count/len(df):.1%})")
        missing = df['decision_clean'].isna().sum()
        print(f"    Missing: {missing} ({missing/len(df):.1%})")

    if total_changes > 0:
        print(f"\n  Saving to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"  Done.")
    else:
        print(f"\n  No changes needed.")


if __name__ == '__main__':
    main()
