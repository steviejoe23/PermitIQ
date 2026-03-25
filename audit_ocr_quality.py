"""
PermitIQ OCR Quality Audit
Samples cases from zba_cases_cleaned.csv and checks for common OCR quality issues.
Produces a quality report without needing to compare against original PDFs.

Run: python3 audit_ocr_quality.py
"""

import pandas as pd
import re
import sys


def audit():
    print("=" * 60)
    print("  PermitIQ OCR Quality Audit")
    print("=" * 60)

    df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
    print(f"\nTotal rows: {len(df)}")

    issues = {}

    # 1. Address quality
    print("\n--- ADDRESS QUALITY ---")
    if 'address_clean' in df.columns:
        has_addr = df['address_clean'].notna().sum()
        no_addr = df['address_clean'].isna().sum()
        print(f"  Has address: {has_addr} ({has_addr/len(df):.0%})")
        print(f"  Missing address: {no_addr} ({no_addr/len(df):.0%})")

        # Check for garbage addresses
        addrs = df['address_clean'].dropna()
        no_digit = addrs[~addrs.str.contains(r'\d', regex=True)]
        too_short = addrs[addrs.str.len() < 5]
        too_long = addrs[addrs.str.len() > 80]
        has_newline = addrs[addrs.str.contains(r'\n', regex=True)]

        print(f"  No digit in address: {len(no_digit)} ({len(no_digit)/len(addrs):.1%})")
        print(f"  Too short (<5 chars): {len(too_short)} ({len(too_short)/len(addrs):.1%})")
        print(f"  Too long (>80 chars): {len(too_long)} ({len(too_long)/len(addrs):.1%})")
        print(f"  Contains newlines: {len(has_newline)} ({len(has_newline)/len(addrs):.1%})")

        if len(too_long) > 0:
            print(f"\n  Sample long addresses:")
            for a in too_long.head(5).values:
                print(f"    [{len(a)} chars] {a[:80]}...")

        issues['address_missing'] = no_addr
        issues['address_garbage'] = len(no_digit) + len(too_short) + len(too_long)

    # 2. Case number quality
    print("\n--- CASE NUMBER QUALITY ---")
    if 'case_number' in df.columns:
        cases = df['case_number'].dropna()
        valid_boa = cases.str.match(r'^BOA[\-]?\d{5,}$')
        print(f"  Valid BOA format: {valid_boa.sum()} ({valid_boa.mean():.0%})")
        print(f"  Non-standard: {(~valid_boa).sum()}")

        bad_cases = cases[~valid_boa].head(10)
        if len(bad_cases) > 0:
            print(f"  Sample non-standard case numbers:")
            for c in bad_cases.values:
                print(f"    {c}")

        # Duplicates
        dupes = cases.duplicated().sum()
        print(f"  Duplicate case numbers: {dupes}")
        issues['case_dupes'] = dupes

    # 3. Decision quality
    print("\n--- DECISION QUALITY ---")
    if 'decision_clean' in df.columns:
        dec = df['decision_clean'].value_counts()
        print(f"  Decision distribution:")
        for d, count in dec.items():
            print(f"    {d}: {count} ({count/len(df):.1%})")

        missing_dec = df['decision_clean'].isna().sum()
        print(f"  Missing decision: {missing_dec} ({missing_dec/len(df):.1%})")
        issues['decision_missing'] = missing_dec

    # 4. Raw text quality
    print("\n--- RAW TEXT QUALITY ---")
    if 'raw_text' in df.columns:
        texts = df['raw_text'].dropna()
        print(f"  Has raw_text: {len(texts)} ({len(texts)/len(df):.0%})")

        avg_len = texts.str.len().mean()
        median_len = texts.str.len().median()
        print(f"  Average text length: {avg_len:,.0f} chars")
        print(f"  Median text length: {median_len:,.0f} chars")

        # Check for OCR garbage indicators
        high_special = texts[texts.str.count(r'[^a-zA-Z0-9\s\.\,\;\:\-\(\)]') / texts.str.len() > 0.1]
        print(f"  High special char ratio (>10%): {len(high_special)} ({len(high_special)/len(texts):.1%})")

        very_short = texts[texts.str.len() < 100]
        print(f"  Very short text (<100 chars): {len(very_short)} ({len(very_short)/len(texts):.1%})")

        issues['text_garbage'] = len(high_special)
        issues['text_too_short'] = len(very_short)

    # 5. Feature extraction quality
    print("\n--- FEATURE EXTRACTION QUALITY ---")
    key_features = ['has_attorney', 'num_variances', 'proposed_units', 'proposed_stories', 'ward']
    for feat in key_features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            non_zero = (df[feat].fillna(0) != 0).sum()
            print(f"  {feat}: {non_null} non-null ({non_null/len(df):.0%}), {non_zero} non-zero ({non_zero/len(df):.0%})")

    # 6. Filing date quality
    print("\n--- DATE QUALITY ---")
    if 'filing_date' in df.columns:
        has_date = df['filing_date'].notna().sum()
        parseable = pd.to_datetime(df['filing_date'], errors='coerce').notna().sum()
        print(f"  Has filing_date: {has_date} ({has_date/len(df):.0%})")
        print(f"  Parseable dates: {parseable} ({parseable/len(df):.0%})")

    # Summary
    print("\n" + "=" * 60)
    print("  QUALITY SUMMARY")
    print("=" * 60)

    total_issues = sum(issues.values())
    print(f"\n  Total issues found: {total_issues}")
    for name, count in sorted(issues.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {name}: {count}")

    quality_score = max(0, 100 - (total_issues / len(df) * 100))
    print(f"\n  Overall quality score: {quality_score:.0f}/100")

    if quality_score >= 80:
        print("  Status: GOOD — ready for training")
    elif quality_score >= 60:
        print("  Status: FAIR — run cleanup_ocr.py before training")
    else:
        print("  Status: POOR — significant OCR issues, manual review recommended")


if __name__ == '__main__':
    audit()
