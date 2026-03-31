#!/usr/bin/env python3
"""
enrich_from_agendas.py — Extract features from ZBA agenda raw_text and merge into zba_cases_cleaned.csv.

Features extracted:
  - agenda_articles: list of distinct article numbers cited (e.g., [50, 51, 65])
  - agenda_num_articles: count of distinct articles cited
  - agenda_num_sections: count of distinct article+section pairs cited
  - agenda_gcod: whether GCOD (Groundwater Conservation Overlay District) is mentioned
  - agenda_ipod: whether IPOD (Interim Planning Overlay District) is mentioned
  - agenda_demolition: whether demolition is mentioned
  - agenda_proposed_units: unit count extracted from agenda text
  - agenda_proposed_stories: story count extracted from agenda text
  - agenda_conditional_use: whether conditional use is mentioned
  - agenda_num_violations: count of violation keywords (insufficient, excessive, forbidden)
  - agenda_has_purpose: whether a Purpose section exists (richer project description)

Merge strategy:
  - New agenda-only columns are always added
  - Existing columns (proposed_units, proposed_stories, num_zoning_sections) are only
    overwritten if the agenda value is non-null AND the existing value is null/0/NaN
"""

import pandas as pd
import numpy as np
import re
from collections import Counter


def extract_articles(text):
    """Extract distinct article numbers from agenda text."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Match 'Article 50', 'Art. 50', 'Art 50', 'Articles 50'
    matches = re.findall(r'(?:Articles?|Art\.?)\s*(\d+)', text, re.IGNORECASE)
    return sorted(set(int(m) for m in matches if 1 <= int(m) <= 100))


def extract_sections(text):
    """Extract distinct article+section pairs."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Match 'Article 50, Section 29' or 'Art. 50 Sec. 29' etc.
    pairs = re.findall(
        r'(?:Articles?|Art\.?)\s*(\d+)[,\s]+(?:Sections?|Sec\.?)\s*(\d+)',
        text, re.IGNORECASE
    )
    return list(set((int(a), int(s)) for a, s in pairs if 1 <= int(a) <= 100))


def extract_proposed_stories(text):
    """Extract proposed story count from agenda text.

    Looks for patterns like:
      'Height Excessive (Stories) - Max. allowed: 3 Proposed: 5'
      'Proposed 3 stories'
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Pattern 1: "Stories ... Proposed: X"
    m = re.search(
        r'(?:Height\s+Excessive\s*\(Stories\)|stories).*?[Pp]roposed:?\s*(\d+)',
        text
    )
    if m:
        val = int(m.group(1))
        if 1 <= val <= 30:
            return val

    # Pattern 2: "X-story" or "X story" in Purpose section
    purpose_match = re.search(r'Purpose\s*:?\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if purpose_match:
        purpose = purpose_match.group(1)
        m = re.search(r'(\d+)\s*[\s-]stor(?:y|ie)', purpose, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 30:
                return val

    return None


def extract_proposed_units(text):
    """Extract proposed unit count from agenda text.

    Looks for patterns in Purpose section like:
      'erect ... 9 unit residential building'
      'change of occupancy ... to 5 units'
      'building with 36 dwelling units'
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Focus on Purpose section for unit counts (most reliable)
    purpose_match = re.search(r'Purpose\s*:?\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if purpose_match:
        purpose = purpose_match.group(1)
        # "X units" or "X dwelling units" or "X residential units"
        m = re.search(
            r'(\d+)\s*(?:dwelling\s+)?(?:residential\s+)?(?:unit|Unit)',
            purpose
        )
        if m:
            val = int(m.group(1))
            if 1 <= val <= 500:
                return val
        # "X family" dwelling
        m = re.search(r'(\d+)\s*(?:family|Family)', purpose)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                return val

    # Fallback: "Proposed X units" outside Purpose
    m = re.search(r'[Pp]roposed:?\s*(\d+)\s*(?:unit|dwelling)', text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 500:
            return val

    return None


def count_violations(text):
    """Count violation keywords: insufficient, excessive, forbidden."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r'[Ii]nsufficient|[Ee]xcessive|[Ff]orbidden', text))


def has_gcod(text):
    """Check for Groundwater Conservation Overlay District mention."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'GCOD|[Gg]roundwater\s+[Cc]onservation', text))


def has_ipod(text):
    """Check for Interim Planning Overlay District mention."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'IPOD|[Ii]nterim\s+[Pp]lanning\s+[Oo]verlay', text))


def has_demolition(text):
    """Check for demolition mention."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[Dd]emoli', text))


def has_conditional(text):
    """Check for conditional use mention."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[Cc]onditional', text))


def has_purpose(text):
    """Check if a Purpose section exists (richer project description)."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'Purpose\s*:', text, re.IGNORECASE))


def main():
    print("=" * 70)
    print("Enriching zba_cases_cleaned.csv from ZBA agenda data")
    print("=" * 70)

    # Load data
    agendas = pd.read_csv('zba_agendas.csv')
    cleaned = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)

    print(f"\nAgenda cases: {len(agendas)}")
    print(f"Cleaned cases: {len(cleaned)}")

    # Filter to agendas with raw_text
    agendas_with_text = agendas[agendas['raw_text'].notna()].copy()
    print(f"Agendas with raw_text: {len(agendas_with_text)}")

    # Find overlap
    overlap_cases = set(agendas_with_text['case_number']) & set(cleaned['case_number'])
    print(f"Overlapping cases (will be enriched): {len(overlap_cases)}")

    # ── Extract features from agenda text ──────────────────────────────
    print("\nExtracting features from agenda raw_text...")

    agenda_features = []
    for _, row in agendas_with_text.iterrows():
        case = row['case_number']
        text = row['raw_text']

        articles = extract_articles(text)
        sections = extract_sections(text)

        agenda_features.append({
            'case_number': case,
            'agenda_articles': str(articles) if articles else None,
            'agenda_num_articles': len(articles),
            'agenda_num_sections': len(sections),
            'agenda_gcod': int(has_gcod(text)),
            'agenda_ipod': int(has_ipod(text)),
            'agenda_demolition': int(has_demolition(text)),
            'agenda_proposed_units': extract_proposed_units(text),
            'agenda_proposed_stories': extract_proposed_stories(text),
            'agenda_conditional_use': int(has_conditional(text)),
            'agenda_num_violations': count_violations(text),
            'agenda_has_purpose': int(has_purpose(text)),
        })

    feat_df = pd.DataFrame(agenda_features)

    # ── Report extraction stats ────────────────────────────────────────
    print("\n── Extraction Summary ──")
    print(f"  Cases with articles extracted:    {(feat_df['agenda_num_articles'] > 0).sum()}")
    print(f"  Cases with sections extracted:    {(feat_df['agenda_num_sections'] > 0).sum()}")
    print(f"  Cases with GCOD:                  {feat_df['agenda_gcod'].sum()}")
    print(f"  Cases with IPOD:                  {feat_df['agenda_ipod'].sum()}")
    print(f"  Cases with demolition:            {feat_df['agenda_demolition'].sum()}")
    print(f"  Cases with conditional use:       {feat_df['agenda_conditional_use'].sum()}")
    print(f"  Cases with proposed units:        {feat_df['agenda_proposed_units'].notna().sum()}")
    print(f"  Cases with proposed stories:      {feat_df['agenda_proposed_stories'].notna().sum()}")
    print(f"  Cases with Purpose section:       {feat_df['agenda_has_purpose'].sum()}")
    print(f"  Avg violations per case:          {feat_df['agenda_num_violations'].mean():.1f}")
    print(f"  Max violations in a case:         {feat_df['agenda_num_violations'].max()}")

    # Top articles cited
    all_articles = []
    for arts in feat_df['agenda_articles'].dropna():
        all_articles.extend(eval(arts))
    art_counts = Counter(all_articles).most_common(10)
    print(f"\n  Top articles cited:")
    for art, cnt in art_counts:
        print(f"    Article {art}: {cnt} cases")

    # ── Merge into cleaned dataset ─────────────────────────────────────
    print("\n── Merging into zba_cases_cleaned.csv ──")

    # Keep only cases that exist in cleaned
    feat_df = feat_df[feat_df['case_number'].isin(overlap_cases)].copy()
    print(f"  Features to merge: {len(feat_df)} cases")

    # Handle duplicates in agenda data (keep first / most complete)
    feat_df = feat_df.sort_values('agenda_num_sections', ascending=False).drop_duplicates(
        subset='case_number', keep='first'
    )
    print(f"  After dedup: {len(feat_df)} cases")

    # ── Add new agenda-only columns ────────────────────────────────────
    new_cols = [
        'agenda_articles', 'agenda_num_articles', 'agenda_num_sections',
        'agenda_gcod', 'agenda_ipod', 'agenda_demolition',
        'agenda_conditional_use', 'agenda_num_violations', 'agenda_has_purpose'
    ]

    # Merge new columns
    merged = cleaned.merge(
        feat_df[['case_number'] + new_cols],
        on='case_number', how='left'
    )

    # Fill NaN for new columns with 0 (for cases not in agendas)
    for col in new_cols:
        if col == 'agenda_articles':
            continue  # Keep as NaN string — it's a list representation
        merged[col] = merged[col].fillna(0).astype(int)

    # ── Conditionally update existing columns ──────────────────────────
    # Only overwrite if existing value is null/0/NaN AND agenda has a value
    units_from_agenda = feat_df[['case_number', 'agenda_proposed_units']].dropna(subset=['agenda_proposed_units'])
    stories_from_agenda = feat_df[['case_number', 'agenda_proposed_stories']].dropna(subset=['agenda_proposed_stories'])

    # Track updates
    units_updated = 0
    stories_updated = 0
    sections_updated = 0

    # Create lookup dicts
    agenda_units = dict(zip(units_from_agenda['case_number'], units_from_agenda['agenda_proposed_units']))
    agenda_stories = dict(zip(stories_from_agenda['case_number'], stories_from_agenda['agenda_proposed_stories']))
    agenda_sections = dict(zip(feat_df['case_number'], feat_df['agenda_num_sections']))

    for idx, row in merged.iterrows():
        case = row['case_number']

        # Update proposed_units if currently missing
        if case in agenda_units:
            existing = row.get('proposed_units')
            if pd.isna(existing) or existing == 0:
                merged.at[idx, 'proposed_units'] = agenda_units[case]
                units_updated += 1

        # Update proposed_stories if currently missing
        if case in agenda_stories:
            existing = row.get('proposed_stories')
            if pd.isna(existing) or existing == 0:
                merged.at[idx, 'proposed_stories'] = agenda_stories[case]
                stories_updated += 1

        # Update num_zoning_sections if agenda has more
        if case in agenda_sections and agenda_sections[case] > 0:
            existing = row.get('num_zoning_sections', 0)
            if pd.isna(existing) or existing == 0:
                merged.at[idx, 'num_zoning_sections'] = agenda_sections[case]
                sections_updated += 1

    print(f"\n── Existing Column Updates (fill-only, no overwrites) ──")
    print(f"  proposed_units filled:       {units_updated} cases")
    print(f"  proposed_stories filled:     {stories_updated} cases")
    print(f"  num_zoning_sections filled:  {sections_updated} cases")

    # ── Save ───────────────────────────────────────────────────────────
    merged.to_csv('zba_cases_cleaned.csv', index=False)
    print(f"\n── Saved ──")
    print(f"  Output: zba_cases_cleaned.csv")
    print(f"  Total cases: {len(merged)}")
    print(f"  Total columns: {len(merged.columns)} (was {len(cleaned.columns)})")
    new_col_names = set(merged.columns) - set(cleaned.columns)
    if new_col_names:
        print(f"  New columns added: {sorted(new_col_names)}")

    # ── Verification ───────────────────────────────────────────────────
    print(f"\n── Verification ──")
    enriched = merged[merged['case_number'].isin(overlap_cases)]
    print(f"  Enriched cases: {len(enriched)}")
    for col in new_cols:
        if col == 'agenda_articles':
            nonzero = enriched[col].notna().sum()
        else:
            nonzero = (enriched[col] > 0).sum()
        print(f"    {col}: {nonzero} non-zero")

    # Units/stories coverage
    has_units = merged['proposed_units'].notna() & (merged['proposed_units'] > 0)
    has_stories = merged['proposed_stories'].notna() & (merged['proposed_stories'] > 0)
    print(f"\n  Overall proposed_units coverage:   {has_units.sum()} / {len(merged)} ({100*has_units.mean():.1f}%)")
    print(f"  Overall proposed_stories coverage: {has_stories.sum()} / {len(merged)} ({100*has_stories.mean():.1f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
