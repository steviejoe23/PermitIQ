"""
PermitIQ — Deep Re-extraction from OCR Text
Extracts maximum structured data from the raw_text column of ZBA cases.
Much more aggressive regex patterns than the original extraction.
"""

import pandas as pd
import numpy as np
import re
import os

print("Loading dataset...")
df = pd.read_csv('zba_cases_cleaned.csv', low_memory=False)
print(f"Loaded {len(df)} cases")

# ========================================
# 1. BETTER ADDRESS EXTRACTION
# ========================================
print("\n--- Extracting addresses ---")

def extract_address_v3(text):
    """Much more aggressive address extraction."""
    if pd.isna(text):
        return None, None

    text = str(text)

    # Pattern 1: "premises located at ADDRESS, Ward XX"
    m = re.search(r'premises\s+located\s+at\s*\n?\s*(.+?),?\s*Ward\s+(\d+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2)

    # Pattern 2: "premises located at ADDRESS" (no ward)
    m = re.search(r'premises\s+located\s+at\s*\n?\s*(.+?)(?:\n|For relief)', text, re.IGNORECASE)
    if m:
        addr = m.group(1).strip().rstrip(',').strip()
        # Try to extract ward from elsewhere
        w = re.search(r'Ward\s+(\d+)', text)
        ward = w.group(1) if w else None
        if len(addr) > 5 and re.match(r'\d', addr):
            return addr, ward

    # Pattern 3: "appeal of NAME Concerning ... ADDRESS"
    m = re.search(r'appeal\s+of\s+.+?\n\s*(?:Concerning\s+)?(?:the\s+)?(?:premises\s+)?(?:located\s+)?(?:at\s+)?\n?\s*(\d+[^,\n]+?)(?:,\s*Ward\s+(\d+))?(?:\n|$)', text, re.IGNORECASE)
    if m:
        addr = m.group(1).strip()
        ward = m.group(2) if m.group(2) else None
        if not ward:
            w = re.search(r'Ward\s+(\d+)', text)
            ward = w.group(1) if w else None
        if len(addr) > 5:
            return addr, ward

    # Pattern 4: Any "Ward XX" with a street address nearby
    m = re.search(r'(\d+[A-Za-z\s\-\.]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Place|Pl|Terrace|Ter|Court|Ct|Lane|Ln|Boulevard|Blvd|Parkway|Circle|Square|Highway|Hwy)[^,\n]*),?\s*(?:Ward\s+(\d+))?', text, re.IGNORECASE)
    if m:
        addr = m.group(1).strip()
        ward = m.group(2) if m.group(2) else None
        if not ward:
            w = re.search(r'Ward\s+(\d+)', text)
            ward = w.group(1) if w else None
        return addr, ward

    # Pattern 5: Just find Ward
    w = re.search(r'Ward\s+(\d+)', text)
    return None, w.group(1) if w else None


new_addresses = 0
new_wards = 0

for idx, row in df.iterrows():
    addr, ward = extract_address_v3(row['raw_text'])

    if addr and pd.isna(row.get('address_clean')):
        df.at[idx, 'address_clean'] = addr
        new_addresses += 1

    if ward and pd.isna(row.get('ward')):
        df.at[idx, 'ward'] = float(ward)
        new_wards += 1

print(f"  New addresses extracted: {new_addresses}")
print(f"  New wards extracted: {new_wards}")
print(f"  Total addresses now: {df['address_clean'].notna().sum()}/{len(df)}")
print(f"  Total wards now: {df['ward'].notna().sum()}/{len(df)}")


# ========================================
# 2. APPLICANT NAME EXTRACTION
# ========================================
print("\n--- Extracting applicant names ---")

def extract_applicant(text):
    if pd.isna(text):
        return None
    text = str(text)

    # "appeal of NAME"
    m = re.search(r'appeal\s+of\s*\n?\s*([A-Z][A-Za-z\s\.\-,&\']+?)(?:\n|Concerning|concerning)', text)
    if m:
        name = m.group(1).strip().rstrip(',').strip()
        # Filter out garbage
        if len(name) > 2 and len(name) < 100 and not re.search(r'(?:premises|located|ward|relief|zoning)', name, re.I):
            return name

    return None

df['applicant_name'] = df['raw_text'].apply(extract_applicant)
print(f"  Applicant names extracted: {df['applicant_name'].notna().sum()}/{len(df)}")


# ========================================
# 3. PERMIT NUMBER EXTRACTION
# ========================================
print("\n--- Extracting permit numbers ---")

def extract_permit(text):
    if pd.isna(text):
        return None
    m = re.search(r'PERMIT\s*#?\s*([A-Z]{2,4}\d{5,10})', str(text), re.IGNORECASE)
    return m.group(1) if m else None

df['permit_number'] = df['raw_text'].apply(extract_permit)
print(f"  Permit numbers extracted: {df['permit_number'].notna().sum()}/{len(df)}")


# ========================================
# 4. BETTER VARIANCE EXTRACTION
# ========================================
print("\n--- Re-extracting variance types ---")

VARIANCE_PATTERNS = {
    'height': r'(?:height|story|stories|floor.*height|building.*height|exceed.*height|taller)',
    'far': r'(?:floor\s*area\s*ratio|FAR|floor.*area.*exceed|excessive.*floor)',
    'lot_area': r'(?:lot\s*area|insufficient\s*lot|lot.*size|lot.*sq|minimum.*lot)',
    'lot_frontage': r'(?:lot\s*frontage|frontage|insufficient.*frontage|front.*width)',
    'front_setback': r'(?:front\s*(?:yard|setback)|front.*set\s*back|insufficient.*front)',
    'rear_setback': r'(?:rear\s*(?:yard|setback)|rear.*set\s*back|insufficient.*rear)',
    'side_setback': r'(?:side\s*(?:yard|setback)|side.*set\s*back|insufficient.*side)',
    'parking': r'(?:parking|off-street.*parking|insufficient.*parking|parking.*space|garage)',
    'conditional_use': r'(?:conditional\s*use|conditional.*permit|use\s*(?:permit|variance))',
    'open_space': r'(?:open\s*space|usable.*open|landscap)',
    'density': r'(?:density|dwelling.*unit.*per|units.*per.*acre|too.*many.*units)',
    'nonconforming': r'(?:nonconform|non-conform|pre-existing|existing.*nonconform|legal.*nonconform)',
}

new_variance_count = 0
for idx, row in df.iterrows():
    text = str(row.get('raw_text', ''))
    if not text or text == 'nan':
        continue

    text_lower = text.lower()
    found = []

    for var_type, pattern in VARIANCE_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(var_type)

    if found:
        existing = str(row.get('variance_types', ''))
        if existing == 'nan' or not existing:
            df.at[idx, 'variance_types'] = ','.join(found)
            df.at[idx, 'num_variances'] = len(found)
            new_variance_count += 1
        else:
            # Merge new findings with existing
            existing_set = set(existing.split(','))
            new_set = set(found)
            merged = existing_set | new_set
            if len(merged) > len(existing_set):
                df.at[idx, 'variance_types'] = ','.join(sorted(merged))
                df.at[idx, 'num_variances'] = len(merged)
                new_variance_count += 1

print(f"  Variance records updated: {new_variance_count}")
print(f"  Cases with variances now: {(df['num_variances'] > 0).sum()}/{len(df)}")


# ========================================
# 5. SPECIFIC ZONING ARTICLE EXTRACTION
# ========================================
print("\n--- Extracting specific zoning articles ---")

def extract_zoning_articles(text):
    if pd.isna(text):
        return {}
    text = str(text).lower()

    articles = {}
    # Article numbers
    for art_num in [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 25, 29, 32, 33, 39, 41, 43,
                    44, 50, 51, 53, 55, 56, 60, 62, 63, 64, 65, 66, 67, 68, 69, 80]:
        if re.search(rf'article\s*{art_num}\b', text):
            articles[f'zoning_article_{art_num}'] = 1

    # Section numbers
    sections = re.findall(r'section\s*(\d+)', text)
    articles['num_zoning_sections'] = len(set(sections))

    return articles

# Apply to all rows
article_data = df['raw_text'].apply(extract_zoning_articles)
article_df = pd.DataFrame(article_data.tolist()).fillna(0).astype(int)

# Add key articles to main df
for col in ['zoning_article_7', 'zoning_article_8', 'zoning_article_51',
            'zoning_article_65', 'zoning_article_80', 'num_zoning_sections']:
    if col in article_df.columns:
        df[col] = article_df[col]

# Update existing article columns
df['article_7'] = article_df.get('zoning_article_7', 0)
df['article_8'] = article_df.get('zoning_article_8', 0)
df['article_51'] = article_df.get('zoning_article_51', 0)
df['article_65'] = article_df.get('zoning_article_65', 0)
df['article_80'] = article_df.get('zoning_article_80', 0)

print(f"  Article 7 (dimensional): {(df['article_7']==1).sum()}")
print(f"  Article 8 (use): {(df['article_8']==1).sum()}")
print(f"  Article 51 (Roxbury): {(df['article_51']==1).sum()}")
print(f"  Article 65 (Mission Hill): {(df['article_65']==1).sum()}")
print(f"  Article 80 (large project): {(df['article_80']==1).sum()}")


# ========================================
# 6. BOARD MEMBER VOTE EXTRACTION
# ========================================
print("\n--- Extracting board member votes ---")

def extract_votes(text):
    if pd.isna(text):
        return 0, 0, []
    text = str(text)

    # Common patterns for votes
    in_favor = len(re.findall(r'In\s+Favor', text))
    opposed = len(re.findall(r'(?:Opposed|Against|Dissenting|Not\s+in\s+Favor)', text, re.IGNORECASE))

    # Try to find specific vote counts
    m = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:vote|decision)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), []

    # Extract board member names
    members = re.findall(r'(?:Board\s+Member|Member)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)

    return in_favor, opposed, members

vote_data = df['raw_text'].apply(lambda x: extract_votes(x))
df['votes_in_favor'] = vote_data.apply(lambda x: x[0])
df['votes_opposed'] = vote_data.apply(lambda x: x[1])
df['total_votes'] = df['votes_in_favor'] + df['votes_opposed']
df['unanimous'] = ((df['total_votes'] > 0) & (df['votes_opposed'] == 0)).astype(int)

print(f"  Cases with vote data: {(df['total_votes'] > 0).sum()}")
print(f"  Unanimous decisions: {df['unanimous'].sum()}")


# ========================================
# 7. PROVISOS EXTRACTION (what conditions?)
# ========================================
print("\n--- Extracting proviso details ---")

PROVISO_PATTERNS = {
    'proviso_bpda_review': r'(?:bpda|boston\s+planning|planning.*development.*agency).*(?:review|approval)',
    'proviso_design_review': r'(?:design\s*review|architectural\s*review)',
    'proviso_community': r'(?:community\s*(?:meeting|process|engagement|review))',
    'proviso_abutter_notice': r'(?:abutt|neighbor.*noti|adjacent.*property)',
    'proviso_time_limit': r'(?:within\s+\d+\s+(?:days|months|years)|expire|time\s*limit)',
    'proviso_plans_compliance': r'(?:in\s+accordance\s+with.*plans|substantial\s+compliance|as\s+(?:shown|submitted))',
    'proviso_traffic': r'(?:traffic|transportation|parking\s+management)',
    'proviso_environmental': r'(?:environmental|stormwater|drainage|erosion)',
}

for prov_name, pattern in PROVISO_PATTERNS.items():
    df[prov_name] = df['raw_text'].apply(
        lambda x: int(bool(re.search(pattern, str(x).lower()))) if pd.notna(x) else 0
    )

proviso_cols = [c for c in df.columns if c.startswith('proviso_')]
df['num_provisos'] = df[proviso_cols].sum(axis=1)

print(f"  Cases with any proviso detail: {(df['num_provisos'] > 0).sum()}")
for col in proviso_cols:
    print(f"    {col}: {(df[col]==1).sum()}")


# ========================================
# 8. RELIEF TYPE EXTRACTION
# ========================================
print("\n--- Extracting relief types ---")

df['appeal_sustained'] = df['raw_text'].apply(
    lambda x: int(bool(re.search(r'appeal\s+sustained', str(x), re.I))) if pd.notna(x) else 0
)
df['appeal_denied'] = df['raw_text'].apply(
    lambda x: int(bool(re.search(r'appeal\s+denied|petition.*denied', str(x), re.I))) if pd.notna(x) else 0
)
df['with_provisos'] = df['raw_text'].apply(
    lambda x: int(bool(re.search(r'with\s+provisos', str(x), re.I))) if pd.notna(x) else 0
)
df['without_prejudice'] = df['raw_text'].apply(
    lambda x: int(bool(re.search(r'without\s+prejudice', str(x), re.I))) if pd.notna(x) else 0
)

print(f"  Appeal sustained: {df['appeal_sustained'].sum()}")
print(f"  Appeal denied: {df['appeal_denied'].sum()}")
print(f"  With provisos: {df['with_provisos'].sum()}")
print(f"  Without prejudice: {df['without_prejudice'].sum()}")


# ========================================
# 9. PROPOSED UNITS & STORIES (better extraction)
# ========================================
print("\n--- Re-extracting units and stories ---")

def extract_units_stories(text):
    if pd.isna(text):
        return 0, 0
    text = str(text).lower()

    units = 0
    stories = 0

    # Units patterns
    m = re.search(r'(\d+)\s*(?:unit|dwelling|apartment|condo)', text)
    if m:
        units = int(m.group(1))

    # Multi-family specific
    m = re.search(r'(\d+)\s*(?:-\s*)?family', text)
    if m:
        val = int(m.group(1))
        if val <= 20:  # reasonable
            units = max(units, val)

    # Stories
    m = re.search(r'(\d+)\s*(?:-\s*)?stor(?:y|ies)', text)
    if m:
        stories = int(m.group(1))

    return units, stories

units_stories = df['raw_text'].apply(extract_units_stories)
df['proposed_units'] = units_stories.apply(lambda x: x[0])
df['proposed_stories'] = units_stories.apply(lambda x: x[1])

print(f"  Cases with unit count: {(df['proposed_units'] > 0).sum()}")
print(f"  Cases with story count: {(df['proposed_stories'] > 0).sum()}")


# ========================================
# 10. FILING & EXPIRY DATE EXTRACTION
# ========================================
print("\n--- Extracting filing dates ---")

def extract_filing_date(text):
    if pd.isna(text):
        return None
    text = str(text)

    m = re.search(r'(?:filing|filed).*?(?:was|:)\s*(\d{1,2}/\d{1,2}/\d{4})', text, re.I)
    if m:
        return m.group(1)

    m = re.search(r'(?:filing|filed).*?(?:was|:)\s*(\w+\s+\d{1,2},?\s+\d{4})', text, re.I)
    if m:
        return m.group(1)

    return None

df['filing_date'] = df['raw_text'].apply(extract_filing_date)
print(f"  Filing dates extracted: {df['filing_date'].notna().sum()}/{len(df)}")


# ========================================
# 8. ATTORNEY EXTRACTION (broader pattern)
# ========================================
print("\n--- Extracting attorney representation ---")
rt = df['raw_text'].fillna('').str.lower()
df['has_attorney'] = rt.str.contains(
    r'attorney|counsel|esq\.?|law\s*office|represented\s*by|on\s*behalf\s*of.*(?:llc|inc|esq)',
    regex=True
).astype(int)
print(f"  Has attorney: {df['has_attorney'].sum()}/{len(df)} ({df['has_attorney'].mean():.1%})")


# ========================================
# SAVE
# ========================================
print("\n--- Saving enhanced dataset ---")

# Summary stats
print(f"\nFINAL DATASET SUMMARY:")
print(f"  Total cases: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Addresses: {df['address_clean'].notna().sum()} ({df['address_clean'].notna().mean():.1%})")
print(f"  Wards: {df['ward'].notna().sum()} ({df['ward'].notna().mean():.1%})")
print(f"  Applicant names: {df['applicant_name'].notna().sum()} ({df['applicant_name'].notna().mean():.1%})")
print(f"  Permit numbers: {df['permit_number'].notna().sum()} ({df['permit_number'].notna().mean():.1%})")
print(f"  With variances: {(df['num_variances'] > 0).sum()} ({(df['num_variances'] > 0).mean():.1%})")
print(f"  With vote data: {(df['total_votes'] > 0).sum()} ({(df['total_votes'] > 0).mean():.1%})")

# Save
df.to_csv('zba_cases_cleaned.csv', index=False)
print(f"\n✅ Saved to zba_cases_cleaned.csv ({len(df.columns)} columns)")
