"""
Pure utility functions for PermitIQ API.

No dependency on shared state — these are stateless helpers.
normalize_address lives here so a regex bug cannot cascade into other modules.
"""

import re
import numpy as np
import pandas as pd


def safe_float(val, default=0.0):
    """Convert to float, replacing NaN/None with default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    """Convert to int, replacing NaN/None with default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else int(f)
    except (ValueError, TypeError):
        return default


def safe_str(val, default=""):
    """Convert to string, replacing NaN/None with default."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)


def _format_date(val) -> str:
    """Format a date value to a clean string. Handles NaN, None, various formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val).strip()
    if s.lower() in ('nan', 'none', 'nat', ''):
        return ''
    try:
        dt = pd.to_datetime(s)
        return dt.strftime('%b %d, %Y')  # e.g. "Jul 21, 2023"
    except Exception:
        return s  # return raw string, don't truncate


def _clean_case_date(row):
    """Extract a clean date string from case data."""
    for field in ('hearing_date', 'filing_date', 'date'):
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        date_str = str(val).strip()
        if date_str in ('', 'nan', 'None', 'NaT'):
            continue
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return date_str.strip()
    return ''


def _clean_case_address(row):
    """Clean OCR garbage from case addresses, trying multiple fallback fields."""
    for field in ('address_clean', 'address', 'property_address'):
        addr = str(row.get(field) or '')
        if not addr or addr in ('', 'nan', 'None', 'Unknown'):
            continue
        if len(addr) > 60 or '\n' in addr:
            continue
        if any(w in addr.lower() for w in ('record', 'conformity', 'hearing', 'board', 'appeal')):
            continue
        stripped = addr.replace(' ', '').replace('-', '')
        if stripped.isdigit():
            continue
        if not any(c.isalpha() for c in addr):
            continue
        return addr
    return 'Address not available'


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two (lat, lon) points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def normalize_address(addr):
    """Normalize an address string for better matching."""
    if not addr:
        return ""
    addr = str(addr).lower().strip()

    # Normalize em-dashes and en-dashes to hyphens
    addr = addr.replace('\u2014', '-').replace('\u2013', '-')

    # Remove ward info
    addr = re.sub(r',?\s*ward\s*\d+', '', addr)

    # Remove trailing "in the X neighborhood" / "in South Boston"
    addr = re.sub(r'\s+in\s+(the\s+)?.*$', '', addr)

    # Strip zip codes (Boston 021xx)
    addr = re.sub(r'\b0\d{4}\b', '', addr)

    # Normalize street suffixes
    addr = re.sub(r'\bstreet\b', 'st', addr)
    addr = re.sub(r'\bavenue\b', 'av', addr)
    addr = re.sub(r'\bave\b', 'av', addr)
    addr = re.sub(r'\broad\b', 'rd', addr)
    addr = re.sub(r'\bdrive\b', 'dr', addr)
    addr = re.sub(r'\bboulevard\b', 'blvd', addr)
    addr = re.sub(r'\blane\b', 'ln', addr)
    addr = re.sub(r'\bcourt\b', 'ct', addr)
    addr = re.sub(r'\bplace\b', 'pl', addr)
    addr = re.sub(r'\bterrace\b', 'ter', addr)

    # Strip neighborhood/city names after street suffix
    boston_places = [
        'jamaica plain', 'hyde park', 'west roxbury', 'east boston',
        'south boston', 'north end', 'south end', 'back bay', 'beacon hill',
        'mission hill',
        'dorchester', 'roxbury', 'mattapan', 'roslindale',
        'brighton', 'allston', 'charlestown', 'fenway', 'boston',
    ]
    for place in boston_places:
        addr = re.sub(
            r'((?:st|av|rd|dr|blvd|ln|ct|pl|ter|way|pk|sq)\b)\s+' + place + r'\b',
            r'\1', addr
        )
        addr = re.sub(r',\s*' + place + r'\s*$', '', addr)

    # "144A to 146 South ST" -> "144 - 146 South ST"
    addr = re.sub(r'(\d+[a-z]?)\s+to\s+(\d+)', r'\1 - \2', addr)

    # Remove suffix letters from house numbers: "69R" -> "69", "146A" -> "146"
    addr = re.sub(r'\b(\d+)[a-z]\b', r'\1', addr)

    # Normalize directionals
    addr = re.sub(r'\beast\b', 'e', addr)
    addr = re.sub(r'\bwest\b', 'w', addr)
    addr = re.sub(r'\bnorth\b', 'n', addr)
    addr = re.sub(r'\bsouth\b', 's', addr)
    addr = re.sub(r'\bsaint\b', 'st', addr)

    # Collapse whitespace
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr
