import re


ZONING_PATTERNS = [
    r"(Article\s*\d+[, ]*Section\s*\d+)",
    r"(Section\s*\d+[, ]*Article\s*\d+)",
    r"(Art\.?\s*\d+[, ]*Sec\.?\s*\d+)",
    r"(Article\s*\d+)",
    r"(Section\s*\d+)"
]


DECISION_PATTERNS = [
    "GRANTED",
    "DENIED",
    "REFUSED",
    "REJECTED",
    "APPEAL DENIED",
    "APPEAL SUSTAINED"
]


def extract_zoning(text):
    for pattern in ZONING_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_decision(text):
    for decision in DECISION_PATTERNS:
        if decision in text.upper():
            return decision
    return None


def clean_address(text):
    """
    Extracts first clean address-like string
    """
    match = re.search(r"\d{1,5}\s+[A-Za-z0-9\s]+", text)
    if match:
        address = match.group(0)
        address = address.split("\n")[0]
        return address.strip()
    return None


def split_cases(text):
    """
    Primary case splitter using BOA pattern
    """
    pattern = r"(BOA[\s\-]?\d{6,7}.*?)((?=BOA[\s\-]?\d{6,7})|$)"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return [m[0].strip() for m in matches if len(m[0]) > 100]


def parse_cases(text, source_pdf):
    cases = []

    split_texts = split_cases(text)

    for case_text in split_texts:

        # Case number
        case_number_match = re.search(r"(BOA[\s\-]?\d{6,7})", case_text, re.IGNORECASE)
        case_number = case_number_match.group(1).replace(" ", "").upper() if case_number_match else None

        # Address
        address = clean_address(case_text)

        # Zoning
        zoning = extract_zoning(case_text)

        # Decision
        decision = extract_decision(case_text)

        cases.append({
            "case_number": case_number,
            "address": address,
            "zoning": zoning,
            "decision": decision,
            "raw_text": case_text,
            "source_pdf": source_pdf
        })

    return cases