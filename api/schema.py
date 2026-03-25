import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "boston_parcel_zoning.schema.json"

with open(SCHEMA_PATH) as f:
    PARCEL_SCHEMA = json.load(f)

from jsonschema import Draft202012Validator

schema_validator = Draft202012Validator(PARCEL_SCHEMA)

def validate_response(data: dict):
    errors = sorted(schema_validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        raise ValueError(
            "Schema validation failed: " +
            "; ".join(e.message for e in errors)
        )