ZONING_RULES = {
    "2A": {
        "max_far": 0.8,
        "max_height": 35,
        "min_lot_size": 4000,

        # Density logic
        "min_lot_area_per_unit": 4000,   # sqft per unit
        # OR
        # "max_units_per_acre": 20,

        "allowed_uses": ["single_family", "two_family"],
        "conditional_uses": [],
        "forbidden_uses": ["commercial"]
    },

    "3F-4000": {
        "max_far": 1.0,
        "max_height": 35,
        "min_lot_size": 4000,

        "min_lot_area_per_unit": 1333,

        "allowed_uses": ["single_family", "two_family", "three_family"],
        "conditional_uses": [],
        "forbidden_uses": ["commercial"]
    }
}
