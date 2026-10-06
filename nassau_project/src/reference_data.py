"""
Static reference data for the Nassau Candy factory network.

Sources:
 - Factory coordinates & product->factory assignments: provided in the
   project brief ("Factories Co-ordinates" / "Products and Factories
   Correlation" tables).
 - State/Province centroids: standard public-domain approximate geographic
   centroids for US states and Canadian provinces, used ONLY to estimate
   shipping distance for this project (see reports/research_paper.md,
   Methodology & Assumptions, for why this approximation is necessary and
   how it's used).
"""

FACTORIES = {
    "Lot's O' Nuts":       {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's":     {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack":         {"lat": 48.11914,  "lon": -96.18115},
    "Secret Factory":      {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory":   {"lat": 35.1175,   "lon": -89.971107},
}

# Current (baseline) product -> factory assignment, as given in the brief.
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

DIVISION_OF = {
    "Wonka Bar - Nutty Crunch Surprise": "Chocolate",
    "Wonka Bar - Fudge Mallows": "Chocolate",
    "Wonka Bar -Scrumdiddlyumptious": "Chocolate",
    "Wonka Bar - Milk Chocolate": "Chocolate",
    "Wonka Bar - Triple Dazzle Caramel": "Chocolate",
    "Laffy Taffy": "Sugar",
    "SweeTARTS": "Sugar",
    "Nerds": "Sugar",
    "Fun Dip": "Sugar",
    "Fizzy Lifting Drinks": "Other",
    "Everlasting Gobstopper": "Sugar",
    "Hair Toffee": "Sugar",
    "Lickable Wallpaper": "Other",
    "Wonka Gum": "Other",
    "Kazookles": "Other",
}

# Approximate geographic centroids (lat, lon) for US states/DC and Canadian
# provinces/territories present in the dataset. Used only for distance
# estimation, not for any precision-critical purpose.
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817), "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074), "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canadian provinces / territories
    "Alberta": (53.933271, -116.576504), "British Columbia": (53.726669, -127.647621),
    "Manitoba": (53.760860, -98.813873), "New Brunswick": (46.565314, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435), "Nova Scotia": (44.681892, -63.744312),
    "Ontario": (51.253775, -85.323214), "Prince Edward Island": (46.510712, -63.416812),
    "Quebec": (52.939916, -73.549136), "Saskatchewan": (52.935397, -106.450864),
    "Northwest Territories": (64.825546, -124.845833), "Nunavut": (70.299774, -83.107277),
    "Yukon": (64.282326, -135.000000),
}

# Ship-mode logistics assumptions, used only to build an explainable transit
# time / cost model (see research paper for justification).
#   base_days: fixed handling/dispatch time
#   km_per_day: assumed line-haul travel speed for that service tier
#   cost_per_km_per_unit: assumed per-unit shipping cost rate ($/km/unit)
SHIP_MODE_PARAMS = {
    "Same Day":       {"base_days": 0.3, "km_per_day": 2200, "cost_per_km_per_unit": 0.00075},
    "First Class":    {"base_days": 1.0, "km_per_day": 1400, "cost_per_km_per_unit": 0.00045},
    "Second Class":   {"base_days": 2.0, "km_per_day": 900,  "cost_per_km_per_unit": 0.00028},
    "Standard Class": {"base_days": 3.0, "km_per_day": 550,  "cost_per_km_per_unit": 0.00015},
}
