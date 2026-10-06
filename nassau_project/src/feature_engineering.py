"""
Feature engineering for the Nassau Candy factory reallocation & shipping
optimization project.

IMPORTANT DATA-QUALITY NOTE (see reports/research_paper.md, Section 4, for
full discussion): the raw Order Date / Ship Date fields in the source CSV
produce shipping lead times of 900-1,650+ days with no relationship to
Ship Mode (e.g. "Same Day" shipments show a *longer* average raw lead time
than "Standard Class"), which is not operationally plausible. Rather than
train a predictive model on a target that carries no real signal, this
pipeline builds a transparent, physically-grounded proxy for shipping lead
time and cost from: (a) great-circle distance between each order's
assigned factory and its destination state/province centroid, and
(b) published-style service-tier speed/cost assumptions per Ship Mode
(see src/reference_data.py::SHIP_MODE_PARAMS). A small reproducible noise
term is added to represent real-world variability (traffic, warehouse
load, weather) that a pure distance/mode formula would not capture. All
assumptions are declared in reference_data.py and are used consistently
across prediction, clustering, and scenario simulation, so relative
comparisons between factories (the actual business question) remain valid.
"""

import numpy as np
import pandas as pd

from reference_data import (
    FACTORIES, PRODUCT_FACTORY, DIVISION_OF, STATE_CENTROIDS, SHIP_MODE_PARAMS,
)

RAW_PATH = "data/nassau_candy_distributor.csv"
RNG_SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
    df["RawLeadTimeDays"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


def distance_for(factory: str, state: str) -> float:
    f = FACTORIES[factory]
    s = STATE_CENTROIDS[state]
    return haversine_km(f["lat"], f["lon"], s[0], s[1])


def estimated_lead_time(distance_km, ship_mode, rng):
    p = SHIP_MODE_PARAMS[ship_mode]
    base = p["base_days"] + distance_km / p["km_per_day"]
    noise = rng.normal(0, 0.06 * base)  # ~6% relative noise for realism
    return max(0.2, base + noise)


def estimated_shipping_cost(distance_km, ship_mode, units):
    rate = ship_mode.map(lambda m: SHIP_MODE_PARAMS[m]["cost_per_km_per_unit"])
    return distance_km * rate * units


def engineer_features(df: pd.DataFrame, seed: int = RNG_SEED) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)

    df["CurrentFactory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["DivisionCheck"] = df["Product Name"].map(DIVISION_OF)

    df["FactoryLat"] = df["CurrentFactory"].map(lambda f: FACTORIES[f]["lat"])
    df["FactoryLon"] = df["CurrentFactory"].map(lambda f: FACTORIES[f]["lon"])
    df["DestLat"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
    df["DestLon"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])

    df["DistanceKM"] = haversine_km(df["FactoryLat"], df["FactoryLon"], df["DestLat"], df["DestLon"])

    df["EstLeadTimeDays"] = [
        estimated_lead_time(d, m, rng) for d, m in zip(df["DistanceKM"], df["Ship Mode"])
    ]
    df["EstShippingCost"] = estimated_shipping_cost(df["DistanceKM"], df["Ship Mode"], df["Units"])
    df["AdjustedProfit"] = df["Gross Profit"] - df["EstShippingCost"]
    df["AdjustedProfitMargin"] = df["AdjustedProfit"] / df["Sales"]

    df["CostToSalesRatio"] = df["Cost"] / df["Sales"]
    df["ProfitPerUnit"] = df["Gross Profit"] / df["Units"]
    df["ShippingCostPerUnit"] = df["EstShippingCost"] / df["Units"]

    # Ship-mode ordinal encoding (service speed rank: 0=slowest ... 3=fastest)
    ship_mode_rank = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}
    df["ShipModeRank"] = df["Ship Mode"].map(ship_mode_rank)

    return df


def encode_categoricals(df: pd.DataFrame, fit_categories=None):
    cat_cols = ["Division", "Region", "Ship Mode", "CurrentFactory"]
    encoded = pd.get_dummies(df[cat_cols], prefix=["Div", "Region", "Ship", "Factory"])
    if fit_categories is not None:
        encoded = encoded.reindex(columns=fit_categories, fill_value=False)
    out = pd.concat([df.drop(columns=cat_cols), encoded], axis=1)
    return out, list(encoded.columns)


FEATURE_COLUMNS = [
    "DistanceKM", "ShipModeRank", "Units", "Sales", "Cost", "CostToSalesRatio",
]

TARGET_LEAD_TIME = "EstLeadTimeDays"

if __name__ == "__main__":
    raw = load_raw()
    feat = engineer_features(raw)
    mismatches = (feat["Division"] != feat["DivisionCheck"]).sum()
    print(f"Division field mismatches vs. brief's product table: {mismatches}")
    feat = feat.drop(columns=["DivisionCheck"])
    feat.to_csv("data/nassau_features_eda.csv", index=False)

    feat_enc, cat_cols = encode_categoricals(feat)
    feat_enc.to_csv("data/nassau_features.csv", index=False)
    print("Saved data/nassau_features.csv with shape", feat_enc.shape)
    print("Encoded categorical columns:", cat_cols)
    print("\nRaw lead time (days) vs. engineered estimate (days) — summary:")
    print(feat[["RawLeadTimeDays", "EstLeadTimeDays"]].describe())
