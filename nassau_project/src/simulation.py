"""
Factory Reallocation Scenario Simulation & Recommendation Engine.

For a given product (or product + destination region), simulates
reassigning production to each candidate factory, predicts the resulting
lead time with the trained model, estimates shipping cost and profit
impact from the distance change, and ranks the alternatives.

KPIs computed per recommendation:
  - Lead Time Reduction (%)      vs. current factory
  - Profit Impact ($ and %)      vs. current factory
  - Scenario Confidence Score    based on model CV R^2 and route sample size
"""

import os

import numpy as np
import pandas as pd
import joblib

from reference_data import FACTORIES, STATE_CENTROIDS, SHIP_MODE_PARAMS
from feature_engineering import haversine_km

# Anchor all data/model paths to the project root (parent of this file's
# src/ directory), regardless of the working directory the caller is
# running from. This makes the module safe to import from the Streamlit
# app on any host (local, Streamlit Cloud, etc.).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)


def load_model():
    model = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model_leadtime.pkl"))
    feature_cols = joblib.load(os.path.join(PROJECT_ROOT, "models", "feature_columns_leadtime.pkl"))
    return model, feature_cols


def _build_feature_row(distance_km, ship_mode, units, sales, cost, division, region, factory, feature_cols):
    ship_mode_rank = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}[ship_mode]
    row = {c: 0.0 for c in feature_cols}
    row["DistanceKM"] = distance_km
    row["ShipModeRank"] = ship_mode_rank
    row["Units"] = units
    row["Sales"] = sales
    row["Cost"] = cost
    row["CostToSalesRatio"] = cost / sales if sales else 0
    for key, col in [
        (f"Div_{division}", None), (f"Region_{region}", None),
        (f"Ship_{ship_mode}", None), (f"Factory_{factory}", None),
    ]:
        if key in row:
            row[key] = 1.0
    return pd.DataFrame([row])[feature_cols]


def predict_lead_time(distance_km, ship_mode, units, sales, cost, division, region, factory, model, feature_cols):
    X = _build_feature_row(distance_km, ship_mode, units, sales, cost, division, region, factory, feature_cols)
    return float(model.predict(X)[0])


def simulate_reassignment(
    product_division: str, region: str, ship_mode: str,
    units: float, sales: float, cost: float,
    current_factory: str, model, feature_cols,
    cv_r2: float = 0.98,
) -> pd.DataFrame:
    """Simulate assigning this product/order profile to every candidate factory."""
    dest_lat, dest_lon = STATE_CENTROIDS_REGION_PROXY(region)
    rows = []
    for factory, coords in FACTORIES.items():
        dist = haversine_km(coords["lat"], coords["lon"], dest_lat, dest_lon)
        lead_time = predict_lead_time(dist, ship_mode, units, sales, cost,
                                       product_division, region, factory, model, feature_cols)
        rate = SHIP_MODE_PARAMS[ship_mode]["cost_per_km_per_unit"]
        ship_cost = dist * rate * units
        adj_profit = (sales - cost) - ship_cost
        rows.append({
            "Factory": factory, "DistanceKM": round(dist, 1),
            "PredictedLeadTimeDays": round(lead_time, 2),
            "EstShippingCost": round(ship_cost, 3),
            "AdjustedProfit": round(adj_profit, 3),
        })
    result = pd.DataFrame(rows)

    current = result[result["Factory"] == current_factory].iloc[0]
    result["LeadTimeReductionPct"] = round(
        (current["PredictedLeadTimeDays"] - result["PredictedLeadTimeDays"]) / current["PredictedLeadTimeDays"] * 100, 2
    )
    result["ProfitImpact"] = round(result["AdjustedProfit"] - current["AdjustedProfit"], 3)
    result["ProfitImpactPct"] = round(
        (result["AdjustedProfit"] - current["AdjustedProfit"]) / abs(current["AdjustedProfit"]) * 100, 2
    ) if current["AdjustedProfit"] != 0 else 0.0
    result["IsCurrent"] = result["Factory"] == current_factory
    result["ScenarioConfidenceScore"] = round(cv_r2 * 100, 1)  # simple: model CV R^2 as % confidence

    return result.sort_values("PredictedLeadTimeDays")


_REGION_CENTROIDS = None


def STATE_CENTROIDS_REGION_PROXY(region: str):
    """
    Approximate a destination point for a Region by averaging the centroids
    of the states/provinces that belong to it in the dataset. Computed once
    and cached in-process for consistency across calls.
    """
    global _REGION_CENTROIDS
    if _REGION_CENTROIDS is None:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "nassau_features_eda.csv"))
        _REGION_CENTROIDS = df.groupby("Region")[["DestLat", "DestLon"]].mean().apply(tuple, axis=1).to_dict()
    return _REGION_CENTROIDS[region]


def rank_recommendations_for_product(product_name: str, division: str, top_n: int = 3):
    """
    Aggregate a product's historical orders by region, and for each region
    simulate reassignment, ranking candidate factories by a blended score
    of lead-time reduction and profit impact. Returns top-N distinct
    factory recommendations across the product's regional footprint.
    """
    from reference_data import PRODUCT_FACTORY
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "nassau_features_eda.csv"))
    prod_df = df[df["Product Name"] == product_name]
    current_factory = PRODUCT_FACTORY[product_name]
    model, feature_cols = load_model()

    all_scenarios = []
    for region, grp in prod_df.groupby("Region"):
        avg_units = grp["Units"].mean()
        avg_sales = grp["Sales"].mean()
        avg_cost = grp["Cost"].mean()
        ship_mode = grp["Ship Mode"].mode().iloc[0]
        sim = simulate_reassignment(division, region, ship_mode, avg_units, avg_sales, avg_cost,
                                     current_factory, model, feature_cols)
        sim["Region"] = region
        sim["OrderVolume"] = len(grp)
        all_scenarios.append(sim)

    combined = pd.concat(all_scenarios, ignore_index=True)
    # Volume-weighted score per factory across all regions this product ships to
    combined["WeightedScore"] = (
        0.6 * combined["LeadTimeReductionPct"] + 0.4 * combined["ProfitImpactPct"]
    ) * combined["OrderVolume"]
    factory_scores = combined.groupby("Factory").agg(
        TotalWeightedScore=("WeightedScore", "sum"),
        AvgLeadTimeReductionPct=("LeadTimeReductionPct", "mean"),
        AvgProfitImpactPct=("ProfitImpactPct", "mean"),
        TotalOrderVolume=("OrderVolume", "sum"),
    ).reset_index()
    factory_scores = factory_scores[factory_scores["Factory"] != current_factory]
    factory_scores = factory_scores.sort_values("TotalWeightedScore", ascending=False).head(top_n)
    factory_scores.insert(0, "Product", product_name)
    factory_scores.insert(1, "CurrentFactory", current_factory)
    return factory_scores, combined


if __name__ == "__main__":
    recs, detail = rank_recommendations_for_product("Wonka Bar - Milk Chocolate", "Chocolate")
    print("Top factory reassignment recommendations for 'Wonka Bar - Milk Chocolate':")
    print(recs.to_string(index=False))
