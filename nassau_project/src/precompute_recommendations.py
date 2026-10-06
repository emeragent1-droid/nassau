"""
Precompute factory-reassignment recommendations for every product in the
catalog, so the Streamlit dashboard can load them instantly rather than
re-running the simulation on every interaction.
"""

import pandas as pd
from reference_data import PRODUCT_FACTORY, DIVISION_OF
from simulation import rank_recommendations_for_product

if __name__ == "__main__":
    df = pd.read_csv("data/nassau_features_eda.csv")
    all_recs = []
    all_detail = []
    for product in PRODUCT_FACTORY:
        division = df.loc[df["Product Name"] == product, "Division"].mode().iloc[0]
        recs, detail = rank_recommendations_for_product(product, division, top_n=4)
        all_recs.append(recs)
        detail["Product"] = product
        all_detail.append(detail)
        print(f"{product:38s} -> best alt: {recs.iloc[0]['Factory']} "
              f"(score {recs.iloc[0]['TotalWeightedScore']:.0f})" if len(recs) else f"{product}: no alternatives")

    recs_all = pd.concat(all_recs, ignore_index=True)
    detail_all = pd.concat(all_detail, ignore_index=True)
    recs_all.to_csv("models/all_product_recommendations.csv", index=False)
    detail_all.to_csv("models/all_product_scenario_detail.csv", index=False)
    print(f"\nSaved {len(recs_all)} recommendation rows and {len(detail_all)} scenario-detail rows.")
