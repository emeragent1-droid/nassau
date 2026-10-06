"""
Exploratory Data Analysis for the Nassau Candy shipping/factory dataset.
Saves charts to /assets and summary stats to assets/eda_summary.json.
"""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_csv("data/nassau_features_eda.csv")
ASSETS = "assets"

# ---------------------------------------------------------------- 1. Raw lead time is unreliable (data-quality exhibit)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
sns.boxplot(data=df, x="Ship Mode", y="RawLeadTimeDays",
            order=["Same Day", "First Class", "Second Class", "Standard Class"], ax=axes[0])
axes[0].set_title("Raw (Ship Date − Order Date) by Ship Mode\n— not operationally plausible")
axes[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=df, x="Ship Mode", y="EstLeadTimeDays",
            order=["Same Day", "First Class", "Second Class", "Standard Class"], ax=axes[1])
axes[1].set_title("Engineered Estimated Lead Time by Ship Mode\n— distance + service-tier model")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{ASSETS}/01_raw_vs_estimated_leadtime.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 2. Distance distribution by factory
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="CurrentFactory", y="DistanceKM", ax=ax)
ax.set_title("Shipping Distance by Assigned Factory")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(f"{ASSETS}/02_distance_by_factory.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 3. Lead time vs distance, colored by ship mode
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=1),
                 x="DistanceKM", y="EstLeadTimeDays", hue="Ship Mode", alpha=0.4, ax=ax)
ax.set_title("Estimated Lead Time vs. Shipping Distance")
plt.tight_layout()
plt.savefig(f"{ASSETS}/03_leadtime_vs_distance.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 4. Profit margin by region
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=df, x="Region", y="AdjustedProfitMargin", estimator="mean", ax=ax,
            order=df.groupby("Region")["AdjustedProfitMargin"].mean().sort_values(ascending=False).index)
ax.set_title("Avg. Shipping-Cost-Adjusted Profit Margin by Region")
plt.tight_layout()
plt.savefig(f"{ASSETS}/04_adjusted_margin_by_region.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 5. Route (factory x region) performance heatmap
route_perf = df.groupby(["CurrentFactory", "Region"])["EstLeadTimeDays"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.heatmap(route_perf, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg Est. Lead Time (days)"})
ax.set_title("Route Performance: Avg Estimated Lead Time by Factory × Region")
plt.tight_layout()
plt.savefig(f"{ASSETS}/05_route_leadtime_heatmap.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 6. Shipping cost as % of gross profit, by division
fig, ax = plt.subplots(figsize=(7, 4.5))
df["ShipCostPctOfProfit"] = (df["EstShippingCost"] / df["Gross Profit"]).clip(0, 2)
sns.violinplot(data=df, x="Division", y="ShipCostPctOfProfit", ax=ax)
ax.set_title("Estimated Shipping Cost as a Share of Gross Profit, by Division")
ax.set_ylabel("Shipping Cost / Gross Profit")
plt.tight_layout()
plt.savefig(f"{ASSETS}/06_shipcost_share_of_profit.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 7. Correlation heatmap
key_cols = ["DistanceKM", "ShipModeRank", "Units", "Sales", "Cost", "Gross Profit",
            "EstLeadTimeDays", "EstShippingCost", "AdjustedProfit"]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
sns.heatmap(df[key_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix: Shipping & Profitability Drivers")
plt.tight_layout()
plt.savefig(f"{ASSETS}/07_correlation_heatmap.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- Summary stats
summary = {
    "n_orders": int(df.shape[0]),
    "n_products": int(df["Product ID"].nunique()),
    "n_factories": int(df["CurrentFactory"].nunique()),
    "raw_leadtime_mean_days": round(df["RawLeadTimeDays"].mean(), 1),
    "raw_leadtime_same_day_vs_standard_note": "Same Day raw lead time is NOT shorter than Standard Class — raw dates are unreliable (see research paper)",
    "est_leadtime_mean_days": round(df["EstLeadTimeDays"].mean(), 2),
    "avg_distance_km": round(df["DistanceKM"].mean(), 1),
    "avg_gross_profit": round(df["Gross Profit"].mean(), 2),
    "avg_est_shipping_cost": round(df["EstShippingCost"].mean(), 3),
    "avg_adjusted_profit": round(df["AdjustedProfit"].mean(), 2),
    "corr_distance_vs_leadtime": round(df["DistanceKM"].corr(df["EstLeadTimeDays"]), 3),
    "corr_distance_vs_shipcost": round(df["DistanceKM"].corr(df["EstShippingCost"]), 3),
    "worst_route_leadtime": route_perf.stack().idxmax(),
    "worst_route_leadtime_days": round(route_perf.stack().max(), 2),
    "best_route_leadtime": route_perf.stack().idxmin(),
    "best_route_leadtime_days": round(route_perf.stack().min(), 2),
}
with open(f"{ASSETS}/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("EDA complete. Charts saved to assets/. Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
