"""
Nassau Candy — Factory Reallocation & Shipping Optimization Dashboard.

Run with:  streamlit run app/streamlit_app.py   (from the project root)
"""

import sys
import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to the repo root, regardless of the directory the
# app was launched from (important for cloud deployment where the working
# directory isn't guaranteed to be the project root).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))
from reference_data import FACTORIES, PRODUCT_FACTORY, SHIP_MODE_PARAMS
from simulation import simulate_reassignment, load_model, rank_recommendations_for_product

st.set_page_config(page_title="Nassau Candy Shipping Optimizer", layout="wide", page_icon="🍬")


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ROOT_DIR, "data", "nassau_features_eda.csv"))


@st.cache_resource
def load_models():
    model, feature_cols = load_model()
    with open(os.path.join(ROOT_DIR, "models", "metrics_leadtime.json")) as f:
        metrics = json.load(f)
    route_clusters = pd.read_csv(os.path.join(ROOT_DIR, "models", "route_clusters.csv"))
    all_recs = pd.read_csv(os.path.join(ROOT_DIR, "models", "all_product_recommendations.csv"))
    return model, feature_cols, metrics, route_clusters, all_recs


df = load_data()
model, feature_cols, metrics, route_clusters, all_recs = load_models()
best_model_name = metrics["best_model"]
model_r2 = metrics["results"][best_model_name]["R2"]

st.title("🍬 Nassau Candy — Factory Reallocation & Shipping Optimizer")
st.caption(
    "Simulate factory reassignment scenarios, predict lead-time and profit impact, "
    "and get ranked recommendations — replacing static factory-assignment rules with "
    "data-driven decision support."
)

st.info(
    "**Note on lead time:** the source order/ship-date fields in this dataset are not "
    "operationally reliable (see the research paper, Section 4). Lead-time figures shown "
    "here are a transparent distance + service-tier estimate built for this project, "
    "so relative comparisons between factories are meaningful even though absolute "
    "day counts are modeled rather than observed.",
    icon="ℹ️",
)

# ----------------------------------------------------------------- Sidebar controls
st.sidebar.header("Scenario Controls")
product = st.sidebar.selectbox("Product", sorted(PRODUCT_FACTORY.keys()))
current_factory = PRODUCT_FACTORY[product]
division = df.loc[df["Product Name"] == product, "Division"].mode().iloc[0]

region = st.sidebar.selectbox("Destination Region", sorted(df["Region"].unique()))
ship_mode = st.sidebar.selectbox("Ship Mode", ["Same Day", "First Class", "Second Class", "Standard Class"],
                                  index=3)

prod_region_df = df[(df["Product Name"] == product) & (df["Region"] == region)]
default_units = prod_region_df["Units"].mean() if len(prod_region_df) else df["Units"].mean()
default_sales = prod_region_df["Sales"].mean() if len(prod_region_df) else df["Sales"].mean()
default_cost = prod_region_df["Cost"].mean() if len(prod_region_df) else df["Cost"].mean()

st.sidebar.markdown("---")
units = st.sidebar.slider("Order Size (units)", 1, 15, int(round(default_units)))
priority = st.sidebar.slider(
    "Optimization Priority: Speed ⟷ Profit", 0, 100, 50,
    help="0 = prioritize lead-time reduction only, 100 = prioritize profit impact only"
)
speed_weight = (100 - priority) / 100
profit_weight = priority / 100

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{best_model_name}** · Test R² = **{model_r2:.3f}**")

# ----------------------------------------------------------------- Run scenario
sim = simulate_reassignment(
    division, region, ship_mode, units, default_sales, default_cost,
    current_factory, model, feature_cols, cv_r2=metrics["results"][best_model_name]["CV_R2_mean"],
)
sim["BlendedScore"] = speed_weight * sim["LeadTimeReductionPct"] + profit_weight * sim["ProfitImpactPct"]
sim_sorted = sim.sort_values("BlendedScore", ascending=False)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Simulator", "🔀 What-If Comparison", "📋 Recommendations", "⚠️ Risk & Impact"
])

# ============================================================== TAB 1: Factory simulator
with tab1:
    st.subheader(f"Predicted Performance by Factory — {product}")
    st.caption(f"Currently produced at **{current_factory}** · Division: {division} · "
               f"Destination: {region} · Ship mode: {ship_mode}")

    c1, c2, c3 = st.columns(3)
    cur_row = sim[sim["IsCurrent"]].iloc[0]
    best_row = sim_sorted.iloc[0]
    c1.metric("Current Factory Lead Time", f"{cur_row['PredictedLeadTimeDays']:.2f} days")
    fastest_gap = cur_row["PredictedLeadTimeDays"] - sim["PredictedLeadTimeDays"].min()
    c2.metric("Fastest Alternative", f"{sim['PredictedLeadTimeDays'].min():.2f} days",
              delta=f"{fastest_gap:.2f} days faster" if fastest_gap > 0 else "same as current")
    c3.metric("Most Profitable Option", f"${sim['AdjustedProfit'].max():.2f}",
              delta=f"${(sim['AdjustedProfit'].max()-cur_row['AdjustedProfit']):.2f} vs current")

    fig = px.bar(sim.sort_values("PredictedLeadTimeDays"), x="Factory", y="PredictedLeadTimeDays",
                 color="IsCurrent", color_discrete_map={True: "#2E5EAA", False: "#8896A6"},
                 title="Predicted Lead Time by Factory", labels={"PredictedLeadTimeDays": "Lead Time (days)"})
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        sim[["Factory", "DistanceKM", "PredictedLeadTimeDays", "EstShippingCost", "AdjustedProfit", "IsCurrent"]]
        .sort_values("PredictedLeadTimeDays").rename(columns={"IsCurrent": "Current Factory?"}),
        use_container_width=True, hide_index=True,
    )

# ============================================================== TAB 2: What-if comparison
with tab2:
    st.subheader("Current vs. Recommended Assignment")
    recommended = sim_sorted[~sim_sorted["IsCurrent"]].iloc[0]

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### 🏭 Current: {current_factory}")
        st.metric("Lead Time", f"{cur_row['PredictedLeadTimeDays']:.2f} days")
        st.metric("Distance", f"{cur_row['DistanceKM']:.0f} km")
        st.metric("Adjusted Profit / order", f"${cur_row['AdjustedProfit']:.2f}")
    with colB:
        st.markdown(f"### ✅ Recommended: {recommended['Factory']}")
        st.metric("Lead Time", f"{recommended['PredictedLeadTimeDays']:.2f} days",
                   delta=f"{recommended['LeadTimeReductionPct']:.1f}%")
        st.metric("Distance", f"{recommended['DistanceKM']:.0f} km")
        st.metric("Adjusted Profit / order", f"${recommended['AdjustedProfit']:.2f}",
                   delta=f"{recommended['ProfitImpactPct']:.1f}%")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Current", x=["Lead Time (days)", "Adjusted Profit ($)"],
                           y=[cur_row["PredictedLeadTimeDays"], cur_row["AdjustedProfit"]], marker_color="#8896A6"))
    fig2.add_trace(go.Bar(name="Recommended", x=["Lead Time (days)", "Adjusted Profit ($)"],
                           y=[recommended["PredictedLeadTimeDays"], recommended["AdjustedProfit"]],
                           marker_color="#3F9142"))
    fig2.update_layout(barmode="group", height=380, title="Current vs. Recommended — Key Metrics")
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================== TAB 3: Recommendations
with tab3:
    st.subheader("Ranked Factory Reassignment Recommendations")
    st.caption("Blended score = weighted combination of lead-time reduction and profit impact, "
               "using the Speed ⟷ Profit slider in the sidebar. Volume-weighted across this "
               "product's historical regional footprint.")
    if st.button("🔍 Generate Full Recommendation (all regions this product ships to)", type="primary"):
        with st.spinner("Running scenario simulation across all regions..."):
            recs, detail = rank_recommendations_for_product(product, division, top_n=4)
        st.dataframe(recs.drop(columns=["Product", "CurrentFactory"]), use_container_width=True, hide_index=True)
        fig3 = px.bar(recs, x="Factory", y="TotalWeightedScore", color="AvgProfitImpactPct",
                      color_continuous_scale="RdYlGn", title=f"Alternative Factories for {product}")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        # Show precomputed version for instant feedback
        precomputed = all_recs[all_recs["Product"] == product].drop(columns=["Product", "CurrentFactory"])
        st.dataframe(precomputed, use_container_width=True, hide_index=True)
        st.caption("Showing precomputed recommendations. Click the button above to recompute live.")

    st.markdown("---")
    st.subheader("Route Performance Clusters")
    st.caption("Factory × Region routes grouped by performance similarity (lead time, distance, "
               "margin, volume).")
    fig4 = px.scatter(route_clusters, x="AvgDistance", y="AvgLeadTime", size="OrderVolume",
                      color="PerformanceLabel", hover_data=["CurrentFactory", "Region", "AvgAdjMargin"],
                      title="Route Clusters: Distance vs. Lead Time (bubble size = order volume)")
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================== TAB 4: Risk & impact
with tab4:
    st.subheader("Risk & Impact Panel")

    high_risk = sim[(sim["ProfitImpactPct"] < -5) & (~sim["IsCurrent"])]
    if len(high_risk):
        st.warning(f"⚠️ {len(high_risk)} candidate factory option(s) would **reduce** profit by more "
                   f"than 5% for this scenario: {', '.join(high_risk['Factory'].tolist())}. "
                   "Not recommended despite any lead-time gains.")
    else:
        st.success("✅ No high-risk (profit-negative) reassignment options detected for this scenario.")

    st.markdown("##### Profit Impact vs. Lead Time Trade-off")
    fig5 = px.scatter(sim, x="LeadTimeReductionPct", y="ProfitImpactPct", text="Factory",
                      color="IsCurrent", color_discrete_map={True: "#2E5EAA", False: "#B4530A"},
                      title="Every Factory Option: Lead-Time Reduction vs. Profit Impact")
    fig5.add_hline(y=0, line_dash="dash", line_color="gray")
    fig5.add_vline(x=0, line_dash="dash", line_color="gray")
    fig5.update_traces(textposition="top center")
    fig5.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Top-right quadrant = win-win (faster AND more profitable). "
               "Bottom-right = faster but costs profit. Top-left = cheaper/more profit but slower.")

    st.markdown("##### Portfolio-Wide Congestion Check")
    congested = route_clusters[route_clusters["PerformanceLabel"].isin(["Slow", "Consistently Slow / Congested"])]
    st.dataframe(
        congested[["CurrentFactory", "Region", "AvgLeadTime", "AvgDistance", "OrderVolume", "PerformanceLabel"]]
        .sort_values("AvgLeadTime", ascending=False),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")
st.caption(
    f"Model: {best_model_name} · Trained on {df.shape[0]:,} historical orders · "
    f"Test R² = {model_r2:.3f} · Unified Mentor — Nassau Candy Distributor project"
)
