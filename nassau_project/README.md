# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization

A predictive and prescriptive decision-support system that replaces
Nassau Candy's static factory-assignment rules with data-driven shipping
lead-time prediction, route clustering, and factory-reassignment scenario
simulation. Built for the **Unified Mentor** Data Analyst project.

📄 [Research Paper](reports/research_paper.md) · 📋 [Executive Summary](reports/executive_summary.md)

---

## 🎯 Project Overview

Nassau Candy ships 15 confectionery products from 5 factories to
customers across the US and Canada, currently assigned by static rule.
This project answers: **what should we change to improve performance?**

- **Predictive model** (Gradient Boosting, R² = 0.980) forecasting
  shipping lead time from product, origin factory, destination region,
  and ship mode.
- **Route clustering** identifying consistently slow/congested
  factory–region combinations.
- **A scenario simulation & recommendation engine** that tests every
  product against all 5 factories and ranks alternatives by lead-time
  reduction and profit impact, adjustable via a speed-vs-profit slider.
- **An interactive Streamlit dashboard** — factory simulator, what-if
  comparison, ranked recommendations, and a risk/impact panel.

> ⚠️ **Data-quality note:** the source order/ship-date fields are not
> usable for real lead-time calculation (see [research paper, §4](reports/research_paper.md#4-methodology--key-assumption-rebuilding-the-lead-time-target)
> for the full explanation). This project builds a transparent,
> distance + service-tier based lead-time estimate instead, documented
> in `src/reference_data.py`, so that relative factory comparisons stay
> meaningful.

## 📊 Dataset

`data/nassau_candy_distributor.csv` — 10,194 order records across 15
products, 3 divisions, 5 factories, and 4 regions, plus factory
coordinates and the current product→factory assignment table (both
supplied in the project brief and encoded in `src/reference_data.py`).

## 🗂️ Project Structure

```
nassau_project/
├── app/
│   └── streamlit_app.py              # Interactive dashboard (4 modules)
├── src/
│   ├── reference_data.py             # Factory coords, product-factory map, state centroids
│   ├── feature_engineering.py        # Distance, estimated lead time/cost, encoding
│   ├── eda.py                        # Exploratory analysis + chart generation
│   ├── modeling.py                   # Lead-time model training + route clustering
│   ├── simulation.py                 # Scenario simulation & recommendation engine
│   └── precompute_recommendations.py # Precomputes recs for all 15 products
├── models/                            # Saved model, metrics, clusters, recommendations (generated)
├── assets/                            # EDA charts (generated)
├── data/                              # Raw + engineered datasets (generated)
├── reports/
│   ├── research_paper.md              # Full methodology, EDA, results, discussion
│   └── executive_summary.md           # Stakeholder-facing summary
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone <your-repo-url>
cd nassau_project
pip install -r requirements.txt

# 2. Rebuild features, EDA charts, models, and recommendations from raw data
python src/feature_engineering.py
python src/eda.py
python src/modeling.py
python src/precompute_recommendations.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Select a product, region,
and ship mode, adjust the Speed ⟷ Profit priority slider, and explore the
**Factory Simulator**, **What-If Comparison**, **Recommendations**, and
**Risk & Impact** tabs.

## 🧠 Methodology

1. **Lead-time reconstruction** — the raw date fields are unusable (see
   note above), so lead time is estimated from haversine distance
   (factory ↔ destination state/province centroid) and ship-mode service
   tier assumptions, with realistic noise added.
2. **Feature engineering** — shipping cost & adjusted-profit estimates,
   ordinal ship-mode rank, one-hot encoded categoricals.
3. **Modeling** — Linear Regression, Random Forest, and Gradient Boosting
   compared via 80/20 split + 5-fold CV; Gradient Boosting selected
   (R² = 0.980).
4. **Route clustering** — K-Means (k=4) on lead time, distance, margin,
   and volume, labeled into interpretable performance tiers.
5. **Simulation & optimization** — every product is tested against all 5
   factories; recommendations are volume-weighted and blended by a
   configurable speed/profit priority.

Full details, charts, and results tables:
[`reports/research_paper.md`](reports/research_paper.md).

## 📈 Model Performance

| Target | Best Model | R² | RMSE |
|---|---|---:|---:|
| Estimated Shipping Lead Time | Gradient Boosting | 0.980 | 0.356 days |

## 🛠️ Tech Stack

Python · pandas · scikit-learn · Streamlit · Plotly · Matplotlib/Seaborn

## 📄 License

For educational / portfolio use as part of the Unified Mentor Data Analyst program.
