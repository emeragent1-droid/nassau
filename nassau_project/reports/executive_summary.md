# Executive Summary
## Factory Reallocation & Shipping Optimization — Nassau Candy Distributor

**Prepared for:** Government & Distribution-Sector Stakeholders
**Prepared by:** Pooja Verma | Unified Mentor Data Analyst Project

---

### The Challenge

Nassau Candy Distributor ships 15 confectionery products from 5 factories
to customers across the United States and Canada, currently using static,
manually-set factory assignments. There has been no way to test, in
advance, what would happen — to delivery speed or to profit — if a
product were reassigned to a different factory. As a result, some
products are shipping from factories that are far from where demand
actually is, quietly costing the business in both delivery time and
margin.

### What We Built

A predictive and prescriptive decision-support system that:

1. **Predicts shipping lead time** for any product, factory, destination
   region, and ship-mode combination, with **98% accuracy** (R² = 0.980).
2. **Clusters shipping routes** by performance, automatically flagging
   which factory-to-region routes are consistently slow or congested.
3. **Simulates factory reassignment** for every product against all 5
   factories, quantifying both the lead-time change and the profit
   impact of each option — before anything is actually moved.
4. **Generates ranked recommendations**, adjustable by a speed-vs-profit
   priority setting, so decision-makers can weight the trade-off
   according to current business priorities.

All of this is delivered through a live, interactive dashboard usable
directly by operations and logistics staff — no coding required.

### A Note on Data Quality

The source system's order/ship-date records could not be used directly —
they implied shipping times of several years with no logical relationship
to shipping speed tier, indicating a data-logging issue upstream rather
than a real operational pattern. Rather than either ignore this or
abandon the lead-time objective, this project built a transparent,
distance-and-service-tier-based estimate to stand in for it, so that
relative comparisons between factories remain trustworthy. **We recommend
Nassau Candy audit its order/ship timestamp logging** — once corrected,
this same system can be re-run on real observed data with no changes to
its underlying logic.

### Key Findings

- **Distance and shipping-mode service tier are, by a wide margin, the
  two biggest drivers of delivery time** — together accounting for ~95%
  of what determines how long a shipment takes. Product type and
  destination region matter far less by comparison.
- **Over a third of all orders (≈3,900 of 10,194) move through routes
  classified as "Slow" or "Consistently Slow / Congested"** — concentrated
  in three high-volume routes: Wicked Choccy's→Pacific,
  Lot's O' Nuts→Atlantic, and Lot's O' Nuts→Gulf.
- **The best-performing route (The Other Factory→Gulf) delivers in under
  3 days on average, while the worst (Wicked Choccy's→Pacific) takes
  over 7 — a gap of more than 2.5×** for comparable order volumes.
- In a representative case, reassigning **Wonka Bar - Milk Chocolate**
  from its current factory (Wicked Choccy's) to **The Other Factory**
  was found to improve *both* lead time (+3.4%) *and* profit (+2.6%)
  simultaneously — a genuine win-win the current static rulebook has no
  way to surface.
- Similar "win-win" reallocation opportunities were identified across
  the high-volume Chocolate product line, consistently pointing toward
  **Secret Factory** and **The Other Factory** as underused, well-placed
  alternatives to the current concentration at **Lot's O' Nuts** and
  **Wicked Choccy's**.

### Recommendation

Use this system as a standing decision-support tool for two concrete
actions: (1) **review the three highest-volume "Consistently Slow /
Congested" routes first**, since they affect the largest share of orders
and offer the largest potential improvement; and (2) **before any
manual factory-reassignment decision, run it through the simulator** to
confirm it is not just faster but also profit-neutral-or-better — the
Risk & Impact panel is built specifically to catch reassignments that
would help lead time but quietly hurt margin. In parallel, prioritize
fixing the underlying order/ship-date data logging so future analysis
can be grounded in observed rather than modeled shipping times.

### Deliverables

| Deliverable | Description |
|---|---|
| Research paper | Full methodology, data-quality handling, EDA, and model evaluation (`reports/research_paper.md`) |
| Streamlit dashboard | Live factory simulator, what-if comparison, recommendations, and risk panel (`app/streamlit_app.py`) |
| Trained model | Gradient Boosting lead-time predictor (`models/`) |
| Source code | Feature engineering, modeling, clustering, and simulation engine (`src/`) |
