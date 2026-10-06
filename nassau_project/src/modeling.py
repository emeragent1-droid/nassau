"""
Model development for the Nassau Candy shipping lead-time prediction task,
plus route clustering.

Predicts EstLeadTimeDays (engineered target — see feature_engineering.py
docstring) given: product/division, origin factory, destination region,
and ship mode (+ distance and order size, which are what actually drive
real-world transit time and are derivable from those fields).
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

from feature_engineering import FEATURE_COLUMNS, TARGET_LEAD_TIME

DATA_PATH = "data/nassau_features.csv"
MODELS_DIR = "models"


def get_feature_matrix(df: pd.DataFrame):
    cat_cols = [c for c in df.columns if c.startswith(("Div_", "Region_", "Ship_", "Factory_"))]
    feature_cols = FEATURE_COLUMNS + cat_cols
    X = df[feature_cols].astype(float)
    return X, feature_cols


def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_lead_time_models():
    df = pd.read_csv(DATA_PATH)
    X, feature_cols = get_feature_matrix(df)
    y = df[TARGET_LEAD_TIME].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                        learning_rate=0.05, random_state=42),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        cv = cross_val_score(model, X, y, cv=5, scoring="r2")
        metrics["CV_R2_mean"] = float(cv.mean())
        metrics["CV_R2_std"] = float(cv.std())
        results[name] = metrics
        fitted[name] = model
        print(f"{name:18s} RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  "
              f"R2={metrics['R2']:.4f}  CV_R2={metrics['CV_R2_mean']:.4f}")

    # Select by R2, tie-break toward simpler/more interpretable model if close
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    importance = None
    if best_name == "LinearRegression":
        coefs = best_model.named_steps["model"].coef_
        importance = dict(zip(feature_cols, coefs.tolist()))
    elif hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feature_cols, best_model.feature_importances_.tolist()))

    joblib.dump(best_model, f"{MODELS_DIR}/best_model_leadtime.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_columns_leadtime.pkl")
    with open(f"{MODELS_DIR}/metrics_leadtime.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    if importance:
        pd.Series(importance).sort_values(key=abs, ascending=False).to_csv(
            f"{MODELS_DIR}/feature_importance_leadtime.csv", header=["importance"])

    return results, best_name, best_model, feature_cols


def cluster_routes(n_clusters: int = 4):
    """
    Cluster Factory x Region routes by performance similarity: avg lead
    time, avg distance, avg adjusted profit margin, order volume.
    """
    df = pd.read_csv("data/nassau_features_eda.csv")
    route_stats = df.groupby(["CurrentFactory", "Region"]).agg(
        AvgLeadTime=("EstLeadTimeDays", "mean"),
        AvgDistance=("DistanceKM", "mean"),
        AvgAdjMargin=("AdjustedProfitMargin", "mean"),
        OrderVolume=("Row ID", "count"),
    ).reset_index()

    X = route_stats[["AvgLeadTime", "AvgDistance", "AvgAdjMargin", "OrderVolume"]]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    route_stats["ClusterID"] = km.fit_predict(X_scaled)

    # Label clusters by their lead-time rank so labels are interpretable
    cluster_order = route_stats.groupby("ClusterID")["AvgLeadTime"].mean().sort_values()
    labels = ["Fast & Efficient", "Moderate", "Slow", "Consistently Slow / Congested"]
    label_map = {cid: labels[min(i, len(labels) - 1)] for i, cid in enumerate(cluster_order.index)}
    route_stats["PerformanceLabel"] = route_stats["ClusterID"].map(label_map)

    route_stats = route_stats.sort_values("AvgLeadTime", ascending=False)
    route_stats.to_csv(f"{MODELS_DIR}/route_clusters.csv", index=False)
    print("\nRoute clustering complete:")
    print(route_stats.to_string(index=False))
    return route_stats


if __name__ == "__main__":
    print("=" * 60)
    print("Training lead-time prediction models")
    print("=" * 60)
    train_lead_time_models()

    print("\n" + "=" * 60)
    print("Clustering routes by performance")
    print("=" * 60)
    cluster_routes()
