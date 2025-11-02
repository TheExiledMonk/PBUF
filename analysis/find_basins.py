#!/usr/bin/env python3
"""
find_basins.py – Basin / island detection using scikit-learn
--------------------------------------------------------------
Usage:
    python find_basins.py path/to/grid_*.csv [chi2_threshold] [--method=dbscan|kmeans]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_and_filter(csv_path, chi2_threshold=None):
    """Load CSV and filter by χ² threshold if given."""
    df = pd.read_csv(csv_path)
    if chi2_threshold:
        df = df[df["chi2_total"] < chi2_threshold]
        print(f"→ {len(df)} models below χ²_total < {chi2_threshold}")
    else:
        print(f"Loaded {len(df)} models.")

    # Select numeric columns and drop NaNs
    df = df.select_dtypes(include=["number"]).dropna(axis=0)
    return df


def detect_feature_set(df):
    """Automatically detect which cosmological parameters are available."""
    all_features = ["H0", "Om0", "alpha", "Rmax", "k_sat", "Or0", "Ok0"]
    features = [f for f in all_features if f in df.columns]
    print(f"Detected features: {features}")
    return features


def cluster_data(df, features, method="dbscan", eps=None, min_samples=10, k=3):
    """Cluster parameter space using DBSCAN or KMeans."""
    X = StandardScaler().fit_transform(df[features])

    # Estimate eps if not provided (scales with log(N))
    if eps is None and method == "dbscan":
        eps = 0.3 + 1.2 * np.log10(len(df) / 100 + 1)

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

    df["cluster"] = labels
    print(f"Clustering done using {method.upper()} (eps={eps:.3f}) → "
          f"{len(set(labels))} clusters (including noise)")
    return df, model


def summarize_clusters(df, features):
    """Print descriptive statistics per cluster."""
    clusters = sorted(df["cluster"].unique())
    summaries = []
    print("\n=== Cluster Summary ===")
    for c in clusters:
        subset = df[df["cluster"] == c]
        label = "noise" if c == -1 else f"cluster {c}"
        print(f"\n{label} — {len(subset)} models")
        print(f"  χ²_min:  {subset['chi2_total'].min():.3f}")
        print(f"  χ²_mean: {subset['chi2_total'].mean():.3f}")
        for p in features:
            vals = subset[p]
            print(f"  {p:6s}: {vals.mean():.5g} ± {vals.std():.5g}")
        summaries.append({
            "cluster": c,
            "count": len(subset),
            "chi2_min": subset["chi2_total"].min(),
            "chi2_mean": subset["chi2_total"].mean(),
            **{f"{p}_mean": subset[p].mean() for p in features},
        })
    return pd.DataFrame(summaries)


def plot_clusters(df, features):
    """2D PCA projection colored by cluster."""
    if len(features) < 2:
        print("(Skipping PCA plot — not enough features.)")
        return

    X = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=2)
    pts = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        pts[:, 0], pts[:, 1],
        c=df["cluster"], cmap="tab10", s=40, alpha=0.8, edgecolors="k"
    )
    plt.colorbar(sc, label="Cluster ID")
    plt.title("PCA projection of parameter space (colored by cluster)")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()


def main(csv_file, chi_cut=None, method="dbscan"):
    df = load_and_filter(csv_file, chi_cut)
    features = detect_feature_set(df)
    if len(features) == 0:
        print("❌ No suitable numeric features found.")
        return

    df, model = cluster_data(df, features, method=method)
    summary_df = summarize_clusters(df, features)

    out_summary = Path(csv_file).with_suffix(".clusters.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"\n✅ Saved cluster summary → {out_summary}")

    try:
        plot_clusters(df, features)
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_basins.py file.csv [chi2_threshold] [--method=dbscan|kmeans]")
        sys.exit(1)

    csv_file = sys.argv[1]
    chi_cut = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    method = "dbscan"
    for arg in sys.argv:
        if arg.startswith("--method="):
            method = arg.split("=")[1]
    main(csv_file, chi_cut, method)
