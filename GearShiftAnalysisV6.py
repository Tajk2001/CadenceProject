
# --- Imports ---
import pandas as pd
import numpy as np
from fitparse import FitFile
from glob import glob
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# --- Parameters ---
folder_path = r"C:\\Users\\User\\Desktop\\FitFiles"

# --- Lag/Lead Feature Generator ---
def add_lagged_lead_features(df):
    variables = ["Effective_Gradient", "Cadence", "Power", "Heart_Rate", "Speed"]
    for var in variables:
        df[f"{var}_lag_5s"] = df[var].shift(5)
        df[f"{var}_lead_5s"] = df[var].shift(-5)
    return df

# --- Rolling Average Features ---
def add_rolling_features(df, window=3):
    for var in ["Power", "Cadence", "Heart_Rate", "Speed", "Effective_Gradient", "Gradient"]:
        df[f"{var}_roll{window}s"] = df[var].rolling(window=window, center=True, min_periods=1).mean()
    return df

# --- FIT File Parser ---
def parse_fit_file(file_path, ride_id):
    fitfile = FitFile(file_path)
    data = {}

    for record in fitfile.get_messages("record"):
        record_data = {field.name: field.value for field in record}
        timestamp = record_data.get("timestamp")
        if timestamp:
            data.setdefault(timestamp, {}).update({
                "Power": record_data.get("power"),
                "Cadence": record_data.get("cadence"),
                "Speed": record_data.get("speed"),
                "Distance": record_data.get("distance"),
                "Elevation": record_data.get("altitude"),
                "Heart_Rate": record_data.get("heart_rate"),
            })

    for event in fitfile.get_messages("event"):
        event_data = {field.name: field.value for field in event}
        timestamp = event_data.get("timestamp")
        if timestamp:
            if event_data.get("front_gear") is not None:
                data.setdefault(timestamp, {})["Front_Gear"] = event_data["front_gear"]
            if event_data.get("rear_gear") is not None:
                data.setdefault(timestamp, {})["Rear_Gear"] = event_data["rear_gear"]

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "Time"}, inplace=True)
    df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()
    df["Ride_ID"] = ride_id

    df[["Front_Gear", "Rear_Gear"]] = df[["Front_Gear", "Rear_Gear"]].ffill()
    df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"].replace(0, np.nan)
    df["Gear_Ratio_Change"] = df["Gear_Ratio"].diff()
    df["Speed"] = df["Speed"] * 3.6  # Convert from m/s to km/h
    df["Shift"] = df["Gear_Ratio_Change"].apply(lambda x: 1 if abs(x) > 0 else 0)
    df["Shift_Type"] = np.where(df["Gear_Ratio_Change"] > 0, "Harder",
                                np.where(df["Gear_Ratio_Change"] < 0, "Easier", "No Shift"))
    df["Shift_Magnitude"] = (df["Gear_Ratio_Change"] / df["Gear_Ratio"].shift(1)) * 100
    df["Gradient"] = df["Elevation"].diff() / df["Distance"].diff()
    mean_speed = df["Speed"].replace(0, np.nan).mean()
    df["Effective_Gradient"] = df["Gradient"] * (1 + (df["Speed"] / mean_speed))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df = add_lagged_lead_features(df)
    df = add_rolling_features(df, window=3)
    df.fillna(0, inplace=True)
    return df

# --- Regression Models ---
def run_logistic_regression(df, features, target="Shift"):
    X = sm.add_constant(df[features])
    y = df[target]
    model = sm.Logit(y, X).fit(method="newton", maxiter=2000)
    print("\nLogistic Regression Model (Shift vs No Shift)")
    print(model.summary())
    return model

def run_multinomial_logistic(df, features):
    df = df[df["Shift"] == 1].copy()
    df["Shift_Type_Numeric"] = df["Shift_Type"].map({"Easier": 0, "Harder": 1})
    X = sm.add_constant(df[features])
    y = df["Shift_Type_Numeric"]
    model = sm.MNLogit(y, X).fit(method="newton", maxiter=2000)
    print("\nMultinomial Logistic Regression Model (Shift Type)")
    print(model.summary())
    return model

def run_linear_regression(df, features, target="Shift_Magnitude"):
    formula = f"{target} ~ " + " + ".join(features)
    model = ols(formula, data=df).fit()
    print("\nLinear Regression Model (Shift Magnitude)")
    print(model.summary())
    return model

def correlation_analysis(df, features):
    # Short name map for display
    short_name_map = {
        "Effective_Gradient_roll3s": "EG_roll",
        "Effective_Gradient_lag_5s": "EG_lag",
        "Effective_Gradient_lead_5s": "EG_lead",
        "Gradient_roll3s": "Grad_roll",
        "Gradient": "Grad",
        "Cadence": "Cad",
        "Cadence_lag_5s": "Cad_lag",
        "Cadence_lead_5s": "Cad_lead",
        "Power": "Pwr",
        "Power_lag_5s": "Pwr_lag",
        "Power_lead_5s": "Pwr_lead",
        "Heart_Rate": "HR",
        "Heart_Rate_lag_5s": "HR_lag",
        "Speed": "Spd",
        "Speed_lag_5s": "Spd_lag",
        "Speed_lead_5s": "Spd_lead",
        "Elevation": "Elev",
    }

    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

    # Find top correlation pairs
    top_corr = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    top_corr.columns = ["Variable 1", "Variable 2", "Correlation"]
    top_corr["Abs Correlation"] = top_corr["Correlation"].abs()

    # Shorten variable names for printout
    top_corr["Variable 1"] = top_corr["Variable 1"].map(lambda x: short_name_map.get(x, x))
    top_corr["Variable 2"] = top_corr["Variable 2"].map(lambda x: short_name_map.get(x, x))

    top_corr = top_corr.sort_values("Abs Correlation", ascending=False)
    print("\nTop Correlation Pairs:")
    print(top_corr.head(20).to_string(index=False))


    # --- Additional Visualizations ---

    # Use unscaled data for visualization
    unscaled_df = df.copy()
    unscaled_df[reduced_features] = scaler.inverse_transform(df[reduced_features])

    # Power by Shift Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=unscaled_df, x="Shift_Type", y="Power", hue="Shift_Type", palette="pastel", legend=False)
    plt.title("Power Distribution by Shift Type", fontsize=14)
    plt.xlabel("Shift Type", fontsize=12)
    plt.ylabel("Power (Watts)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Cadence by Shift Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=unscaled_df, x="Shift_Type", y="Cadence", hue="Shift_Type", palette="pastel", legend=False)
    plt.title("Cadence Distribution by Shift Type", fontsize=14)
    plt.xlabel("Shift Type", fontsize=12)
    plt.ylabel("Cadence (RPM)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gradient_roll3s vs Cadence Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(unscaled_df, x="Gradient_roll3s", y="Cadence", hue="Shift_Type", alpha=0.3, palette="Set2")
    plt.title("Gradient_roll3s vs Cadence by Shift Type", fontsize=14)
    plt.xlabel("Gradient_roll3s", fontsize=12)
    plt.ylabel("Cadence (RPM)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


    print_ride_summary_stats(all_rides)
    full_df = pd.concat(all_rides, ignore_index=True)
    print(f"\nAll rides processed. Total rows: {len(full_df)}")

def run_double_clustering(df, features, n_components=2, n_clusters=3):
    print("\n🔍 Running Clustering on Original and PCA Features...")

    # Drop NaNs
    df_clean = df[features].dropna().copy()

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # --- KMeans on Original Features ---
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean["Cluster_Original"] = kmeans_orig.fit_predict(scaled_data)

    # --- PCA ---
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled_data)
    explained = pca.explained_variance_ratio_
    print(f"\n PCA Explained Variance Ratio: {explained}")

    # --- KMeans on PCA ---
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean["Cluster_PCA"] = kmeans_pca.fit_predict(pcs)

    # --- Short name mapping ---
    short_name_map = {
        "Effective_Gradient_roll3s": "EG_roll",
        "Effective_Gradient_lag_5s": "EG_lag",
        "Effective_Gradient_lead_5s": "EG_lead",
        "Gradient_roll3s": "Grad_roll",
        "Gradient": "Grad",
        "Cadence": "Cad",
        "Cadence_lag_5s": "Cad_lag",
        "Cadence_lead_5s": "Cad_lead",
        "Power": "Pwr",
        "Power_lag_5s": "Pwr_lag",
        "Power_lead_5s": "Pwr_lead",
        "Heart_Rate_lag_5s": "HR_lag",
        "Speed": "Spd",
        "Speed_lag_5s": "Spd_lag",
        "Speed_lead_5s": "Spd_lead",
        "Elevation": "Elev",
    }

    # --- Summary Stats: Original Clustering ---
    orig_cluster_means = df_clean.groupby("Cluster_Original")[features].mean().round(2)
    orig_cluster_means.rename(columns=short_name_map, inplace=True)
    print("\n Cluster Means (Original Features):")
    print(orig_cluster_means.to_string())

    # --- Summary Stats: PCA-Based Clustering ---
    pca_cluster_means = df_clean.groupby("Cluster_PCA")[features].mean().round(2)
    pca_cluster_means.rename(columns=short_name_map, inplace=True)
    print("\ Cluster Means (PCA-based Clustering):")
    print(pca_cluster_means.to_string())

    # --- PCA Scatter Plot ---
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df_clean["Cluster_PCA"].values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", alpha=0.7)
    plt.title("PCA-Based Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()




# --- Main ---
if __name__ == "__main__":
    fit_files = glob(os.path.join(folder_path, "*.fit"))
    all_rides = []

    for idx, file_path in enumerate(fit_files):
        ride_id = f"Ride_{idx + 1}"
        df = parse_fit_file(file_path, ride_id)
        print(f"Processed: {ride_id} | {file_path}")
        all_rides.append(df)

    full_df = pd.concat(all_rides, ignore_index=True)
    print(f"\nAll rides processed. Total rows: {len(full_df)}")

    full_df.dropna(subset=["Power", "Cadence", "Speed", "Heart_Rate", "Elevation", "Distance"], inplace=True)
    full_df = full_df[(full_df["Power"] >= 0) & (full_df["Power"] <= 2000)]
    full_df = full_df[(full_df["Heart_Rate"] >= 30) & (full_df["Heart_Rate"] <= 220)]
    full_df = full_df[(full_df["Cadence"] >= 10) & (full_df["Cadence"] <= 180)]
    full_df = full_df[(full_df["Speed"] >= 0) & (full_df["Speed"] <= 120)]

    excluded_keywords = ["Gear_Ratio"]
    expected_features = [
        col for col in full_df.columns
        if (("_lag" in col or "_lead" in col or "_roll3s" in col or col in [
            "Power", "Cadence", "Heart_Rate", "Speed", "Gradient", "Effective_Gradient", "Elevation"
        ]) and not any(kw in col for kw in excluded_keywords))
    ]

    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(subset=expected_features, inplace=True)


    def print_ride_summary_stats(all_rides):
        summary_rows = []
        for ride_df in all_rides:
            ride_id = ride_df["Ride_ID"].iloc[0]
            row = {
                "Ride_ID": ride_id,
                "Power_mean": ride_df["Power"].mean(),
                "Power_min": ride_df["Power"].min(),
                "Power_max": ride_df["Power"].max(),
                "Cadence_mean": ride_df["Cadence"].mean(),
                "Cadence_min": ride_df["Cadence"].min(),
                "Cadence_max": ride_df["Cadence"].max(),
                "Heart_Rate_mean": ride_df["Heart_Rate"].mean(),
                "Heart_Rate_min": ride_df["Heart_Rate"].min(),
                "Heart_Rate_max": ride_df["Heart_Rate"].max(),
                "Speed_mean": ride_df["Speed"].mean(),
                "Speed_min": ride_df["Speed"].min(),
                "Speed_max": ride_df["Speed"].max(),
                "Elevation_mean": ride_df["Elevation"].mean(),
                "Elevation_min": ride_df["Elevation"].min(),
                "Elevation_max": ride_df["Elevation"].max(),
            }
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        print("\n Ride Summary Statistics (Per Ride):")
        print(summary_df.to_string(index=False))


    def drop_high_vif(df, feature_list, threshold=200.0):
        X = df[feature_list].copy()
        while True:
            vif_data = pd.DataFrame()
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data["Feature"] = X.columns
            max_vif = vif_data["VIF"].max()
            if max_vif > threshold:
                drop_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
                print(f"Dropping {drop_feature} due to high VIF: {max_vif:.2f}")
                X.drop(columns=[drop_feature], inplace=True)
            else:
                break
        return list(X.columns)

    reduced_features = drop_high_vif(full_df, expected_features)
    scaler = StandardScaler()
    full_df[reduced_features] = scaler.fit_transform(full_df[reduced_features])

    logit_model = run_logistic_regression(full_df, reduced_features)
    mnlogit_model = run_multinomial_logistic(full_df, reduced_features)
    ols_model = run_linear_regression(full_df[full_df["Shift"] == 1], reduced_features)
    correlation_analysis(full_df, reduced_features)
    run_double_clustering(full_df[full_df["Shift"] == 1], reduced_features)


