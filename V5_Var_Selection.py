import pandas as pd
import numpy as np
from fitparse import FitFile
from glob import glob
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")  # Suppress convergence warnings

# ---------- PARAMETERS ----------
folder_path = r"C:\Users\User\Desktop\FitFiles"  # Update this to your directory
time_intervals = [1, 3, 10, 30]
remove_multicollinearity = True  # Toggle this to filter multicollinear features


# ---------- FUNCTION: PARSE SINGLE FIT FILE ----------
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

    last_front_gear, last_rear_gear = None, None
    for event in fitfile.get_messages("event"):
        event_data = {field.name: field.value for field in event}
        timestamp = event_data.get("timestamp")
        if timestamp:
            if event_data.get("front_gear") is not None:
                last_front_gear = event_data["front_gear"]
                data.setdefault(timestamp, {}).update({"Front_Gear": last_front_gear})
            if event_data.get("rear_gear") is not None:
                last_rear_gear = event_data["rear_gear"]
                data.setdefault(timestamp, {}).update({"Rear_Gear": last_rear_gear})

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "Time"}, inplace=True)
    df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()
    df["Ride_ID"] = ride_id

    df[["Front_Gear", "Rear_Gear"]] = df[["Front_Gear", "Rear_Gear"]].ffill()
    df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"].replace(0, np.nan)
    df["Gear_Ratio_Change"] = df["Gear_Ratio"].diff()
    df["Shift"] = df["Gear_Ratio_Change"].apply(lambda x: 1 if abs(x) > 0 else 0)
    df["Shift_Type"] = np.where(df["Gear_Ratio_Change"] > 0, "Harder",
                         np.where(df["Gear_Ratio_Change"] < 0, "Easier", "No Shift"))
    df["Shift_Magnitude"] = (df["Gear_Ratio_Change"] / df["Gear_Ratio"].shift(1)) * 100

    df["Gradient"] = df["Elevation"].diff() / df["Distance"].diff()
    mean_speed = df["Speed"].replace(0, np.nan).mean()
    df["Effective_Gradient"] = df["Gradient"] * (1 + (df["Speed"] / mean_speed))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    for interval in time_intervals:
        for var in ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient"]:
            df[f"{var}_Δ_{interval}s"] = df[var].diff(periods=interval)

    df.fillna(0, inplace=True)
    return df


# ---------- FEATURE SELECTION ----------
def select_features(df, candidate_vars):
    print("\nAvailable variables for modeling:")
    for i, var in enumerate(candidate_vars):
        print(f"{i + 1}: {var}")

    selected_indices = input("\nEnter the numbers of the variables you want to include (comma-separated): ")
    selected_indices = [int(i.strip()) - 1 for i in selected_indices.split(",")]
    selected_vars = [candidate_vars[i] for i in selected_indices]

    print(f"\n✅ Selected Features: {selected_vars}\n")
    return selected_vars


# ---------- VIF FILTER ----------
def remove_high_vif_features(X, threshold=10.0):
    while True:
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        if vif.max() > threshold:
            drop_feature = vif.idxmax()
            print(f"Removing multicollinear feature: {drop_feature} (VIF={vif.max():.2f})")
            X = X.drop(columns=[drop_feature])
        else:
            break
    return X


# ---------- ANALYSIS FUNCTIONS ----------
def run_logistic_regression(df, features, target="Shift"):
    X = df[features].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df[target]

    if remove_multicollinearity:
        X = remove_high_vif_features(X)

    model = sm.Logit(y, X).fit(method="newton", maxiter=2000)
    print("\nLogistic Regression Model (Shift vs No Shift)")
    print(model.summary())
    return model


def run_multinomial_logistic(df, features):
    df = df[df["Shift"] == 1].copy()
    df["Shift_Type_Numeric"] = df["Shift_Type"].map({"Easier": 0, "Harder": 1})

    X = df[features].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df["Shift_Type_Numeric"]

    if remove_multicollinearity:
        X = remove_high_vif_features(X)

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
    corr_matrix = df[features].corr()
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    corr_pairs = corr_pairs.sort_values(by="Abs Correlation", ascending=False)

    print("\nTop Correlation Pairs:")
    print(corr_pairs.head(20))

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()


# ---------- MAIN ----------
if __name__ == "__main__":
    fit_files = glob(os.path.join(folder_path, "*.fit"))
    if not fit_files:
        print(f"No .fit files found in: {folder_path}")
        exit()

    all_rides = []
    for idx, file_path in enumerate(fit_files):
        ride_id = f"Ride_{idx + 1}"
        try:
            df = parse_fit_file(file_path, ride_id)
            if not df.empty:
                all_rides.append(df)
                print(f"Processed: {ride_id} | {file_path}")
            else:
                print(f"Skipped empty DataFrame: {ride_id}")
        except Exception as e:
            print(f"Error processing {ride_id} | {file_path}: {e}")

    if not all_rides:
        print("❌ No usable rides parsed.")
        exit()

    full_df = pd.concat(all_rides, ignore_index=True)
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.fillna(0, inplace=True)

    base_vars = ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient", "Gear_Ratio"]
    delta_vars = [f"{var}_Δ_{interval}s" for interval in time_intervals for var in base_vars if var != "Gear_Ratio"]
    all_vars = base_vars + delta_vars

    selected_features = select_features(full_df, all_vars)

    # Run Analyses
    logit_model = run_logistic_regression(full_df, selected_features)
    mnlogit_model = run_multinomial_logistic(full_df, selected_features)
    ols_model = run_linear_regression(full_df[full_df["Shift"] == 1], selected_features)

    correlation_analysis(full_df, selected_features)
