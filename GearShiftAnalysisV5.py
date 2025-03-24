"""
GearShiftAnalysisV5.py

Improvements over V4:

1. Modular Design:
   - Functions created for parsing FIT files and running models.
   - Easier to maintain, test, and reuse.

2. Selective Forward Fill:
   - Only forward-fills gear data (Front_Gear, Rear_Gear).
   - Prevents incorrect filling of sensor data (e.g., Power, Speed).

3. Safe Gear Ratio Calculation:
   - Avoids division by zero when computing Gear_Ratio.

4. Smarter Effective Gradient:
   - Uses non-zero mean speed for better velocity adjustment.

5. Cleaner Rate-of-Change Feature Calculation:
   - Efficient loop to compute Δ (change over time) for multiple metrics.

6. Flexible Feature Handling:
   - Prevents model failures from missing columns (e.g., Gear_Ratio_Δ_*).
   - Easy to customize feature sets for analysis.

7. Clear Progress Logging:
   - Prints file names and row counts for tracking during processing.
"""

import pandas as pd
import numpy as np
from fitparse import FitFile
from glob import glob
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- PARAMETERS ----------
folder_path = r"C:\Users\User\Desktop\FitFiles"  # Update this to your directory
time_intervals = [1, 3, 10, 30]


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

    # Extract gear events
    last_front_gear, last_rear_gear = None, None
    for event in fitfile.get_messages("event"):
        event_data = {field.name: field.value for field in event}
        timestamp = event_data.get("timestamp")
        if timestamp:
            front_gear = event_data.get("front_gear")
            rear_gear = event_data.get("rear_gear")
            if front_gear is not None:
                last_front_gear = front_gear
                data.setdefault(timestamp, {}).update({"Front_Gear": front_gear})
            if rear_gear is not None:
                last_rear_gear = rear_gear
                data.setdefault(timestamp, {}).update({"Rear_Gear": rear_gear})

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "Time"}, inplace=True)
    df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()
    df["Ride_ID"] = ride_id

    # Selective forward-fill: only for gear columns
    df[["Front_Gear", "Rear_Gear"]] = df[["Front_Gear", "Rear_Gear"]].ffill()

    # Compute gear ratio safely
    df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"].replace(0, np.nan)
    df["Gear_Ratio_Change"] = df["Gear_Ratio"].diff()
    df["Shift"] = df["Gear_Ratio_Change"].apply(lambda x: 1 if abs(x) > 0 else 0)

    df["Shift_Type"] = np.where(df["Gear_Ratio_Change"] > 0, "Harder",
                        np.where(df["Gear_Ratio_Change"] < 0, "Easier", "No Shift"))
    df["Shift_Magnitude"] = (df["Gear_Ratio_Change"] / df["Gear_Ratio"].shift(1)) * 100

    df["Gradient"] = df["Elevation"].diff() / df["Distance"].diff()
    mean_speed = df["Speed"].replace(0, np.nan).mean()
    df["Effective_Gradient"] = df["Gradient"] * (1 + (df["Speed"] / mean_speed))

    # Handle NaNs and Infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Rate of change features
    for interval in time_intervals:
        for var in ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient"]:
            df[f"{var}_Δ_{interval}s"] = df[var].diff(periods=interval)

    df.fillna(0, inplace=True)
    return df


# ---------- ANALYSIS FUNCTIONS ----------
def run_logistic_regression(df, features, target="Shift"):
    X = df[features].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df[target]
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

    # Ranked pairs
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

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()


# ---------- MAIN ----------
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

    # Clean again (just in case)
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.fillna(0, inplace=True)

    # Define features for modeling
    base_vars = ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient", "Gear_Ratio"]
    delta_vars = [f"{var}_Δ_{interval}s" for interval in time_intervals for var in base_vars if var != "Gear_Ratio"]
    expected_features = base_vars + delta_vars

    # Run Analyses
    logit_model = run_logistic_regression(full_df, expected_features)
    mnlogit_model = run_multinomial_logistic(full_df, expected_features)
    ols_model = run_linear_regression(full_df[full_df["Shift"] == 1], expected_features)

    # Optional Correlation Analysis
    correlation_analysis(full_df, expected_features)
