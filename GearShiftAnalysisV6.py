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

# --- Parameters ---
folder_path = r"C:\\Users\\tajkr\\Desktop\\FitFiles"

# --- Lag/Lead Feature Generator ---
def add_lagged_lead_features(df):
    variables = ["Effective_Gradient", "Cadence", "Power", "Heart_Rate", "Speed", "Gear_Ratio"]
    for var in variables:
        df[f"{var}_lag_5s"] = df[var].shift(5)
        df[f"{var}_lead_5s"] = df[var].shift(-5)
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

    df = add_lagged_lead_features(df)
    df.fillna(0, inplace=True)
    return df

# --- Regression Models ---
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

# --- Correlation Analysis ---
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

# --- Policy Modeling ---
def create_state_action_dataset(df):
    df = df.copy()
    df["Action"] = df["Shift_Type"].map({"No Shift": 0, "Easier": 1, "Harder": 2})
    state_vars = [
        "Power", "Cadence", "Speed", "Gradient", "Effective_Gradient",
        "Heart_Rate", "Gear_Ratio",
        "Cadence_lag_5s", "Cadence_lead_5s",
        "Power_lag_5s", "Power_lead_5s"
    ]
    return df.dropna(subset=state_vars + ["Action"])[state_vars + ["Action"]]

def train_policy_model(state_action_df):
    X = state_action_df.drop("Action", axis=1)
    y = state_action_df["Action"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    print("\n🚴 Policy Model Evaluation:")
    print(classification_report(y_test, model.predict(X_test), target_names=["No Shift", "Easier", "Harder"]))
    return model, X

def interpret_policy_model(model, feature_names):
    coef_df = pd.DataFrame(model.coef_, columns=feature_names, index=["Easier vs No Shift", "Harder vs No Shift"])
    print("\n🧠 Policy Model Coefficients:")
    print(coef_df.T.sort_values(by="Harder vs No Shift", ascending=False))

def detect_policy_deviations(state_action_df, model):
    df = state_action_df.copy()
    X = df.drop("Action", axis=1)
    df["Predicted_Action"] = model.predict(X)
    df["Deviation"] = df["Action"] != df["Predicted_Action"]
    return df[df["Deviation"] == True]

# --- Shift Prediction Modeling ---
def build_shift_prediction_dataset(df, window=5):
    rows = []
    shift_indices = df.index[df["Shift"] == 1]

    for idx in shift_indices:
        if idx - window < 0 or idx + window >= len(df):
            continue
        row = {"Shift": 1, "Shift_Type": df.loc[idx, "Shift_Type"]}
        for offset in range(-window, 0):
            for var in ["Power", "Cadence", "Speed", "Gradient", "Effective_Gradient", "Heart_Rate"]:
                row[f"{var}_t{offset}"] = df.loc[idx + offset, var]
        for offset in range(1, window + 1):
            for var in ["Cadence", "Power"]:
                row[f"{var}_t+{offset}"] = df.loc[idx + offset, var]
        rows.append(row)

    return pd.DataFrame(rows)

def train_shift_classifier(shift_df):
    X = shift_df.drop(columns=["Shift", "Shift_Type"])
    y = shift_df["Shift"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("\n⏱ Shift Classifier Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))
    return model

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

    base_vars = ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient", "Gear_Ratio"]
    lag_lead_vars = [
        "Effective_Gradient_lag_5s", "Effective_Gradient_lead_5s",
        "Cadence_lag_5s", "Cadence_lead_5s",
        "Power_lag_5s", "Power_lead_5s",
        "Heart_Rate_lag_5s", "Heart_Rate_lead_5s",
        "Speed_lag_5s", "Speed_lead_5s",
        "Gear_Ratio_lag_5s", "Gear_Ratio_lead_5s"
    ]
    expected_features = base_vars + lag_lead_vars

    logit_model = run_logistic_regression(full_df, expected_features)
    mnlogit_model = run_multinomial_logistic(full_df, expected_features)
    ols_model = run_linear_regression(full_df[full_df["Shift"] == 1], expected_features)
    correlation_analysis(full_df, expected_features)

    state_action_df = create_state_action_dataset(full_df)
    policy_model, X_features = train_policy_model(state_action_df)
    interpret_policy_model(policy_model, X_features.columns)
    deviations_df = detect_policy_deviations(state_action_df, policy_model)
    print(f"\n🚨 Found {len(deviations_df)} policy deviations out of {len(state_action_df)} actions")

    shift_prediction_df = build_shift_prediction_dataset(full_df, window=5)
    if not shift_prediction_df.empty:
        shift_classifier = train_shift_classifier(shift_prediction_df)
    else:
        print("No valid shift-centered windows found for prediction dataset.")
