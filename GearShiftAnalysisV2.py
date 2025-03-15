"""
Gear Shift Analysis

This script analyzes cycling gear shifts using data from a .FIT file.
It applies:
- Binary logistic regression to determine factors influencing shifts.
- Multinomial logistic regression to distinguish between easier and harder shifts.
- Linear regression to quantify shift magnitude.
- A correlation matrix to examine relationships between variables.
- Multiple time intervals (1s, 3s, 10s, 30s) for rate-of-change calculations.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fitparse import FitFile

# Load FIT file
file_path = r"C:\Users\User\Downloads\TestDataTwo.fit"
fitfile = FitFile(file_path)

# Extract data
data = {}
for record in fitfile.get_messages("record"):
    record_data = {field.name: field.value for field in record}
    timestamp = record_data.get("timestamp")
    if timestamp:
        data.setdefault(timestamp, {}).update({
            "Power": record_data.get("power", None),
            "Cadence": record_data.get("cadence", None),
            "Speed": record_data.get("speed", None),
            "Distance": record_data.get("distance", None),
            "Elevation": record_data.get("altitude", None),
            "Heart_Rate": record_data.get("heart_rate", None),
        })

# Extract gear shift events
last_front_gear, last_rear_gear = None, None
for event in fitfile.get_messages("event"):
    event_data = {field.name: field.value for field in event}
    timestamp = event_data.get("timestamp")
    if timestamp:
        front_gear = event_data.get("front_gear")
        rear_gear = event_data.get("rear_gear")
        if front_gear is not None:
            last_front_gear = front_gear
            data.setdefault(timestamp, {}).update({"Front_Gear": last_front_gear})
        if rear_gear is not None:
            last_rear_gear = rear_gear
            data.setdefault(timestamp, {}).update({"Rear_Gear": last_rear_gear})

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient="index").reset_index()
df.rename(columns={"index": "Time"}, inplace=True)

# Convert timestamps
df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values Summary:")
print(missing_values[missing_values > 0])

# Check for duplicate timestamps
duplicates = df[df.duplicated(subset=["Time"], keep=False)]
if not duplicates.empty:
    print("\nWarning: Duplicate timestamps detected:")
    print(duplicates)

# Fill missing values
df.ffill(inplace=True)

# Ensure numeric columns do not contain infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Compute gear ratios
df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"]
df["Gear_Ratio_Change"] = df["Gear_Ratio"].diff()
df["Shift"] = df["Gear_Ratio_Change"].apply(lambda x: 1 if abs(x) > 0 else 0)

df["Shift_Type"] = np.where(df["Gear_Ratio_Change"] > 0, "Harder",
                            np.where(df["Gear_Ratio_Change"] < 0, "Easier", "No Shift"))
df["Shift_Magnitude"] = (df["Gear_Ratio_Change"] / df["Gear_Ratio"].shift(1)) * 100

# Compute Rate of Change Variables at Multiple Time Intervals
time_intervals = [1, 3, 10, 30]
for interval in time_intervals:
    df[f"Power_Δ_{interval}s"] = df["Power"].diff(periods=interval)
    df[f"Cadence_Δ_{interval}s"] = df["Cadence"].diff(periods=interval)
    df[f"Speed_Δ_{interval}s"] = df["Speed"].diff(periods=interval)
    df[f"Elevation_Δ_{interval}s"] = df["Elevation"].diff(periods=interval)
    df[f"Heart_Rate_Δ_{interval}s"] = df["Heart_Rate"].diff(periods=interval)

df.fillna(0, inplace=True)

# Ensure all expected features exist
expected_features = ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate"] + \
                    [f"{var}_Δ_{interval}s" for interval in time_intervals for var in ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate"]]

missing_features = [feature for feature in expected_features if feature not in df.columns]
if missing_features:
    print("\nWarning: Missing expected features:")
    print(missing_features)

# Model Fitting
X = df[expected_features]
X = sm.add_constant(X, has_constant='add')  # Ensure constant is added properly
y = df["Shift"]
logit_model = sm.Logit(y, X).fit(method="newton", maxiter=2000)

print("\nLogistic Regression Model (Shift vs No Shift)")
print(logit_model.summary())

# Multinomial Logistic Regression Model
df_shifts = df[df["Shift"] == 1].copy()
df_shifts["Shift_Type_Numeric"] = df_shifts["Shift_Type"].map({"Easier": 0, "Harder": 1})

X_shifts = df_shifts[X.columns.intersection(df_shifts.columns)]
y_shifts = df_shifts["Shift_Type_Numeric"]

mnlogit_model = sm.MNLogit(y_shifts, X_shifts).fit(method="newton", maxiter=2000)
print("\nMultinomial Logistic Regression Model (Shift Type)")
print(mnlogit_model.summary())

# Linear Regression Model for Shift Magnitude
lm_formula = "Shift_Magnitude ~ " + " + ".join(X_shifts.columns)
lm = ols(lm_formula, data=df_shifts).fit()

print("\nLinear Regression Model (Shift Magnitude)")
print(lm.summary())

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df[expected_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Independent Variables")
plt.show()