


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fitparse import FitFile

# Load the FIT file
file_path = r"C:\Users\User\Downloads\TestDataTwo.fit"
fitfile = FitFile(file_path)

# Dictionary to store all time-based data
data = {}

# Extract record data (Power, Cadence, Speed, HR, etc.)
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
            "Heart Rate": record_data.get("heart_rate", None),
            "Temperature": record_data.get("temperature", None),
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
            data.setdefault(timestamp, {}).update({"Front Gear": last_front_gear})

        if rear_gear is not None:
            last_rear_gear = rear_gear
            data.setdefault(timestamp, {}).update({"Rear Gear": last_rear_gear})

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame.from_dict(data, orient="index").reset_index()
df.rename(columns={"index": "Time"}, inplace=True)

# Convert timestamps to seconds relative to the first timestamp
df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()

# Fill missing values safely
df.loc[:, "Front Gear"] = df["Front Gear"].ffill()
df.loc[:, "Rear Gear"] = df["Rear Gear"].ffill()
df.loc[:, "Gear Ratio"] = (df["Front Gear"] / df["Rear Gear"]).replace([float("inf"), -float("inf")], None).ffill()

# Convert speed to km/h
df.loc[:, "Speed"] = df["Speed"] * 3.6  # Convert from m/s to km/h

# Ensure Distance starts from zero
df.loc[:, "Distance"] = (df["Distance"] - df["Distance"].min()).ffill()

# Fill other missing values
df = df.ffill()

# Detect gear shifts
df["Prev Gear Ratio"] = df["Gear Ratio"].shift(1)
df["Gear Ratio Change"] = df["Gear Ratio"] - df["Prev Gear Ratio"]
df["Shift Type"] = np.where(df["Gear Ratio Change"] > 0, "Harder",
                            np.where(df["Gear Ratio Change"] < 0, "Easier", "No Shift"))
df["Shift Magnitude (%)"] = (df["Gear Ratio Change"] / df["Prev Gear Ratio"]) * 100

# Filter only rows where a gear shift occurred
df_shifts = df[df["Shift Type"] != "No Shift"]

# Select relevant columns
df_shifts = df_shifts[["Time", "Distance", "Shift Type", "Shift Magnitude (%)", "Gear Ratio Change",
                        "Power", "Cadence", "Speed", "Elevation", "Heart Rate", "Temperature"]]

# ==============================
# 🚀 Logistic Regression Analysis
# ==============================

# Define dependent variable (Shift: 0 = No shift, 1 = Shift occurred)
df["Shift"] = df["Gear Ratio Change"].apply(lambda x: 1 if abs(x) > 0 else 0)

## Handle missing values and infinite values
df = df.replace([np.inf, -np.inf], np.nan)  # Convert inf values to NaN
df = df.dropna()  # Drop all rows with NaN values

# Define independent variables
features = ["Power", "Cadence", "Speed", "Elevation", "Heart Rate", "Temperature"]
X = df[features]
X = sm.add_constant(X)  # Add constant for intercept
y = df["Shift"]

# Fit logistic regression model
logit_model = sm.Logit(y, X).fit()

# Display model summary
print("\n=== Logistic Regression Model Summary ===")
print(logit_model.summary())

# ==============================
# 📊 Correlation Matrix Analysis
# ==============================

# Compute correlation matrix
corr_matrix = df[features].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Independent Variables")
plt.show()
