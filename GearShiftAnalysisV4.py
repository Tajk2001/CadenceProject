import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from fitparse import FitFile
import os
from glob import glob

# Define folder containing .FIT files
folder_path = r"C:\Users\User\Desktop\FitFiles"  # Change this to your directory

# Get all .FIT files in the folder
fit_files = glob(os.path.join(folder_path, "*.fit"))

# Initialize list to store DataFrames for multiple rides
all_rides = []

# Process each file separately
for file_idx, file_path in enumerate(fit_files):
    fitfile = FitFile(file_path)
    ride_id = f"Ride_{file_idx + 1}"  # Unique ID for each ride

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

    # Convert timestamps to relative time
    df["Time"] = (df["Time"] - df["Time"][0]).dt.total_seconds()

    # Add Ride ID Column
    df["Ride_ID"] = ride_id

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing Values Summary for {file_path}:")
    print(missing_values[missing_values > 0])

    # Fill missing values
    df.ffill(inplace=True)

    # Compute gear ratios
    df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"]
    df["Gear_Ratio_Change"] = df["Gear_Ratio"].diff()
    df["Shift"] = df["Gear_Ratio_Change"].apply(lambda x: 1 if abs(x) > 0 else 0)

    df["Shift_Type"] = np.where(df["Gear_Ratio_Change"] > 0, "Harder",
                                np.where(df["Gear_Ratio_Change"] < 0, "Easier", "No Shift"))
    df["Shift_Magnitude"] = (df["Gear_Ratio_Change"] / df["Gear_Ratio"].shift(1)) * 100

    # Compute Gradient
    df["Gradient"] = df["Elevation"].diff() / df["Distance"].diff()

    # Compute Effective Gradient (velocity-adjusted)
    df["Effective_Gradient"] = df["Gradient"] * (1 + (df["Speed"] / df["Speed"].mean()))

    # Replace NaNs and infinite values
    df["Gradient"] = df["Gradient"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["Effective_Gradient"] = df["Effective_Gradient"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Compute Rate of Change Variables at Multiple Time Intervals
    time_intervals = [1, 3, 10, 30]
    for interval in time_intervals:
        df[f"Power_Δ_{interval}s"] = df["Power"].diff(periods=interval)
        df[f"Cadence_Δ_{interval}s"] = df["Cadence"].diff(periods=interval)
        df[f"Speed_Δ_{interval}s"] = df["Speed"].diff(periods=interval)
        df[f"Elevation_Δ_{interval}s"] = df["Elevation"].diff(periods=interval)
        df[f"Heart_Rate_Δ_{interval}s"] = df["Heart_Rate"].diff(periods=interval)
        df[f"Gradient_Δ_{interval}s"] = df["Gradient"].diff(periods=interval)
        df[f"Effective_Gradient_Δ_{interval}s"] = df["Effective_Gradient"].diff(periods=interval)

    df.fillna(0, inplace=True)

    # Append processed ride data
    all_rides.append(df)

# Combine all rides into one dataset
full_df = pd.concat(all_rides, ignore_index=True)

# Ensure we only use numeric columns for correlation analysis
numeric_df = full_df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_df.corr()


# Generate ranked correlation pairs, excluding comparisons between like variables
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()


# Function to exclude comparisons between similar variables
def is_same_category(var1, var2):
    base_categories = ["Power", "Cadence", "Speed", "Elevation", "Heart_Rate", "Gradient", "Effective_Gradient"]

    # Prevent direct and rate-of-change comparisons within the same category
    for cat in base_categories:
        if cat in var1 and cat in var2:
            return True
    return False


# Filter out like-variable comparisons
filtered_corr_pairs = corr_pairs[
    ~corr_pairs.apply(lambda x: is_same_category(x["Variable 1"], x["Variable 2"]), axis=1)]

# Sort by absolute correlation strength
filtered_corr_pairs = filtered_corr_pairs.sort_values(by="Abs Correlation", ascending=False)

# Print top correlations
print("\nRanked Correlation Table (Excluding Like Variables):")
print(filtered_corr_pairs.head(20))

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Independent Variables")
plt.show()
