#!/usr/bin/env python3
"""
terminal_shift_analysis.py — with slope cleanup, lap timing, and full time-series plotting
"""

# === CONFIG =================================================================
FOLDER_PATH = r"C:\Users\User\Desktop\New folder"

RIDERS = {
    "165Crank.fit": {"crank_m": 0.165, "height_m": 1.60, "mass_kg": 52},
}
# ============================================================================

import os, math, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fitparse import FitFile
from scipy.stats import ttest_ind

ROLL_WIN = 5
RAW   = ("Power", "Cadence", "Speed", "Heart_Rate")
DERIV = ("Torque_Nm", "Force_N", "Slope")
ALL   = list(RAW) + list(DERIV)

# ----------------------------------------------------------------------------
def compute_ride_and_lap_durations(df: pd.DataFrame):
    ride_duration_min = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds() / 60
    if "Lap_ID" in df.columns:
        lap_durations = df.groupby("Lap_ID")["Timestamp"].agg(["min", "max"])
        lap_durations["Duration_min"] = (lap_durations["max"] - lap_durations["min"]).dt.total_seconds() / 60
        return round(ride_duration_min, 2), lap_durations["Duration_min"].round(2)
    else:
        return round(ride_duration_min, 2), None

# ----------------------------------------------------------------------------
def plot_all_variables(df):
    plt.figure(figsize=(15, 7))
    vars_to_plot = ["Power", "Cadence", "Speed", "Heart_Rate", "Torque_Nm", "Force_N", "Slope"]
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:gray']
    for var, color in zip(vars_to_plot, colors):
        if var in df.columns:
            series = df[var].fillna(0)
            norm = (series - series.min()) / (series.max() - series.min())
            plt.plot(df["Time_s"], norm, label=var, color=color, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Value")
    plt.title("All Metrics Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
def parse_fit(path: str, crank_m: float, label: str) -> pd.DataFrame:
    fit = FitFile(path)
    data, last_fg, last_rg = {}, None, None

    lap_windows = []
    for lap in fit.get_messages("lap"):
        start = end = None
        for f in lap:
            if f.name == "start_time": start = f.value
            elif f.name == "timestamp": end = f.value
        if start and end:
            lap_windows.append((start, end))

    for rec in fit.get_messages("record"):
        ts, row = None, {}
        for f in rec:
            if f.name == "timestamp": ts = f.value
            elif f.name == "power": row["Power"] = f.value
            elif f.name == "cadence": row["Cadence"] = f.value
            elif f.name == "speed": row["Speed_mps"] = f.value
            elif f.name == "heart_rate": row["Heart_Rate"] = f.value
            elif f.name == "distance": row["Distance_m"] = f.value
            elif f.name == "altitude": row["Elevation"] = f.value
        if ts is None: continue
        r = data.setdefault(ts, {})
        r.update(row)
        r["Front_Gear"] = last_fg
        r["Rear_Gear"]  = last_rg
        lap_id = None
        for i, (start, end) in enumerate(lap_windows, 1):
            if start <= ts <= end:
                lap_id = i
                break
        r["Lap_ID"] = lap_id

    for ev in fit.get_messages("event"):
        evd = {f.name: f.value for f in ev}
        ts = evd.get("timestamp")
        if not ts: continue
        if evd.get("event") in ("rear_gear_change", "gear_change"):
            if "front_gear" in evd: last_fg = evd["front_gear"]
            if "rear_gear"  in evd: last_rg = evd["rear_gear"]
            data.setdefault(ts, {})["Front_Gear"] = last_fg
            data.setdefault(ts, {})["Rear_Gear"]  = last_rg

    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Timestamp"}, inplace=True)

    for col in ("Front_Gear", "Rear_Gear", "Distance_m"):
        df[col] = df[col].ffill().fillna(0)

    df["Time_s"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
    df["Speed"] = df["Speed_mps"].fillna(0) * 3.6
    df["Power"] = df["Power"].fillna(0)
    df["Cadence"] = df["Cadence"].fillna(0)
    df["Elevation"] = df["Elevation"].ffill().fillna(0)
    df["Heart_Rate"] = df["Heart_Rate"].ffill().fillna(0)

    cadence_rps = df["Cadence"] * 2 * math.pi / 60
    df["Torque_Nm"] = np.where(df["Cadence"] > 0, df["Power"] / cadence_rps, 0)

    df["Force_N"] = df["Torque_Nm"] / crank_m
    df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"].replace(0, np.nan)

    elev_smoothed = df["Elevation"].rolling(ROLL_WIN, center=True, min_periods=1).mean()
    dist_smoothed = df["Distance_m"].rolling(ROLL_WIN, center=True, min_periods=1).mean()
    delta_elev = elev_smoothed.diff()
    delta_dist = dist_smoothed.diff()
    safe_dist = delta_dist.where(delta_dist > 0.5, np.nan)
    df["Slope"] = (delta_elev / safe_dist * 100).clip(-30, 30).fillna(0)

    for c in ("Power", "Cadence", "Speed", "Heart_Rate",
              "Torque_Nm", "Force_N", "Gear_Ratio"):
        df[f"{c}_sm"] = df[c].rolling(ROLL_WIN, center=True, min_periods=1).mean()

    df["Shift"] = ((df["Front_Gear"].diff() != 0) |
                   (df["Rear_Gear"].diff() != 0)).astype(int)
    df["Shift_mag"] = df["Gear_Ratio"].diff().fillna(0)
    df["Shift_dir"] = np.where(df["Shift_mag"] > 0, "Harder",
                        np.where(df["Shift_mag"] < 0, "Easier", "No Shift"))

    df["Ride"] = label
    df["Crank_m"] = crank_m
    return df

# ----------------------------------------------------------------------------
def summary(df: pd.DataFrame):
    print("\n=== Mean / Min / Max per Ride ===")
    print(df.groupby("Ride")[ALL].agg(["min", "mean", "max"]).round(2))

# ----------------------------------------------------------------------------
def main():
    if not os.path.isdir(FOLDER_PATH):
        raise FileNotFoundError(FOLDER_PATH)

    all_df, shift_rows = [], []
    for fname in sorted(os.listdir(FOLDER_PATH)):
        if not fname.lower().endswith(".fit"):
            continue
        if fname not in RIDERS:
            warnings.warn(f"{fname} not in RIDERS dict – skipped")
            continue
        meta = RIDERS[fname]
        print(f"Parsing {fname} ...")
        df = parse_fit(os.path.join(FOLDER_PATH, fname),
                       meta["crank_m"],
                       os.path.splitext(fname)[0])
        all_df.append(df)
        shift_rows.append(df[df["Shift"] == 1])

        ride_dur, lap_durs = compute_ride_and_lap_durations(df)
        print(f"\n⏱️  Ride Duration: {ride_dur:.2f} minutes")
        if lap_durs is not None:
            print("📊 Lap Durations (minutes):")
            print(lap_durs.to_string())

    if not all_df:
        print("No files parsed successfully.")
        return

    master = pd.concat(all_df, ignore_index=True)
    shifts = pd.concat(shift_rows, ignore_index=True)

    summary(master)
    print(f"\nTotal shift events across rides: {len(shifts)}")

    crank_vals = shifts["Crank_m"].unique()
    if len(crank_vals) == 2:
        a = shifts[shifts["Crank_m"] == crank_vals[0]]["Torque_Nm_sm"].dropna()
        b = shifts[shifts["Crank_m"] == crank_vals[1]]["Torque_Nm_sm"].dropna()
        if len(a) > 1 and len(b) > 1:
            t, p = ttest_ind(a, b, equal_var=False)
            print(f"\nWelch t-test Torque@Shift : "
                  f"{crank_vals[0]*1000:.0f} vs {crank_vals[1]*1000:.0f} mm"
                  f"   t={t:.2f}   p={p:.3f}")

    print("\n=== Torque_Nm_sm at Shifts by Crank Length ===")
    print(shifts.groupby("Crank_m")["Torque_Nm_sm"].describe())

    if "Lap_ID" in master.columns:
        print("\n=== Shift Counts by Lap ===")
        shift_lap_summary = master[master["Shift"] == 1].groupby(["Ride", "Lap_ID"]).size()
        print(shift_lap_summary.rename("Shift_Count").to_string())

    torque_data = shifts[["Crank_m", "Torque_Nm_sm"]].dropna()
    if not torque_data.empty:
        plt.figure(figsize=(5, 4))
        torque_data.boxplot(column="Torque_Nm_sm", by="Crank_m", grid=False)
        plt.title("Torque at Shift vs Crank Length")
        plt.ylabel("Torque (Nm)")
        plt.suptitle("")
        plt.tight_layout()
    else:
        print("⚠️ No valid torque data to plot boxplot.")

    if not shifts.empty:
        plt.figure(figsize=(6, 4))
        for crank, grp in shifts.groupby("Crank_m"):
            plt.hist(grp["Cadence_sm"].dropna(), bins=20, alpha=0.5, density=True,
                     label=f"{crank*1000:.0f} mm")
        plt.title("Cadence at Shifts")
        plt.xlabel("Cadence (RPM)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

    print("\n📈 Plotting all normalized variables over time...")
    plot_all_variables(master)

if __name__ == "__main__":
    main()
