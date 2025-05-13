
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
folder_path = r"C:\\Users\\tajkr\\Desktop\\FitFiles"

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
    print("\\ Cluster Means (PCA-based Clustering):")
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

# --- NEW IMPORTS FOR FULL PIPELINE ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# --- TIME-WINDOWED SEQUENCE DATASET CREATION ---
def create_shift_windows(df, window_size=10, step=1, features=None):
    shift_windows = []
    labels = []
    shift_types = []

    for i in range(window_size, len(df) - window_size, step):
        window = df.iloc[i - window_size:i]
        label = df.iloc[i]["Shift"]
        shift_type = df.iloc[i]["Shift_Type"]

        if features:
            window_data = window[features].values.flatten()
            col_names = [f"{f}_{j}" for j in range(window_size) for f in features]
        else:
            selected = ["Cadence", "Power", "Effective_Gradient", "Speed", "Heart_Rate"]
            window_data = window[selected].values.flatten()
            col_names = [f"{f}_{j}" for j in range(window_size) for f in selected]

        shift_windows.append(window_data)
        labels.append(label)
        shift_types.append(shift_type)

    X = pd.DataFrame(shift_windows, columns=col_names)
    y = pd.Series(labels)
    y_type = pd.Series(shift_types)
    return X, y, y_type

# --- TRAIN RANDOM FOREST AND INTERPRET WITH SHAP ---
def train_rf_with_shap(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y, model.predict(X)))

    sample_idx = np.random.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("\nSHAP Summary Plot (Top features):")
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_sample, plot_type="bar", max_display=10)
        shap.summary_plot(shap_values[1], X_sample)
    else:
        shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=10)
        shap.summary_plot(shap_values, X_sample)

    return model, shap_values

# --- SIMPLE COUNTERFACTUAL CHECKER ---
def test_counterfactual(model, X, y, feature_names, test_feature_idx, delta):
    original_probs = model.predict_proba(X)
    X_cf = X.copy()
    X_cf.iloc[:, test_feature_idx] += delta
    counterfactual_probs = model.predict_proba(X_cf)
    delta_probs = counterfactual_probs - original_probs

    avg_change = delta_probs.mean(axis=0)
    print(f"\nCounterfactual impact of modifying feature {feature_names[test_feature_idx]} by {delta}:")
    print(f"Avg class probability change: {avg_change}")

# --- BASIC LSTM WITHOUT ATTENTION ---
class GearShiftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32).view(-1, 10, X.shape[1] // 10)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_lstm_model(X, y, input_size, epochs=15, batch_size=64):
    dataset = GearShiftDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = LSTMClassifier(input_size=input_size)
    weights = torch.tensor([1.0, 8.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.numpy())

    print("\nLSTM Classification Report:")
    print(classification_report(all_targets, all_preds))
    return model

# --- ATTENTION-BASED LSTM WITH HEATMAP ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = self.attn(lstm_output).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2, dropout=0.3):
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        out = self.fc(context)
        return out, weights

def train_lstm_attention_model(X, y, input_size, window_size=10, epochs=15, batch_size=64):
    dataset = GearShiftDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = LSTMAttentionClassifier(input_size=input_size)
    weights = torch.tensor([1.0, 8.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.4f}")

    model.eval()
    all_preds, all_targets, all_weights = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, attn_weights = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
            all_weights.extend(attn_weights.numpy())

    print("\nLSTM + Attention Classification Report:")
    print(classification_report(all_targets, all_preds))

    print("\nGenerating attention heatmap for first 10 samples...")
    heatmap_data = np.array(all_weights[:10])
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="viridis", cbar=True, xticklabels=[f"t-{9-i}" for i in range(10)])
    plt.xlabel("Time Step")
    plt.ylabel("Sample Index")
    plt.title("Attention Weights (First 10 Samples)")
    plt.tight_layout()
    plt.show()

    return model

print("\n--- Running Time-Windowed Shift Prediction Model ---")
X, y, y_type = create_shift_windows(full_df, window_size=10, features=reduced_features)
print("\nShift label distribution:", y.value_counts())

model_rf, shap_values = train_rf_with_shap(X, y)
test_counterfactual(model_rf, X, y, X.columns.tolist(), test_feature_idx=5, delta=-0.2)

print("\n--- Running LSTM Shift Prediction Model ---")
lstm_model = train_lstm_model(X, y, input_size=X.shape[1] // 10)

print("\n--- Running LSTM + Attention Shift Prediction Model ---")
lstm_att_model = train_lstm_attention_model(X, y, input_size=X.shape[1] // 10)
