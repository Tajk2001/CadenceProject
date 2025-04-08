# 🚴 Gear Shift Analysis & Prediction for Cyclists

This project analyzes **gear shifting behavior in cyclists** using FIT file data. It applies time series modeling, machine learning, and deep learning (including LSTM and attention mechanisms) to answer the research question:

> **"Why and when do cyclists shift gears?"**

---

## 🧠 Project Goals

- Parse and clean cycling FIT data
- Engineer features: gradients, rolling stats, lag/lead metrics
- Model gear shifting decisions:
  - Logistic, multinomial, and linear regression
  - Random Forest with SHAP explainability
  - LSTM and LSTM + Attention (sequence modeling)
- Visualize rider behavior, correlations, and feature influence
- Identify patterns and behavioral policies across rides

---

## 📁 Project Structure

- `GearShiftAnalysis.py`: Main preprocessing, feature engineering, and regression logic
- `LSTM + Attention`: Deep learning models for sequence-based shift prediction
- `clustering`: PCA + KMeans clustering for behavioral profiling
- `visualizations`: Heatmaps, boxplots, scatterplots, SHAP plots

---

## 🧪 Models & Interpretability

### ✅ Logistic & Linear Regression
- Understand how features like cadence, power, gradient affect shift likelihood/magnitude

### 🌲 Random Forest + SHAP
- Feature importance and local explanations (counterfactual testing)

### 🔁 LSTM / LSTM + Attention
- Sequence modeling of shift windows
- **Attention heatmaps** show temporal influence across time steps

---

## 📊 Visual Outputs

- 📈 Correlation heatmaps
- 🎯 SHAP summary plots
- 📦 Power/Cadence by shift type
- ⛰️ Gradient vs Cadence (by shift)
- 🎨 Attention heatmaps (LSTM + Attention)
- 🌐 PCA scatter plots (cluster analysis)

---

## 🔧 Requirements

- Python 3.9+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `fitparse`, `torch`, `sklearn`, `statsmodels`, `shap`

> ✅ Use `pip install -r requirements.txt` (optional)

---

## 📂 Input Data

Place `.fit` files inside:

```bash
C:/Users/USERNAME/Desktop/FitFiles/
