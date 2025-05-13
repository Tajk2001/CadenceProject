# CadenceProject 🚴‍♂️

This project analyzes cycling gear shifts and cadence data from `.fit` files using advanced time-series and machine learning techniques. It's built to explore **why and when cyclists shift gears**.

---

## 📂 Structure
CadenceProject/
├── scripts/ # Python scripts for data parsing, modeling, and visualization
├── data/ # Raw .fit files (ignored in Git)
├── output/ # Results, plots, model outputs
├── requirements.txt
└── README.md

---

## 🚀 Features

- Parses `.fit` files for cadence, power, HR, and gear data
- Detects gear shifts (easier, harder, no shift)
- Computes effective gradient, velocity-adjusted slope
- Performs:
  - Logistic and multinomial regression
  - SHAP interpretability
  - Random Forest and LSTM predictions
- Visualizes gear use and shifting behavior

---

## 📦 Setup

```bash
pip install -r requirements.txt
python scripts/GearShiftAnalysis.py
