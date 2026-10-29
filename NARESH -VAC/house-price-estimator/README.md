# 🏡 House Price Estimation Using Regression

An end-to-end, reproducible college capstone project predicting historical residential property transaction prices using supervised regression machine learning.

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview
This project builds, benchmarks, and deploys machine learning regression models to estimate historical house sale prices based on core physical property attributes:
- **Champion Model**: Random Forest Regressor ($R^2 \approx 0.88$, Test MAE $\approx \$16,900$ USD).
- **Benchmark Models**: Baseline Dummy Regressor (Mean) and Multiple Linear Regression.
- **Uncertainty Communication**: Empirical error-based estimation range ($\pm \$31,000$ USD based on validation residuals).
- **Interactive UI**: A 5-section Streamlit analytical dashboard with real-time prediction and diagnostic visualizations.
- **Dataset**: Ames Housing Dataset (Dean De Cock, 2011; OpenML ID: 43926).

---

## 📁 Project Structure

```text
house-price-estimator/
├── app.py                            # Streamlit interactive application & dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and quickstart
├── .gitignore                        # Git exclusion rules
├── src/
│   └── train_model.py                # Data loading, preprocessing, model training & evaluation
├── notebooks/
│   └── house_price_analysis.ipynb    # Exploratory data analysis (EDA) notebook
├── data/
│   ├── README.md                     # Dataset documentation and provenance
│   ├── ames_housing_raw.csv          # Cached raw dataset from OpenML
│   └── ames_housing_clean.csv        # Cleaned dataset with core features
├── artifacts/
│   ├── .gitkeep
│   ├── model_pipeline.joblib         # Serialized Scikit-learn Random Forest pipeline
│   ├── linear_pipeline.joblib        # Serialized Linear Regression pipeline
│   ├── dummy_pipeline.joblib         # Serialized Dummy Regressor pipeline
│   ├── metrics_summary.json          # Actual computed CV, validation, and test metrics
│   ├── feature_metadata.json         # Feature distributions and input validation bounds
│   └── test_predictions.csv          # Held-out test set actual vs predicted values
└── reports/
    ├── DATASET_CARD.md               # Dataset provenance, license, scope, and limitations
    ├── PROJECT_REPORT.md             # Full capstone report with methodology and findings
    ├── presentation_outline.md       # Six-slide presentation outline for viva/defense
    └── figures/                      # High-resolution generated charts
        ├── eda_price_distribution.png
        ├── eda_area_vs_price.png
        ├── eda_quality_vs_price.png
        ├── eda_correlation_heatmap.png
        ├── actual_vs_predicted.png
        ├── residual_distribution.png
        ├── residuals_vs_predicted.png
        ├── model_performance_comparison.png
        └── feature_importance.png
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup (Windows PowerShell)

```powershell
# Navigate into the project folder
cd house-price-estimator

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install required packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(On macOS/Linux, activate via `source .venv/bin/activate`)*

---

### 2. Train Models and Generate Artifacts

Run the automated training and evaluation pipeline:
```powershell
python src/train_model.py
```
This script will:
1. Fetch the Ames Housing dataset from OpenML (or load from local cache).
2. Clean and engineer features (`HouseAge = YrSold - YearBuilt`).
3. Partition data into 70% Train, 15% Validation, and 15% Held-Out Test sets.
4. Execute 5-fold cross-validation across Baseline, Linear, and Random Forest models.
5. Calibrate the empirical error interval on the validation partition.
6. Evaluate all candidates on the held-out test set and compute MAE, RMSE, and $R^2$.
7. Export trained pipelines to `artifacts/` and visualization figures to `reports/figures/`.

---

### 3. Launch the Streamlit Interactive Dashboard

```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard includes 5 navigable sections:
- **Section A: Project Overview**: Summary cards, research questions, and actual vs. predicted plot.
- **Section B: Explore Dataset**: Interactive distributions, scatter plots, and neighborhood pricing tables.
- **Section C: Model Performance**: Cross-validation vs. test benchmarks, residual histograms, and permutation feature importances.
- **Section D: Estimate a House Price**: Interactive property form with real-time inference and empirical error bands.
- **Section E: Limitations & Responsible Use**: Clarification of geographic boundaries and why statistical regression is not a property appraisal.

---

### 4. Running the Exploratory Analysis Notebook

To run or present the Jupyter notebook:
```powershell
jupyter notebook notebooks/house_price_analysis.ipynb
```

---

## 🌐 Offline & Dataset Troubleshooting
- If OpenML cannot be reached due to an institutional firewall or network error, `src/train_model.py` automatically checks for a cached `data/ames_housing_raw.csv`.
- If downloading from OpenML fails, verify that an active internet connection is present, or download the Ames Housing CSV directly from OpenML (Data ID 43926) and place it in the `data/` folder as `ames_housing_raw.csv`.

---

## ⚖️ Responsible Use & Ethical Caveat
- **Geographic Boundary**: Trained strictly on residential sales in Ames, Iowa (2006–2010). **Do not use for estimating property prices in Chennai, other parts of India, or outside historical Ames.**
- **Not a Certified Valuation**: This tool provides educational regression estimates based on statistical conditional means; it cannot inspect physical structural integrity, assess title legalities, or replace a licensed real estate appraiser.
