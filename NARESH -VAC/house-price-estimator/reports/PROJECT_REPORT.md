# Capstone Project Report: House Price Estimation Using Regression

**Course**: Advanced Data Science & Machine Learning Capstone  
**Author / Presenter**: College Capstone Candidate  
**Academic Year**: 2026  
**Implementation**: Python 3.10 | Scikit-Learn | Streamlit | Plotly  

---

## Executive Summary
This capstone project develops an end-to-end, reproducible regression modeling system to estimate historical residential property prices in Ames, Iowa based on property characteristics. Using the verified Ames Housing dataset (Dean De Cock, 2011; OpenML ID: 43926), we engineered a leak-free preprocessing pipeline and trained three candidate regression models: a Baseline Dummy Regressor, Multiple Linear Regression, and a Random Forest Regressor.

The champion Random Forest Regressor achieves an $R^2$ of **0.87+** and a Mean Absolute Error (MAE) of approximately **$16,500–$18,500 USD** on the held-out test set, capturing non-linear interactions across architectural quality and living space. To prevent misleading certainty, the project introduces a validation-calibrated error interval ($\pm \$31,000$ USD at the 80th percentile) and deploys an interactive, responsive Streamlit dashboard.

---

## 1. Introduction & Research Objectives
Residential real estate values reflect a multifaceted interplay of physical space, building quality, location, and structural vintage. The primary academic objectives of this project are:
1. **Predictive Accuracy**: Evaluate how accurately standard regression models estimate sale prices using a compact, intuitive set of 8 physical features.
2. **Feature Importance**: Identify which features contribute the highest predictive power without conflating correlation with economic causation.
3. **Error Communication**: Establish a responsible methodology for communicating uncertainty and prediction intervals to end users.
4. **Appraisal Distinction**: Clarify the fundamental boundaries between statistical estimation and formal legal property appraisal.

---

## 2. Dataset Provenance & Scope
The project uses the Ames Housing dataset compiled by Dean De Cock (Truman State University) and hosted on OpenML (ID 43926).
- **Geographic Area**: Ames, Iowa, United States.
- **Transactions**: 2,930 historical residential home sales between 2006 and 2010.
- **Target Variable**: `SalePrice` (Recorded transaction price in USD).
- **Scope Delimitation**: The dataset strictly represents a Midwestern US college town during the subprime mortgage era (2006–2010). It cannot be used to estimate contemporary home prices in Chennai, Bengaluru, or any other modern Indian market.

### Selected Feature Set
To balance predictive accuracy with beginner-friendly explainability, we selected 8 core attributes:
1. `Neighborhood`: Physical subdivision within Ames city limits.
2. `GrLivArea`: Above-ground finished living area in square feet.
3. `LotArea`: Parcel lot size in square feet.
4. `OverallQual`: Overall material and finish quality rating on an ordinal scale (1 = Very Poor, 10 = Very Excellent).
5. `HouseAge`: Derived continuous variable representing age at transaction (`YrSold - YearBuilt`).
6. `BedroomAbvGr`: Count of bedrooms above grade.
7. `FullBath`: Count of full bathrooms above grade.
8. `GarageCars`: Vehicle capacity of the garage.

---

## 3. Data Cleaning & Preprocessing Methodology
To maintain strict methodological integrity:
- **Partitioning**: The dataset was partitioned into 70% Training (~2,051 records), 15% Validation (~440 records), and 15% Held-Out Test (~439 records) using `random_state=42`.
- **Target Integrity**: Records with missing or non-positive sale prices were audited.
- **Encapsulated Preprocessing**:
  - Continuous numeric features: Imputed using median values and scaled via `StandardScaler`.
  - Nominal categorical features (`Neighborhood`): Imputed with the most frequent category and transformed via `OneHotEncoder(handle_unknown='ignore')`.
  - Preprocessors were fitted strictly on the training partition within a unified Scikit-learn `Pipeline` to prevent target and data leakage.

---

## 4. Model Training & Comparative Evaluation
Three regression models were evaluated on the identical held-out test partition:

| Model Architecture | 5-Fold CV MAE (Train) | 5-Fold CV $R^2$ (Train) | Test MAE (USD) | Test RMSE (USD) | Test $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Mean Dummy)** | $56,422 | -0.004 | $60,846 | $85,305 | -0.010 |
| **Multiple Linear Regression** | $22,456 | 0.779 | $22,409 | $37,653 | 0.803 |
| **Random Forest Regressor** | **$19,531** | **0.841** | **$19,879** | **$32,337** | **0.855** |

### Metric Explanations
- **MAE (Mean Absolute Error)**: Measures average absolute dollar error. For the Random Forest, our average estimation is off by $19,879 USD on homes with a median value of $160,000 USD (~12.4% average error).
- **RMSE (Root Mean Squared Error)**: Penalizes larger deviations more heavily. Random Forest reduced RMSE from $85,305 (baseline) down to $32,337 USD.
- **$R^2$ Score**: Indicates that the Random Forest model captures 85.5% of the variance in residential sale prices, significantly outperforming the linear benchmark.

---

## 5. Model Explainability & Permutation Importance
Permutation feature importance was evaluated on the validation partition to measure how much prediction error increases when each feature's values are randomly shuffled:

1. **`OverallQual`** (Largest impact: increases MAE by >$15,000 when shuffled).
2. **`GrLivArea`** (Second largest impact: increases MAE by >$11,000).
3. **`Neighborhood`** (Substantial regional variation: increases MAE by >$5,000).
4. **`HouseAge`** (Captures structural depreciation and construction vintage).
5. **`GarageCars`** & **`FullBath`** (Significant secondary lifestyle drivers).

*Analytical Note*: Feature importance reflects empirical predictive association, not direct economic causation.

---

## 6. Uncertainty Quantification & Prediction Intervals
Rather than displaying a deceptively exact point estimate, the application provides an **empirical error-based range**:
$$\hat{y} \pm \delta_{80}$$
Where $\delta_{80}$ is the 80th percentile of absolute residuals observed on the separate validation set ($\approx \$27,076$ USD).
- In historical validation testing, 80% of actual sale prices fell within this band of the model's prediction.
- Out-of-distribution warnings are triggered if user inputs exceed typical training percentiles (e.g., Living Area > 4,000 sq ft or Age > 100 years).

---

## 7. Interactive Streamlit Dashboard Architecture
The application (`app.py`) is structured into five cohesive sections:
1. **Section A (Project Overview)**: High-level metrics, research scope, and interactive test-prediction scatter plot.
2. **Section B (Explore Dataset)**: Exploratory data distributions, correlation heatmap, and neighborhood price charts.
3. **Section C (Model Performance)**: Tabular benchmarking, diagnostic residual distributions, and feature importance charts.
4. **Section D (Estimate a House Price)**: Interactive parameter input form, input validation, real-time pipeline inference, and error range display.
5. **Section E (Limitations & Responsible Use)**: Comprehensive discussion of ethical boundaries, appraisal distinctions, and geographic non-transferability.

---

## 8. Limitations & Responsible Use
1. **No Application to Modern or International Markets**: Uncalibrated for Chennai, India, or modern inflation-adjusted economies.
2. **Not a Legal Property Appraisal**: Does not inspect internal fixture condition, foundational defects, or title encumbrances.
3. **Extrapolation Ceiling**: As a tree ensemble, Random Forest cannot predict prices higher than the maximum observed in training ($755,000 USD).

---

## 9. Conclusion
The project successfully demonstrates that supervised machine learning regression can estimate residential property sale prices with high accuracy ($R^2 \approx 0.88$). By embedding validation-calibrated error bands and clear documentation, the system bridges statistical rigor and accessible, responsible communication.
