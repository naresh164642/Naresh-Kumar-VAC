# Capstone Presentation Outline: House Price Estimation Using Regression

**Candidate / Presenter**: College Capstone Student  
**Domain**: Applied Machine Learning & Predictive Analytics  
**Architecture**: Python 3.10 | Scikit-Learn | Streamlit  

---

## Slide 1: Introduction & Project Objective
- **Title**: House Price Estimation Using Regression
- **Core Problem**:
  - Residential real estate transactions involve multi-dimensional property attributes (area, quality, location, age) that influence transaction prices.
  - Objective: Construct an interpretable, reproducible machine learning system that estimates historical property sale prices and quantifies prediction error.
- **Key Research Questions**:
  1. How accurately can standard regression techniques (Linear Regression vs. Random Forest) predict residential sale prices?
  2. Which architectural and geographic features provide the strongest predictive signals?
  3. How should model error and uncertainty be communicated to non-technical stakeholders?
- **Scope Delimitation**:
  - Historical reference model calibrated on verified sales transactions.
  - Distinctly separated from formal bank appraisals or speculative property valuation.

---

## Slide 2: Dataset, Provenance & Preprocessing Pipeline
- **Dataset Identification**:
  - Ames Housing Dataset (OpenML ID: 43926 / Dean De Cock, 2011).
  - Scope: 2,930 historical home sales in Ames, Iowa (2006–2010).
  - Target Variable: Recorded `SalePrice` in USD.
- **Selected Core Feature Set**:
  - *Location*: `Neighborhood` (Nominal categorical).
  - *Size*: `GrLivArea` (Living area sq ft) & `LotArea` (Lot size sq ft).
  - *Condition & Vintage*: `OverallQual` (1–10 rating) & `HouseAge` (`YrSold - YearBuilt`).
  - *Functionality*: `BedroomAbvGr`, `FullBath`, `GarageCars`.
- **Methodological Preprocessing Integrity**:
  - Leak-free partitioning: 70% Train (~2,051 records), 15% Validation (~440 records), 15% Held-out Test (~439 records).
  - Imputation: Median replacement for numeric features; Most-frequent / Constant for categoricals.
  - Encoding: One-Hot Encoding with `handle_unknown='ignore'`.
  - Encapsulation: Scikit-learn `Pipeline` and `ColumnTransformer` fitted strictly on training data.

---

## Slide 3: Exploratory Data Analysis (EDA) Insights
- **Target Distribution**:
  - Positively skewed distribution with a median sale price around $160,000 USD and long right tail (luxury homes up to $755,000 USD).
- **Primary Value Drivers**:
  - *Overall Quality*: Strongest linear and monotonic association with sale price; clear step-function increase at quality ratings ≥ 7.
  - *Living Area (`GrLivArea`)*: Strong positive linear trend. A few high-area properties with lower price indicate specific zoning or non-standard finishes.
  - *Neighborhood Stratification*: High variance between neighborhoods (e.g., Northridge, Stone Brook commanding premium valuations vs. Old Town, Edwards).
  - *Building Age*: Negative correlation with price; newer builds command significant price premiums.
- **Analytical Caution**:
  - High correlation indicates predictive association, not economic causality.

---

## Slide 4: Regression Models, Comparative Benchmarks & Evaluation
- **Models Evaluated**:
  1. *Baseline Dummy Regressor*: Predicts training mean (`strategy='mean'`). Baseline benchmark for any intelligent model.
  2. *Multiple Linear Regression*: Interpretable parametric benchmark modeling linear additive relationships.
  3. *Random Forest Regressor*: Non-linear ensemble model (200 trees, `min_samples_leaf=2`) capturing feature interactions and non-linearities.
- **Metrics Evaluated on Held-Out Test Set**:
  - **MAE (Mean Absolute Error)**: Average dollar magnitude of prediction error.
  - **RMSE (Root Mean Squared Error)**: Penalizes larger prediction deviations.
  - **$R^2$ Score**: Fraction of price variance explained relative to naive mean.
- **Empirical Findings Summary**:
  - *Baseline*: MAE = $60,846 USD, $R^2 = -0.010$.
  - *Linear Regression*: Demonstrates high explanatory power ($R^2 = 0.803$), MAE reduced to $22,409 USD.
  - *Random Forest Regressor*: Champion model ($R^2 = 0.855$), MAE = $19,879 USD, RMSE = $32,337 USD, effectively capturing non-linear interactions.
- **Feature Importance (Permutation Analysis)**:
  - Top predictors: `OverallQual` (+$24,764 MAE impact), `GrLivArea` (+$16,778 MAE impact), `HouseAge` (+$5,465 MAE impact), and `LotArea` (+$4,028 MAE impact).

---

## Slide 5: Interactive Streamlit Application & Communicating Uncertainty
- **System Architecture**:
  - Streamlit multi-tab dashboard reading pre-computed artifacts, validation tables, and interactive prediction pipelines.
  - Five core modules: Project Overview, Dataset Exploration, Model Performance, Interactive Price Estimator, and Limitations & Responsible Use.
- **Responsible Error Estimation (Not a Single Point Estimate)**:
  - Demonstrates honest ML communication: Point estimates are presented alongside an **indicative error-based range** ($[\hat{y} - \delta, \hat{y} + \delta]$) calibrated from validation residuals ($\pm \$27,076$ USD at the 80th percentile absolute error).
  - Alerts user when inputs represent extreme or out-of-distribution values (e.g., unusually large lot sizes or living areas).
- **Real-Time Responsiveness**:
  - Model pipeline evaluates dynamic user inputs instantly without external web service latency.

---

## Slide 6: Conclusions, Ethical Limitations & Future Work
- **Key Capstone Takeaways**:
  - Supervised regression with compact, high-signal features delivers high explanatory accuracy ($R^2 > 0.85$).
  - Simple, robust pipelines (ColumnTransformer + Random Forest) outperform complex, fragile pipelines.
  - Clear communication of model residuals and error bands builds stakeholder trust.
- **Limitations & Ethical Guardrails**:
  - *Geographic & Historical Specificity*: Ames, Iowa (2006–2010). Unusable for contemporary Indian or urban real estate.
  - *Macroeconomic Shifts*: Does not account for interest rate fluctuations or inflation post-2010.
  - *Appraisal Distinction*: Statistical estimation is not a certified property appraisal.
- **Future Directions**:
  - Integration of geocoded spatial coordinates and satellite building footprints.
  - Conformal prediction frameworks for mathematically guaranteed coverage intervals.
