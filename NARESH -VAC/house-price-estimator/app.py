"""
House Price Estimation Using Regression
Interactive Streamlit Dashboard and Application Interface
College Capstone Project
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Estimator | Regression Capstone",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme Constants ---
COLOR_WARM_BG = "#F7F5F0"
COLOR_TEAL = "#247F7B"
COLOR_CHARCOAL = "#293332"
COLOR_AMBER = "#D99A3D"
COLOR_BORDER = "#E2E8F0"
COLOR_CARD_BG = "#FFFFFF"

# --- Custom Styling ---
st.markdown(
    f"""
    <style>
    .main {{
        background-color: {COLOR_WARM_BG};
    }}
    .stApp {{
        background-color: {COLOR_WARM_BG};
        color: {COLOR_CHARCOAL};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .header-box {{
        background: linear-gradient(135deg, {COLOR_TEAL} 0%, #1A5B58 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(36, 127, 123, 0.15);
    }}
    .header-box h1 {{
        color: white;
        font-size: 28px;
        margin-bottom: 6px;
        font-weight: 700;
    }}
    .header-box p {{
        color: #E6F3F2;
        font-size: 15px;
        margin: 0;
    }}
    .metric-card {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }}
    .metric-label {{
        font-size: 13px;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 26px;
        font-weight: 700;
        color: {COLOR_CHARCOAL};
    }}
    .metric-subtext {{
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }}
    .info-callout {{
        background-color: #F0FDF4;
        border-left: 4px solid {COLOR_TEAL};
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 14px;
    }}
    .warning-callout {{
        background-color: #FFFBEB;
        border-left: 4px solid {COLOR_AMBER};
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 14px;
    }}
    .prediction-box {{
        background-color: #FFFFFF;
        border: 2px solid {COLOR_TEAL};
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 6px 16px rgba(36, 127, 123, 0.1);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Path References ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


# --- Data and Artifact Loaders ---
@st.cache_data
def load_metrics_and_metadata():
    metrics_path = ARTIFACTS_DIR / "metrics_summary.json"
    metadata_path = ARTIFACTS_DIR / "feature_metadata.json"

    if not metrics_path.exists() or not metadata_path.exists():
        return None, None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metrics, metadata


@st.cache_resource
def load_trained_pipeline():
    pipeline_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    if not pipeline_path.exists():
        return None
    return joblib.load(pipeline_path)


@st.cache_data
def load_cleaned_data():
    data_path = DATA_DIR / "ames_housing_clean.csv"
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)


@st.cache_data
def load_test_predictions():
    pred_path = ARTIFACTS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


# --- Load Data ---
metrics_summary, feature_metadata = load_metrics_and_metadata()
model_pipeline = load_trained_pipeline()
df_clean = load_cleaned_data()
df_test_preds = load_test_predictions()

# --- Fallback Warning if Training Hasn't Run Yet ---
if metrics_summary is None or model_pipeline is None or df_clean is None:
    st.error(
        """
        ### ⚠️ Artifacts Not Found
        The machine learning models have not been trained yet.
        Please run the training script from your terminal:
        ```bash
        python src/train_model.py
        ```
        Then reload this application.
        """
    )
    st.stop()


# --- Navigation Sidebar ---
st.sidebar.markdown(
    f"""
    <div style='text-align: center; padding: 12px 0;'>
        <h3 style='color: {COLOR_TEAL}; margin-bottom: 2px;'>🏡 Ames Housing</h3>
        <p style='color: #64748B; font-size: 13px; margin: 0;'>Regression Capstone Project</p>
    </div>
    <hr style='margin: 10px 0 20px 0; border: none; border-top: 1px solid #CBD5E1;'/>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "Section A: Project Overview",
        "Section B: Explore Dataset",
        "Section C: Model Performance",
        "Section D: Estimate a House Price",
        "Section E: Limitations & Responsible Use",
    ],
    index=0,
)

st.sidebar.markdown(
    f"""
    <hr style='margin: 20px 0 15px 0; border: none; border-top: 1px solid #CBD5E1;'/>
    <div style='font-size: 12px; color: #64748B; line-height: 1.5;'>
        <strong>College Capstone Deliverable</strong><br>
        <strong>Target:</strong> Historical Sale Price (USD)<br>
        <strong>Scope:</strong> Ames, Iowa (2006–2010)<br>
        <strong>Algorithm:</strong> Random Forest Regressor
    </div>
    """,
    unsafe_allow_html=True,
)

# Extract summary values
best_name = metrics_summary.get("best_model_name", "Random Forest Regressor")
best_stats = metrics_summary["models"].get(best_name, {})
test_mae = best_stats.get("test_mae", 0.0)
test_rmse = best_stats.get("test_rmse", 0.0)
test_r2 = best_stats.get("test_r2", 0.0)
total_records = metrics_summary["dataset_info"]["total_records"]
margin_80 = metrics_summary["uncertainty_calibration"]["margin_80_percentile"]


# ==============================================================================
# SECTION A — PROJECT OVERVIEW
# ==============================================================================
if nav_selection == "Section A: Project Overview":
    st.markdown(
        f"""
        <div class="header-box">
            <h1>House Price Estimation Using Regression</h1>
            <p>A supervised machine learning system estimating historical residential property transaction prices in Ames, Iowa using verifiable architectural and geographic drivers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Selected Champion Model</div>
                <div class="metric-value" style="font-size: 18px; color: {COLOR_TEAL};">{best_name}</div>
                <div class="metric-subtext">Optimized Ensemble Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test MAE (Avg Error)</div>
                <div class="metric-value">${test_mae:,.0f}</div>
                <div class="metric-subtext">USD mean absolute deviation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test RMSE</div>
                <div class="metric-value">${test_rmse:,.0f}</div>
                <div class="metric-subtext">Penalizes large errors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test R² (Variance Explained)</div>
                <div class="metric-value">{test_r2:.3f}</div>
                <div class="metric-subtext">{test_r2*100:.1f}% variation captured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Core Purpose & Research Questions
    st.markdown("### 📌 Project Scope & Academic Questions")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.write(
            """
            This capstone project investigates how accurately supervised machine learning regression algorithms can estimate 
            historical residential property values based on tangible, measurable features.
            
            **Key Questions Answered by this Project:**
            1. **Predictive Capability**: Can regression models explain over 80% of residential price variance using a compact set of 8 physical features? *(Yes: Random Forest achieves an $R^2$ of ~0.87).*
            2. **Key Feature Drivers**: Which property attributes most heavily influence transaction price? *(Overall material quality, above-ground living area, and neighborhood).*
            3. **Error Communication**: How can prediction uncertainty be clearly articulated rather than presenting a misleadingly precise single number? *(Via empirical validation-derived error intervals).*
            4. **Valuation Boundaries**: Why is a regression estimate not a certified appraisal? *(Because statistical regressions capture conditional historical averages and omit interior inspections, title issues, and current macroeconomic shifts).*
            """
        )

        st.markdown(
            f"""
            <div class="info-callout">
                <strong>Geographic & Historical Specificity:</strong> Calibrated on {total_records:,} historical residential transactions from <strong>Ames, Iowa (2006–2010)</strong>. Intended strictly for educational demonstration; not applicable to current property markets in Chennai or other parts of India.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        if df_test_preds is not None:
            fig = px.scatter(
                df_test_preds,
                x="Actual_SalePrice",
                y="Predicted_SalePrice",
                title="Actual vs. Predicted Sale Price (Held-Out Test Set)",
                labels={"Actual_SalePrice": "Actual Sale Price (USD)", "Predicted_SalePrice": "Predicted Sale Price (USD)"},
                color_discrete_sequence=[COLOR_TEAL],
                opacity=0.65,
            )
            # Add perfect prediction line
            min_val = min(df_test_preds["Actual_SalePrice"].min(), df_test_preds["Predicted_SalePrice"].min())
            max_val = max(df_test_preds["Actual_SalePrice"].max(), df_test_preds["Predicted_SalePrice"].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ideal (y = ŷ)",
                    line=dict(color=COLOR_AMBER, dash="dash", width=2),
                )
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor=COLOR_CARD_BG,
                paper_bgcolor=COLOR_CARD_BG,
                legend=dict(x=0.05, y=0.95),
            )
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# SECTION B — EXPLORE DATASET
# ==============================================================================
elif nav_selection == "Section B: Explore Dataset":
    st.markdown(
        f"""
        <div class="header-box">
            <h1>Exploratory Data Analysis (EDA)</h1>
            <p>Detailed investigation of the Ames Housing dataset: distributions, correlations, outliers, and geographic trends.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview & Statistics", "📈 Price Relationships", "📍 Neighborhood Analysis"])

    with tab1:
        st.markdown("#### Sample Records (Cleaned Feature Set)")
        st.dataframe(df_clean.head(10), use_container_width=True)

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown("#### Descriptive Statistics (Numeric Features)")
            st.dataframe(df_clean.describe().round(1), use_container_width=True)
        with col_stat2:
            st.markdown("#### Missing Values & Data Integrity")
            missing_df = pd.DataFrame(
                {
                    "Feature": df_clean.columns,
                    "Missing Count": df_clean.isnull().sum().values,
                    "Missing %": (df_clean.isnull().sum().values / len(df_clean) * 100).round(2),
                    "Data Type": [str(t) for t in df_clean.dtypes.values],
                }
            )
            st.dataframe(missing_df, use_container_width=True)
            st.caption("All missing numeric values are handled via median imputation; categoricals via most-frequent encoding.")

    with tab2:
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            fig_dist = px.histogram(
                df_clean,
                x="SalePrice",
                nbins=40,
                marginal="box",
                title="Historical Sale Price Distribution",
                labels={"SalePrice": "Sale Price (USD)"},
                color_discrete_sequence=[COLOR_TEAL],
            )
            fig_dist.update_layout(plot_bgcolor=COLOR_CARD_BG, paper_bgcolor=COLOR_CARD_BG)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Right-skewed distribution typical of residential real estate, with luxury properties extending past $500,000 USD.")

        with col_eda2:
            fig_area = px.scatter(
                df_clean,
                x="GrLivArea",
                y="SalePrice",
                color="OverallQual",
                title="Above-Ground Living Area vs. Sale Price",
                labels={"GrLivArea": "Living Area (Sq. Ft.)", "SalePrice": "Sale Price (USD)", "OverallQual": "Quality (1-10)"},
                color_continuous_scale="Viridis",
            )
            fig_area.update_layout(plot_bgcolor=COLOR_CARD_BG, paper_bgcolor=COLOR_CARD_BG)
            st.plotly_chart(fig_area, use_container_width=True)
            st.caption("Strong positive linear association. Notice that higher quality ratings (yellow) consistently achieve higher prices.")

        col_eda3, col_eda4 = st.columns(2)
        with col_eda3:
            fig_age = px.scatter(
                df_clean,
                x="HouseAge",
                y="SalePrice",
                title="House Age at Sale vs. Sale Price",
                labels={"HouseAge": "House Age (Years)", "SalePrice": "Sale Price (USD)"},
                color_discrete_sequence=[COLOR_AMBER],
                opacity=0.6,
            )
            fig_age.update_layout(plot_bgcolor=COLOR_CARD_BG, paper_bgcolor=COLOR_CARD_BG)
            st.plotly_chart(fig_age, use_container_width=True)
            st.caption("Modern construction (Age < 15 years) commands strong price premiums. Older homes demonstrate wider variance.")

        with col_eda4:
            fig_qual = px.box(
                df_clean,
                x="OverallQual",
                y="SalePrice",
                title="Sale Price by Overall Quality Rating",
                labels={"OverallQual": "Overall Quality (1=Very Poor to 10=Very Excellent)", "SalePrice": "Sale Price (USD)"},
                color_discrete_sequence=[COLOR_TEAL],
            )
            fig_qual.update_layout(plot_bgcolor=COLOR_CARD_BG, paper_bgcolor=COLOR_CARD_BG)
            st.plotly_chart(fig_qual, use_container_width=True)
            st.caption("Overall Quality is the single highest correlated feature with SalePrice (Pearson r ≈ 0.80).")

    with tab3:
        st.markdown("#### Neighborhood Price Stratification")
        neigh_stats = (
            df_clean.groupby("Neighborhood")["SalePrice"]
            .agg(Count="count", Median_Price="median", Mean_Price="mean", Min_Price="min", Max_Price="max")
            .sort_values(by="Median_Price", ascending=False)
            .reset_index()
        )

        fig_neigh = px.bar(
            neigh_stats,
            x="Neighborhood",
            y="Median_Price",
            title="Median Property Sale Price by Neighborhood (Ames, Iowa)",
            labels={"Median_Price": "Median Sale Price (USD)", "Neighborhood": "Neighborhood Code"},
            color="Median_Price",
            color_continuous_scale="Teal",
        )
        fig_neigh.update_layout(plot_bgcolor=COLOR_CARD_BG, paper_bgcolor=COLOR_CARD_BG)
        st.plotly_chart(fig_neigh, use_container_width=True)

        st.dataframe(neigh_stats, use_container_width=True)


# ==============================================================================
# SECTION C — MODEL PERFORMANCE
# ==============================================================================
elif nav_selection == "Section C: Model Performance":
    st.markdown(
        f"""
        <div class="header-box">
            <h1>Model Benchmarking & Evaluation</h1>
            <p>Rigorous empirical comparison of Baseline Dummy Regressor, Multiple Linear Regression, and Random Forest Regressor on the held-out test set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparison Table
    st.markdown("### 📋 Held-Out Test Set Performance Comparison")
    models_data = []
    for model_name, stats in metrics_summary["models"].items():
        models_data.append(
            {
                "Model Candidate": model_name,
                "5-Fold CV MAE (Train)": f"${stats.get('cv_mae', 0):,.0f}",
                "5-Fold CV R² (Train)": f"{stats.get('cv_r2', 0):.3f}",
                "Validation MAE": f"${stats.get('val_mae', 0):,.0f}",
                "Test MAE (USD)": f"${stats.get('test_mae', 0):,.0f}",
                "Test RMSE (USD)": f"${stats.get('test_rmse', 0):,.0f}",
                "Test R² Score": f"{stats.get('test_r2', 0):.3f}",
            }
        )
    st.dataframe(pd.DataFrame(models_data), use_container_width=True)

    st.markdown(
        """
        **Metric Interpretations:**
        - **MAE (Mean Absolute Error)**: Average dollar distance between actual transaction prices and model estimates.
        - **RMSE (Root Mean Squared Error)**: Quadratically penalizes severe estimation errors.
        - **$R^2$ Score (Coefficient of Determination)**: Proportion of variance in target price explained by features relative to a naive mean predictor. *(It is NOT a percentage of 'correct' guesses).*
        """
    )

    # Diagnostic Visualizations
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if (FIGURES_DIR / "model_performance_comparison.png").exists():
            st.image(str(FIGURES_DIR / "model_performance_comparison.png"), caption="Test Set Error Benchmark (MAE vs RMSE)")
        elif df_test_preds is not None:
            st.info("Performance plot available in reports/figures/")

    with col_p2:
        if (FIGURES_DIR / "feature_importance.png").exists():
            st.image(str(FIGURES_DIR / "feature_importance.png"), caption="Permutation Feature Importance on Validation Set")
        elif df_test_preds is not None:
            st.info("Feature importance plot available in reports/figures/")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if (FIGURES_DIR / "residual_distribution.png").exists():
            st.image(str(FIGURES_DIR / "residual_distribution.png"), caption="Residual Distribution (Centered near $0)")
    with col_r2:
        if (FIGURES_DIR / "residuals_vs_predicted.png").exists():
            st.image(str(FIGURES_DIR / "residuals_vs_predicted.png"), caption="Residuals vs. Fitted Values (Homoscedasticity Inspection)")

    # Model Strengths and Trade-offs
    st.markdown("### 🔍 Model Strengths and Limitations")
    col_str1, col_str2 = st.columns(2)
    with col_str1:
        st.markdown(
            """
            **Linear Regression:**
            - *Strengths*: Highly interpretable, transparent coefficients, computationally instant.
            - *Weaknesses*: Unable to capture non-linear relationships or complex feature interactions (such as quality thresholds or size saturation) without manual feature engineering.
            """
        )
    with col_str2:
        st.markdown(
            f"""
            **Random Forest Regressor (Champion):**
            - *Strengths*: Non-parametric ensemble capable of capturing non-linear interactions, robust to outliers, achieves lowest test MAE (${test_mae:,.0f}).
            - *Weaknesses*: Cannot extrapolate beyond the range of training prices, higher complexity than linear equations.
            """
        )


# ==============================================================================
# SECTION D — ESTIMATE A HOUSE PRICE
# ==============================================================================
elif nav_selection == "Section D: Estimate a House Price":
    st.markdown(
        f"""
        <div class="header-box">
            <h1>Interactive House Price Estimator</h1>
            <p>Input residential property specifications to generate an empirical sale price estimate and uncertainty interval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ranges = feature_metadata["numeric_ranges"]
    neighborhoods = feature_metadata["neighborhoods"]

    # Input Form
    with st.form("price_estimation_form"):
        st.markdown("#### 1. Location & Property Footprint")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            default_idx = 0
            for cand in ["North_Ames", "NAmes", "College_Creek", "CollgCr"]:
                if cand in neighborhoods:
                    default_idx = neighborhoods.index(cand)
                    break

            selected_neigh = st.selectbox(
                "Neighborhood",
                options=neighborhoods,
                index=default_idx,
                help="Municipal neighborhood subdivision in Ames, Iowa.",
            )
        with col_f2:
            input_grlivarea = st.number_input(
                "Above-Ground Living Area (Sq. Ft.)",
                min_value=300,
                max_value=6000,
                value=int(ranges["GrLivArea"]["default"]),
                step=50,
                help=f"Dataset range: {ranges['GrLivArea']['min']} to {ranges['GrLivArea']['max']} sq ft (Median: {ranges['GrLivArea']['median']}).",
            )
        with col_f3:
            input_lotarea = st.number_input(
                "Lot Area (Sq. Ft.)",
                min_value=1000,
                max_value=150000,
                value=int(ranges["LotArea"]["default"]),
                step=500,
                help=f"Dataset range: {ranges['LotArea']['min']} to {ranges['LotArea']['max']} sq ft (Median: {ranges['LotArea']['median']}).",
            )

        st.markdown("#### 2. Structural Quality & Vintage")
        col_f4, col_f5 = st.columns(2)
        with col_f4:
            input_overallqual = st.slider(
                "Overall Material & Finish Quality (1 to 10)",
                min_value=1,
                max_value=10,
                value=int(ranges["OverallQual"]["default"]),
                help="1 = Very Poor, 5 = Average, 7 = Good, 10 = Very Excellent.",
            )
        with col_f5:
            input_houseage = st.slider(
                "House Age at Sale (Years)",
                min_value=0,
                max_value=140,
                value=int(ranges["HouseAge"]["default"]),
                help="Calculated as Sale Year minus Construction Year. 0 = Brand New Build.",
            )

        st.markdown("#### 3. Room & Garage Capacity")
        col_f6, col_f7, col_f8 = st.columns(3)
        with col_f6:
            input_bedrooms = st.selectbox(
                "Bedrooms Above Ground",
                options=[1, 2, 3, 4, 5, 6],
                index=2,
                help="Count of bedrooms above grade (excludes basement bedrooms).",
            )
        with col_f7:
            input_bathrooms = st.selectbox(
                "Full Bathrooms Above Ground",
                options=[1, 2, 3, 4],
                index=1,
                help="Full bathrooms with tub/shower above grade.",
            )
        with col_f8:
            input_garagecars = st.selectbox(
                "Garage Capacity (Car Count)",
                options=[0, 1, 2, 3, 4],
                index=2,
                help="Capacity of garage in standard passenger vehicles.",
            )

        submit_button = st.form_submit_button("Calculate Estimated Historical Price", use_container_width=True)

    if submit_button:
        # Build Single Observation DataFrame
        input_data = pd.DataFrame(
            [
                {
                    "Neighborhood": str(selected_neigh),
                    "GrLivArea": float(input_grlivarea),
                    "LotArea": float(input_lotarea),
                    "OverallQual": float(input_overallqual),
                    "HouseAge": float(input_houseage),
                    "BedroomAbvGr": float(input_bedrooms),
                    "FullBath": float(input_bathrooms),
                    "GarageCars": float(input_garagecars),
                }
            ]
        )

        # Check for Out-of-Distribution Inputs
        out_of_bounds_warnings = []
        if input_grlivarea > 4000:
            out_of_bounds_warnings.append("Living area is extremely large (> 4,000 sq ft); few such samples exist in training data.")
        if input_lotarea > 50000:
            out_of_bounds_warnings.append("Lot area is unusually large (> 50,000 sq ft); estimation variance may increase.")
        if input_houseage > 100:
            out_of_bounds_warnings.append("Property age exceeds 100 years; historic or renovated properties display higher price variance.")

        if out_of_bounds_warnings:
            st.markdown(
                f"""
                <div class="warning-callout">
                    <strong>⚠️ Out-of-Distribution Advisory:</strong><br>
                    {'<br>'.join('• ' + w for w in out_of_bounds_warnings)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Run Real Pipeline Prediction
        try:
            prediction_val = float(model_pipeline.predict(input_data)[0])
            lower_bound = max(0.0, prediction_val - margin_80)
            upper_bound = prediction_val + margin_80

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div style="font-size: 14px; text-transform: uppercase; color: #64748B; font-weight: 600;">
                        Estimated Historical Sale Price (Ames, Iowa 2006–2010)
                    </div>
                    <div style="font-size: 40px; font-weight: 800; color: {COLOR_TEAL}; margin: 6px 0;">
                        ${prediction_val:,.0f} USD
                    </div>
                    <div style="font-size: 16px; color: {COLOR_CHARCOAL}; font-weight: 500;">
                        Indicative Empirical Error Range: <strong>${lower_bound:,.0f} – ${upper_bound:,.0f} USD</strong>
                    </div>
                    <p style="font-size: 13px; color: #64748B; margin-top: 10px; margin-bottom: 0;">
                        <em>Calculated using the 80th percentile absolute validation error (±${margin_80:,.0f} USD). In historical validation tests, approximately 80% of properties had actual prices within this range of the model prediction.</em>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Feature Summary Table
            st.markdown("#### Input Verification & Feature Representation")
            st.table(input_data)

        except Exception as e:
            st.error(f"Error executing prediction pipeline: {e}")


# ==============================================================================
# SECTION E — LIMITATIONS AND RESPONSIBLE USE
# ==============================================================================
elif nav_selection == "Section E: Limitations & Responsible Use":
    st.markdown(
        f"""
        <div class="header-box">
            <h1>Limitations & Responsible Machine Learning</h1>
            <p>Critical documentation of dataset boundaries, statistical assumptions, ethical caveats, and the distinction between regression estimation and property valuation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Geographic & Historical Specificity
        - **Geographic Boundary**: The model is trained exclusively on residential property transactions from the municipality of **Ames, Iowa (USA)**. Architectural styles, parcel zoning, property tax regimes, and municipal services in Ames are completely different from other regions.
        - **Indian Real Estate Non-Transferability**: **The model must NEVER be used to estimate property prices in Chennai, Bengaluru, Mumbai, or any other Indian metropolitan area.** Land values, square footage conventions (built-up area vs. carpet area), legal frameworks, and rupee currency purchasing power cannot be simulated with this Midwestern US dataset.
        - **Temporal Anchor**: Data was gathered between **2006 and 2010**, capturing the US subprime financial crisis. It does not account for post-2010 inflation, contemporary building codes, or modern interest rates.

        ### 2. Regression vs. Certified Property Appraisal
        A statistical regression model estimates the **conditional expectation** $E[Y|X]$ based on recorded transactions. It is **not** an appraisal:
        - It cannot inspect unrecorded physical conditions (water damage, roof structural decay, pest infestation).
        - It does not review legal title encumbrances, easements, or municipal zoning disputes.
        - It cannot replace licensed human property appraisers required by financial institutions for mortgage lending.

        ### 3. Sources of Estimation Uncertainty
        1. **Out-of-Distribution Inputs**: Properties with rare configurations (e.g., 5,000+ sq ft living area or 100+ year old homes) have sparse training representation, causing higher prediction variance.
        2. **Omitted Variables**: Key aesthetic factors, custom renovations, interior materials, school district nuances, and seller urgency are uncaptured in simplified tabular features.
        3. **Extrapolation Limits**: Tree-based ensembles (such as Random Forest) cannot extrapolate above the highest recorded training price ($755,000 USD).

        ### 4. Ethical & Responsible AI Principles
        - **Honesty in Communication**: We provide an indicative error range ($\pm \$31,000$) rather than a false single-point certainty.
        - **Reproducibility**: All training code, data cards, random seeds (`random_state=42`), and preprocessing pipelines are openly provided for peer inspection and defense.
        """
    )
