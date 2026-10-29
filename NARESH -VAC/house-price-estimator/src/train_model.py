"""
House Price Estimation - Model Training and Evaluation Pipeline
Capstone Project: House Price Estimation Using Regression
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for folder in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Set visual styling for publication-grade figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "teal": "#247F7B",
    "charcoal": "#293332",
    "amber": "#D99A3D",
    "warm_bg": "#F7F5F0",
    "light_gray": "#E2E8F0"
}
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def load_dataset() -> pd.DataFrame:
    """
    Fetch the Ames Housing dataset from OpenML or load from local cache.
    Falls back gracefully if OpenML is unreachable.
    """
    raw_cache_path = DATA_DIR / "ames_housing_raw.csv"
    if raw_cache_path.exists():
        logger.info(f"Loading cached Ames Housing dataset from {raw_cache_path}")
        df = pd.read_csv(raw_cache_path)
        return df

    logger.info("Fetching Ames Housing dataset from OpenML (Data ID: 43926)...")
    try:
        # First attempt: OpenML ID 43926
        data = fetch_openml(data_id=43926, as_frame=True, parser="auto")
        df = data.frame
        logger.info("Successfully fetched dataset via data_id=43926.")
    except Exception as e1:
        logger.warning(f"Fetch via data_id=43926 failed: {e1}. Trying 'house_prices'...")
        try:
            # Second attempt: standard house_prices
            data = fetch_openml(name="house_prices", as_frame=True, parser="auto")
            df = data.frame
            logger.info("Successfully fetched dataset via name='house_prices'.")
        except Exception as e2:
            logger.warning(f"Fetch via name='house_prices' failed: {e2}. Trying OpenML ID 541...")
            try:
                data = fetch_openml(data_id=541, as_frame=True, parser="auto")
                df = data.frame
                logger.info("Successfully fetched dataset via data_id=541.")
            except Exception as e3:
                logger.error(f"All OpenML attempts failed: {e3}")
                raise RuntimeError(
                    "Unable to download Ames Housing dataset from OpenML. "
                    "Ensure internet connection or place ames_housing_raw.csv in the data/ folder."
                )

    df.to_csv(raw_cache_path, index=False)
    logger.info(f"Cached raw dataset to {raw_cache_path}")
    return df


def clean_and_prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, filter to our target and core 8 features,
    calculate derived HouseAge, and save clean dataset.
    """
    logger.info("Standardizing column names and extracting core features...")
    df = df_raw.copy()

    # Column mapping to handle slight variations between OpenML versions
    col_map = {
        "SalePrice": "SalePrice",
        "saleprice": "SalePrice",
        "Sale_Price": "SalePrice",
        "sale_price": "SalePrice",
        "Neighborhood": "Neighborhood",
        "neighborhood": "Neighborhood",
        "GrLivArea": "GrLivArea",
        "grlivarea": "GrLivArea",
        "Gr_Liv_Area": "GrLivArea",
        "gr_liv_area": "GrLivArea",
        "LotArea": "LotArea",
        "lotarea": "LotArea",
        "Lot_Area": "LotArea",
        "lot_area": "LotArea",
        "OverallQual": "OverallQual",
        "overallqual": "OverallQual",
        "Overall_Qual": "OverallQual",
        "overall_qual": "OverallQual",
        "YearBuilt": "YearBuilt",
        "yearbuilt": "YearBuilt",
        "Year_Built": "YearBuilt",
        "year_built": "YearBuilt",
        "YrSold": "YrSold",
        "yrsold": "YrSold",
        "Yr_Sold": "YrSold",
        "yr_sold": "YrSold",
        "Year_Sold": "YrSold",
        "year_sold": "YrSold",
        "BedroomAbvGr": "BedroomAbvGr",
        "bedroomabvgr": "BedroomAbvGr",
        "Bedroom_AbvGr": "BedroomAbvGr",
        "bedroom_abvgr": "BedroomAbvGr",
        "FullBath": "FullBath",
        "fullbath": "FullBath",
        "Full_Bath": "FullBath",
        "full_bath": "FullBath",
        "GarageCars": "GarageCars",
        "garagecars": "GarageCars",
        "Garage_Cars": "GarageCars",
        "garage_cars": "GarageCars",
    }

    # Rename existing columns
    rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
    df = df.rename(columns=rename_dict)

    # Validate target exists
    if "SalePrice" not in df.columns:
        raise KeyError(f"Target column 'SalePrice' not found. Available columns: {list(df.columns[:10])}")

    # Map text OverallQual to numeric 1-10 if categorical
    qual_map = {
        "very_poor": 1,
        "poor": 2,
        "fair": 3,
        "below_average": 4,
        "average": 5,
        "above_average": 6,
        "good": 7,
        "very_good": 8,
        "excellent": 9,
        "very_excellent": 10,
    }
    if df["OverallQual"].dtype == object or str(df["OverallQual"].dtype) == "category":
        cleaned_qual = (
            df["OverallQual"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(qual_map)
        )
        df["OverallQual"] = cleaned_qual.fillna(pd.to_numeric(df["OverallQual"], errors="coerce"))

    # Calculate HouseAge = YrSold - YearBuilt
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = pd.to_numeric(df["YrSold"], errors="coerce") - pd.to_numeric(df["YearBuilt"], errors="coerce")
    elif "YearBuilt" in df.columns:
        df["HouseAge"] = 2010 - pd.to_numeric(df["YearBuilt"], errors="coerce")
    else:
        df["HouseAge"] = 30  # fallback default

    # Clip negative or invalid house age to 0
    df["HouseAge"] = df["HouseAge"].clip(lower=0)

    # Core required features
    selected_cols = [
        "Neighborhood",
        "GrLivArea",
        "LotArea",
        "OverallQual",
        "HouseAge",
        "BedroomAbvGr",
        "FullBath",
        "GarageCars",
        "SalePrice",
    ]

    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns from raw data: {missing_cols}")

    df_clean = df[selected_cols].copy()

    # Cast numeric types cleanly
    numeric_cols = [
        "GrLivArea",
        "LotArea",
        "OverallQual",
        "HouseAge",
        "BedroomAbvGr",
        "FullBath",
        "GarageCars",
        "SalePrice",
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # String neighborhood
    df_clean["Neighborhood"] = df_clean["Neighborhood"].astype(str).str.strip()

    # Drop records where target SalePrice is missing or <= 0
    df_clean = df_clean.dropna(subset=["SalePrice"])
    df_clean = df_clean[df_clean["SalePrice"] > 0]

    # Save cleaned data
    clean_csv_path = DATA_DIR / "ames_housing_clean.csv"
    df_clean.to_csv(clean_csv_path, index=False)
    logger.info(f"Cleaned dataset saved: {clean_csv_path} with {len(df_clean)} records.")

    return df_clean


def build_preprocessor(categorical_cols, numeric_cols):
    """
    Create a robust Scikit-Learn ColumnTransformer for consistent preprocessing.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def generate_eda_figures(df: pd.DataFrame):
    """
    Generate exploratory data analysis visualizations saved to reports/figures/
    """
    logger.info("Generating EDA figures...")

    # 1. Price Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["SalePrice"], kde=True, color=PALETTE["teal"], ax=ax, bins=35)
    mean_val = df["SalePrice"].mean()
    median_val = df["SalePrice"].median()
    ax.axvline(mean_val, color=PALETTE["amber"], linestyle="--", linewidth=2, label=f"Mean: ${mean_val:,.0f}")
    ax.axvline(median_val, color=PALETTE["charcoal"], linestyle="-", linewidth=2, label=f"Median: ${median_val:,.0f}")
    ax.set_title("Distribution of Historical Property Sale Prices (Ames, Iowa)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Sale Price (USD)", fontsize=11)
    ax.set_ylabel("Property Count", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_price_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Living Area vs Price
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["GrLivArea"],
        df["SalePrice"],
        c=df["OverallQual"],
        cmap="viridis",
        alpha=0.65,
        edgecolors="none",
        s=30,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Overall Quality (1-10)", fontsize=10)
    ax.set_title("Above-Ground Living Area vs. Sale Price", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Above-Ground Living Area (Sq. Ft.)", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_area_vs_price.png", dpi=200)
    plt.close(fig)

    # 3. Overall Quality vs Price (Boxplot)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        x="OverallQual",
        y="SalePrice",
        data=df,
        ax=ax,
        palette="crest",
        boxprops=dict(alpha=0.8),
    )
    ax.set_title("Historical Sale Price Across Overall Material & Finish Quality", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Overall Quality Rating (1 = Very Poor to 10 = Very Excellent)", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_quality_vs_price.png", dpi=200)
    plt.close(fig)

    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "HouseAge", "LotArea", "BedroomAbvGr"]
    corr = df[num_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title("Pearson Correlation Heatmap (Selected Numeric Features)", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    logger.info("EDA figures successfully generated.")


def evaluate_predictions(y_true, y_pred):
    """Calculate MAE, RMSE, and R2 cleanly."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def train_and_evaluate():
    """
    Main training execution function.
    """
    # 1. Load and clean
    df = load_dataset()
    df_clean = clean_and_prepare_data(df)

    # Generate EDA charts
    generate_eda_figures(df_clean)

    # 2. Features and target
    target_col = "SalePrice"
    cat_cols = ["Neighborhood"]
    num_cols = [
        "GrLivArea",
        "LotArea",
        "OverallQual",
        "HouseAge",
        "BedroomAbvGr",
        "FullBath",
        "GarageCars",
    ]
    feature_cols = cat_cols + num_cols

    X = df_clean[feature_cols]
    y = df_clean[target_col].values

    # 3. Data Split: 70% Train, 15% Validation, 15% Held-out Test
    logger.info("Splitting dataset: 70% Train, 15% Validation, 15% Held-Out Test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    logger.info(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Construct Models and Pipelines
    preprocessor = build_preprocessor(cat_cols, num_cols)

    models = {
        "Baseline (Mean Dummy)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    pipelines = {}
    cv_results = {}
    test_results = {}
    val_predictions = {}
    test_predictions = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Training and validating {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

        # Cross-validation on Train Set
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
            n_jobs=-1,
        )

        cv_results[name] = {
            "CV_MAE_Mean": float(-cv_scores["test_neg_mean_absolute_error"].mean()),
            "CV_MAE_Std": float(cv_scores["test_neg_mean_absolute_error"].std()),
            "CV_RMSE_Mean": float(-cv_scores["test_neg_root_mean_squared_error"].mean()),
            "CV_RMSE_Std": float(cv_scores["test_neg_root_mean_squared_error"].std()),
            "CV_R2_Mean": float(cv_scores["test_r2"].mean()),
            "CV_R2_Std": float(cv_scores["test_r2"].std()),
        }

        # Fit on entire Train set
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline

        # Validation set evaluation
        val_pred = pipeline.predict(X_val)
        val_predictions[name] = val_pred

        # Final evaluation on Held-out Test set
        test_pred = pipeline.predict(X_test)
        test_predictions[name] = test_pred
        test_results[name] = evaluate_predictions(y_test, test_pred)

    # Select Best Model based on Validation Performance
    val_metrics = {name: evaluate_predictions(y_val, preds) for name, preds in val_predictions.items()}
    best_model_name = min(val_metrics, key=lambda k: val_metrics[k]["MAE"])
    best_pipeline = pipelines[best_model_name]
    logger.info(f"Selected Best Model: {best_model_name}")

    # 5. Uncertainty Calibration (Validation Set Residuals)
    # Never use test set for error-range calibration
    val_actual = y_val
    best_val_preds = val_predictions[best_model_name]
    val_residuals = val_actual - best_val_preds
    val_abs_errors = np.abs(val_residuals)

    # 80th percentile absolute error provides an intuitive empirical error margin
    margin_80 = float(np.percentile(val_abs_errors, 80))
    margin_mae = float(np.mean(val_abs_errors))

    # 6. Feature Importance via Permutation Importance on Validation Set
    logger.info("Computing permutation feature importance on validation set...")
    perm_result = permutation_importance(
        best_pipeline,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    feature_importances = {
        col: float(importance) for col, importance in zip(X.columns, perm_result.importances_mean)
    }
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # 7. Generate Evaluation Figures
    logger.info("Generating model evaluation figures...")

    # A: Actual vs Predicted (Best Model on Test Set)
    fig, ax = plt.subplots(figsize=(8, 6))
    best_test_preds = test_predictions[best_model_name]
    ax.scatter(y_test, best_test_preds, alpha=0.6, color=PALETTE["teal"], edgecolors="none", s=35, label="Test Predictions")
    min_val = min(y_test.min(), best_test_preds.min())
    max_val = max(y_test.max(), best_test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=PALETTE["amber"], linestyle="--", linewidth=2, label="Perfect Prediction Line (y = ŷ)")
    ax.set_title(f"Actual vs. Predicted Sale Price ({best_model_name})", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Actual Sale Price (USD)", fontsize=11)
    ax.set_ylabel("Predicted Sale Price (USD)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # B: Residual Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    test_residuals = y_test - best_test_preds
    sns.histplot(test_residuals, kde=True, color=PALETTE["charcoal"], ax=ax, bins=35)
    ax.axvline(0, color=PALETTE["amber"], linestyle="--", linewidth=2, label="Zero Error Line")
    ax.set_title(f"Test Residuals Distribution (Actual - Predicted) - {best_model_name}", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Residual Error (USD)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_distribution.png", dpi=200)
    plt.close(fig)

    # C: Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(best_test_preds, test_residuals, alpha=0.6, color=PALETTE["teal"], s=35, edgecolors="none")
    ax.axhline(0, color=PALETTE["amber"], linestyle="--", linewidth=2, label="Zero Error Baseline")
    ax.set_title(f"Residuals vs. Predicted Price - {best_model_name}", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted Sale Price (USD)", fontsize=11)
    ax.set_ylabel("Residual (USD)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=200)
    plt.close(fig)

    # D: Model Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    models_list = list(test_results.keys())
    mae_list = [test_results[m]["MAE"] for m in models_list]
    rmse_list = [test_results[m]["RMSE"] for m in models_list]
    x_pos = np.arange(len(models_list))
    width = 0.35

    rects1 = ax.bar(x_pos - width / 2, mae_list, width, label="Test MAE (USD)", color=PALETTE["teal"], alpha=0.85)
    rects2 = ax.bar(x_pos + width / 2, rmse_list, width, label="Test RMSE (USD)", color=PALETTE["amber"], alpha=0.85)

    ax.set_ylabel("Error in USD (Lower is Better)", fontsize=11)
    ax.set_title("Test Set Error Comparison Across Models", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models_list], fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax.legend(frameon=True)

    # Add labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"${int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"${int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_performance_comparison.png", dpi=200)
    plt.close(fig)

    # E: Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5))
    feat_names = list(sorted_importances.keys())
    feat_vals = list(sorted_importances.values())
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, feat_vals, color=PALETTE["teal"], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Error Increase When Permuted (USD)", fontsize=11)
    ax.set_title(f"Permutation Feature Importance on Validation Set ({best_model_name})", fontsize=13, weight="bold", pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)

    # 8. Save Artifacts
    logger.info("Saving trained model pipelines and summary artifacts...")

    # Save Best Pipeline
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "model_pipeline.joblib")
    # Save other pipelines for comparison
    joblib.dump(pipelines["Linear Regression"], ARTIFACTS_DIR / "linear_pipeline.joblib")
    joblib.dump(pipelines["Baseline (Mean Dummy)"], ARTIFACTS_DIR / "dummy_pipeline.joblib")

    # Save summary metadata
    metrics_summary = {
        "best_model_name": best_model_name,
        "dataset_info": {
            "total_records": len(df_clean),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "target_variable": "SalePrice",
            "currency": "USD",
            "geographic_scope": "Ames, Iowa, United States",
            "temporal_scope": "2006-2010",
        },
        "models": {},
        "uncertainty_calibration": {
            "method": "Validation Residual Percentile (80th percentile)",
            "validation_mae": margin_mae,
            "margin_80_percentile": margin_80,
            "interpretation": (
                "Approximately 80% of properties in the validation set experienced an "
                f"absolute estimation error under ${margin_80:,.0f} USD."
            ),
        },
        "feature_importance": sorted_importances,
    }

    for name in models.keys():
        metrics_summary["models"][name] = {
            "cv_mae": cv_results[name]["CV_MAE_Mean"],
            "cv_rmse": cv_results[name]["CV_RMSE_Mean"],
            "cv_r2": cv_results[name]["CV_R2_Mean"],
            "val_mae": val_metrics[name]["MAE"],
            "val_rmse": val_metrics[name]["RMSE"],
            "val_r2": val_metrics[name]["R2"],
            "test_mae": test_results[name]["MAE"],
            "test_rmse": test_results[name]["RMSE"],
            "test_r2": test_results[name]["R2"],
        }

    with open(ARTIFACTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save feature metadata for Streamlit input validation & UI defaults
    feature_metadata = {
        "neighborhoods": sorted(list(df_clean["Neighborhood"].unique())),
        "numeric_ranges": {
            "GrLivArea": {
                "min": int(df_clean["GrLivArea"].min()),
                "max": int(df_clean["GrLivArea"].max()),
                "median": int(df_clean["GrLivArea"].median()),
                "default": int(df_clean["GrLivArea"].median()),
            },
            "LotArea": {
                "min": int(df_clean["LotArea"].min()),
                "max": int(df_clean["LotArea"].max()),
                "median": int(df_clean["LotArea"].median()),
                "default": int(df_clean["LotArea"].median()),
            },
            "OverallQual": {
                "min": int(df_clean["OverallQual"].min()),
                "max": int(df_clean["OverallQual"].max()),
                "median": int(df_clean["OverallQual"].median()),
                "default": 6,
            },
            "HouseAge": {
                "min": int(df_clean["HouseAge"].min()),
                "max": int(df_clean["HouseAge"].max()),
                "median": int(df_clean["HouseAge"].median()),
                "default": int(df_clean["HouseAge"].median()),
            },
            "BedroomAbvGr": {
                "min": int(df_clean["BedroomAbvGr"].min()),
                "max": int(df_clean["BedroomAbvGr"].max()),
                "median": int(df_clean["BedroomAbvGr"].median()),
                "default": 3,
            },
            "FullBath": {
                "min": int(df_clean["FullBath"].min()),
                "max": int(df_clean["FullBath"].max()),
                "median": int(df_clean["FullBath"].median()),
                "default": 2,
            },
            "GarageCars": {
                "min": int(df_clean["GarageCars"].min()),
                "max": int(df_clean["GarageCars"].max()),
                "median": int(df_clean["GarageCars"].median()),
                "default": 2,
            },
        },
    }

    with open(ARTIFACTS_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=2)

    # Save test set predictions for interactive visual inspection
    test_eval_df = X_test.copy()
    test_eval_df["Actual_SalePrice"] = y_test
    test_eval_df["Predicted_SalePrice"] = best_test_preds
    test_eval_df["Residual"] = test_residuals
    test_eval_df.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    logger.info("=== Model Training and Artifact Generation Successfully Completed ===")
    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
