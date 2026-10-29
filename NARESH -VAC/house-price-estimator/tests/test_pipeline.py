"""
Unit and Integration Tests for House Price Estimation Pipeline
Capstone 2: House Price Estimation Using Regression
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"


def test_clean_data_exists_and_valid():
    """Verify cleaned dataset exists and has valid columns and records."""
    clean_path = DATA_DIR / "ames_housing_clean.csv"
    assert clean_path.exists(), "Cleaned dataset data/ames_housing_clean.csv does not exist."

    df = pd.read_csv(clean_path)
    expected_cols = [
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
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} missing from cleaned dataset."

    assert len(df) > 1000, f"Cleaned dataset should have > 1000 records, got {len(df)}."
    assert (df["SalePrice"] > 0).all(), "All SalePrice entries must be strictly positive."
    assert (df["HouseAge"] >= 0).all(), "HouseAge must be non-negative."


def test_artifacts_exist():
    """Verify model pipeline and metadata artifacts exist."""
    pipeline_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics_summary.json"
    metadata_path = ARTIFACTS_DIR / "feature_metadata.json"

    assert pipeline_path.exists(), "Trained model pipeline artifacts/model_pipeline.joblib does not exist."
    assert metrics_path.exists(), "artifacts/metrics_summary.json does not exist."
    assert metadata_path.exists(), "artifacts/feature_metadata.json does not exist."


def test_metrics_integrity():
    """Verify metrics values are calculated and within reasonable bounds."""
    metrics_path = ARTIFACTS_DIR / "metrics_summary.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "models" in metrics, "metrics_summary.json must contain 'models' key."
    assert "Random Forest Regressor" in metrics["models"], "Champion model missing from metrics."
    rf_metrics = metrics["models"]["Random Forest Regressor"]

    # R2 should be high for Random Forest on Ames Housing (typically 0.80 - 0.90)
    assert 0.70 <= rf_metrics["test_r2"] <= 0.99, f"Unexpected R2 score: {rf_metrics['test_r2']}"
    # MAE should be around $15k - $25k USD
    assert 10000 <= rf_metrics["test_mae"] <= 35000, f"Unexpected MAE: {rf_metrics['test_mae']}"
    # RMSE should be greater than MAE
    assert rf_metrics["test_rmse"] > rf_metrics["test_mae"], "RMSE must be greater than MAE."

    # Uncertainty calibration
    assert "uncertainty_calibration" in metrics
    margin = metrics["uncertainty_calibration"]["margin_80_percentile"]
    assert margin > 0, "Uncertainty margin must be positive."


def test_pipeline_prediction():
    """Verify that loaded pipeline predicts valid prices for synthetic sample."""
    pipeline_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)

    sample_input = pd.DataFrame(
        [
            {
                "Neighborhood": "NAmes",
                "GrLivArea": 1500.0,
                "LotArea": 9500.0,
                "OverallQual": 6.0,
                "HouseAge": 35.0,
                "BedroomAbvGr": 3.0,
                "FullBath": 2.0,
                "GarageCars": 2.0,
            }
        ]
    )

    pred = pipeline.predict(sample_input)
    assert len(pred) == 1, "Pipeline should return exactly one prediction."
    assert 80000 <= pred[0] <= 350000, f"Prediction {pred[0]} outside plausible range for sample."


if __name__ == "__main__":
    test_clean_data_exists_and_valid()
    test_artifacts_exist()
    test_metrics_integrity()
    test_pipeline_prediction()
    print("All unit and integration tests passed successfully!")
