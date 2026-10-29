# Ames Housing Dataset Information

## Source & Origin
- **Source**: OpenML Ames Housing Dataset (Data ID: 43926 / AmesHousing)
- **Original Compilation**: Dean De Cock, Truman State University (published in *Journal of Statistics Education*, Vol. 19, No. 3, 2011).
- **OpenML Reference**: [OpenML Data ID 43926](https://www.openml.org/search?id=43926&sort=runs&type=data)
- **License**: Public Domain / Creative Commons (educational, research, and non-commercial/commercial usage permitted with standard citation).

## Dataset Scope
- **Geographic Area**: Ames, Iowa, United States.
- **Temporal Coverage**: Historical property transactions recorded between 2006 and 2010.
- **Target Variable**: `SalePrice` — The recorded historical sale price in US Dollars (USD).
- **Intended Use**: Educational price estimation and regression analysis.
- **Important Boundary Condition**: This dataset describes residential properties in Ames, Iowa during 2006–2010. It cannot and should not be used to value contemporary homes in Chennai, other parts of India, or any modern market without recalibration.

## Key Features Used in this Project
1. `Neighborhood`: Physical locations within Ames city limits (e.g., North Ames, College Creek, Old Town).
2. `GrLivArea`: Above grade (ground) living area in square feet.
3. `LotArea`: Lot size in square feet.
4. `OverallQual`: Rates the overall material and finish of the house (Scale 1 to 10, where 10 = Very Excellent).
5. `HouseAge`: Derived feature calculated as `YrSold - YearBuilt` (capped at 0 for newly built properties).
6. `BedroomAbvGr`: Bedrooms above ground level (does not include basement bedrooms).
7. `FullBath`: Full bathrooms above grade.
8. `GarageCars`: Size of garage in car capacity.

## Local Cache
The dataset is downloaded via `scikit-learn`'s `fetch_openml` (ID 43926 or "house_prices") and cached locally in `data/ames_housing_raw.csv` and `data/ames_housing_clean.csv` to ensure offline reproducibility.
