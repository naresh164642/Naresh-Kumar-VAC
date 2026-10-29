# Dataset Card: Ames Housing Dataset

## 1. Summary
The Ames Housing dataset describes the sale of individual residential properties in Ames, Iowa from 2006 to 2010. It contains detailed physical, structural, and geographic attributes along with the final recorded transaction price (`SalePrice`).

## 2. Dataset Provenance & Attribution
- **Author**: Dean De Cock (Truman State University)
- **Paper**: *"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project"*, Journal of Statistics Education, 2011.
- **Repository**: OpenML Data ID 43926 / OpenML "house_prices" (Data ID 541 / 42165)
- **DOI / URL**: https://www.openml.org/search?id=43926&sort=runs&type=data
- **License**: Public Domain (CC0 / Open Data Commons). Unrestricted educational and analytical use.

## 3. Geographic and Temporal Scope
- **Location**: Ames, Iowa, USA (Midwestern United States college town).
- **Timeframe**: 2006 – 2010 (spanning the US subprime mortgage and financial crisis period).
- **Currency**: United States Dollars (USD).

## 4. Target Variable & Features
- **Target Variable**: `SalePrice` — Historical property sale price in USD (Continuous numeric).
- **Primary Selected Features**:
  - `Neighborhood`: Categorical nominal indicating the municipal neighborhood subdivision.
  - `GrLivArea`: Continuous numeric indicating Above Grade (Ground) Living Area in sq. ft.
  - `LotArea`: Continuous numeric indicating lot size in sq. ft.
  - `OverallQual`: Ordinal numeric rating the overall material and finish from 1 (Very Poor) to 10 (Very Excellent).
  - `HouseAge`: Derived continuous numeric (`YrSold - YearBuilt`), representing building age at time of sale.
  - `BedroomAbvGr`: Discrete numeric representing count of bedrooms above ground level.
  - `FullBath`: Discrete numeric representing count of full bathrooms above ground level.
  - `GarageCars`: Discrete numeric representing garage vehicle capacity (0 to 4+ cars).

## 5. Scope & Limitations
- **Geographic Bias**: Patterns, property tax assessments, zoning rules, and square footage pricing strictly reflect Ames, Iowa. The model cannot be transferred to other regional markets (such as Chennai or other Indian metropolitan areas) where land dynamics, currency, and building typology differ completely.
- **Temporal Bias**: The transactions occurred between 2006 and 2010. They do not account for post-2010 macroeconomic inflation, changing mortgage rates, or modern construction standards.
- **Valuation Caveat**: Model predictions reflect statistical conditional means derived from historical data, **not** formal appraisal valuations, lending appraisals, or legal property assessments.
