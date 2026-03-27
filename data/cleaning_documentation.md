# Data Cleaning Documentation: Realty Price Prediction

This document outlines the data cleaning steps performed in the `cleaning-eda.ipynb` notebook for the New York subset of the realtor.com dataset.

## 1. Data Loading and Initial Inspection
- Loaded the raw data from `realtor-data.csv`.
- Renamed the `zip_code` column to `postal_code` for clarity.

## 2. Filtering for New York State
- Used fuzzy string matching to select rows where the `state` column closely matches "New York" (similarity > 80).

## 3. Data Quality Assessment
- **Missingness:**
  - Identified missing values in key columns: `bed`, `bath`, `price`, `house_size`, `acre_lot`.
  - Noted that `house_size` and `acre_lot` had up to 30% missing data.
- **Logical Checks:**
  - Verified unique values in `state` (included territories and DC).
  - Noted that `street` is encoded for privacy.
  - Observed that `acre_lot` values are mostly below 1.
- **Outlier Detection:**
  - Found extreme outliers in `bed` (max 142), `bath` (max 123), and `price` (values of 0).
- **Data Types:**
  - Planned to convert `price`, `bed`, `bath`, and `house_size` to integers after checking for non-integer values.

## 4. Cleaning Steps
- **Drop Rows with Critical Missing Values:**
  - Dropped rows where any of `bed`, `bath`, `price`, `house_size`, or `acre_lot` were missing.
- **Outlier Removal (IQR Method):**
  - For `bed`, `price`, `acre_lot`, and `house_size`, removed rows outside 1.5x the interquartile range (IQR).
  - For `bath`, further removed rows where `bath > 10` (after review, these were deemed unrealistic for modeling).
- **Data Type Conversion:**
  - Converted `price`, `bed`, `bath`, and `house_size` columns to integer type after confirming values were effectively integers.

## 5. Output
- Saved the cleaned New York data to `data/clean_ny_data.csv` for modeling.

---

**Note:**
- All cleaning steps were performed in the notebook `notebooks/cleaning-eda.ipynb`.
- The cleaning process prioritized data quality for modeling, with a focus on removing extreme outliers and ensuring completeness of key features.
