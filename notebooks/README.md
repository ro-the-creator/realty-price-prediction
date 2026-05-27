# Jupyter Notebooks

Summarizing the main actions performed in the following Jupyter notebooks.

## cleaning-eda.ipynb
**Initial cleaning/processing data, followed by thorough exploratory data analysis. Done manually before eventual pipeline creation.**

- Loaded raw realtor listing data and renamed `zip_code` to `postal_code`.
- Filtered records to New York using fuzzy matching on the `state` column.
- Assessed missingness, dropped rows with critical null values, and removed outliers (IQR method).
- Applied an additional bathroom cap (`bath <= 10`) after manual review.
- Converted key numeric columns to integer type.
- Ran exploratory analysis (distributions, feature-to-price plots, and correlation heatmaps).
- Built smoothed target encoding for city (`city_te`).
- Exported cleaned output to CSV.

## model-training.ipynb
**Baseline + Simple model creation, prior to final complex model. Feature inputting and diagnostics.**

- Loaded cleaned data and prepared model-ready columns.
- Built a baseline model using city average price.
- Trained a multiple linear regression model with `bed`, `bath`, and `city_te`.
- Evaluated model quality with MAE, RMSE, and R2 metrics.
- Checked multicollinearity using correlation and VIF.
- Performed residual diagnostics (residual plots, distribution, and Q-Q plot).

## boost-model.ipynb
**Final, tuned XGBoost Regressor model. Feature inputting and diagnostics.**

- Loaded cleaned data and created a leakage-safe train/test workflow.
- Added city target encoding from train data only.
- Trained and tuned `XGBRegressor` with constrained candidate search.
- Compared fit, validation, and test performance using MAE, RMSE, R2, and WAPE.
- Evaluated generalization risk with train/validation/test RMSE gaps and ratios.
- Reviewed feature importance from the tuned XGBoost model.
- Performed residual diagnostics and visualization on test predictions.
