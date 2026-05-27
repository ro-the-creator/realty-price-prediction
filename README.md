# Realty Price Prediction

An end-to-end real estate price prediction project built from scraped realtor.com listing data, with a focus on New York records.

<div align='center'>

#### **The final, deployed model can be found [here](https://realty-prediction.streamlit.app/). Realty price prediction can be conducted by adjusting home listing features (bedroom/bathroom count, acreage, etc.).**

<img width="961" height="625" alt="image" src="https://github.com/user-attachments/assets/aaec528c-c90d-4eac-8fe0-11de9b9d95aa" />

</div>

## Project Purpose

The project goal is to estimate home prices from core listing features and provide a transparent modeling workflow that stakeholders can review and reproduce.

**Primary objectives:**

- Build a reliable cleaning pipeline suitable for modeling.
- Establish baseline and simple benchmark models.
- Improve performance with a tuned boosted-tree model.
- Expose model behavior in a stakeholder-friendly Streamlit app.

## Project Actions

### 1. Data Cleaning and Feature Engineering

Source notebook: [notebooks/cleaning-eda.ipynb](notebooks/cleaning-eda.ipynb)

Actions completed:
- Loaded raw listing data and standardized column naming.
- Filtered New York records using fuzzy state matching.
- Addressed missingness in critical predictors.
- Removed extreme outliers using IQR-based rules.
- Applied a manual cap to unrealistic bathroom counts.
- Cast selected features to integer types for consistency.
- Created a smoothed city target-encoding feature named city_te.
- Exported the cleaned dataset for downstream modeling.

Automation script:
- [pipeline/cleaning_pipeline.py](pipeline/cleaning_pipeline.py) reproduces these transformations as a repeatable pipeline.

> [!NOTE]
> Exact specifications regarding cleaning and transformations are detailed [here](./pipeline).

### 2. Baseline and Simple Modeling

Source notebook: [notebooks/model-training.ipynb](notebooks/model-training.ipynb)

Actions completed:
- Trained a baseline city-average model.
- Trained a multiple linear regression model using bed, bath, and city_te.
- Evaluated performance with MAE, RMSE, and R2.
- Checked multicollinearity (correlation and VIF).
- Performed residual diagnostics to assess model assumptions.

### 3. Final Tuned Boosted Model

Source notebook: [notebooks/boost-model.ipynb](notebooks/boost-model.ipynb)

Actions completed:
- Built leakage-safe train/test target encoding for city.
- Tuned an XGBoost regressor with constrained parameter search.
- Evaluated fit, validation, and test metrics.
- Assessed overfitting via RMSE gap and ratio checks.
- Reviewed feature importance and residual behavior.

### 4. App Delivery

Application script: [app/streamlit_app.py](app/streamlit_app.py)

Capabilities:
- Interactive price prediction from user-controlled inputs.
- Dynamic sensitivity chart for feature sweeps.
- Residual and diagnostic visualizations.
- Model summary view with split-level performance metrics.
- Automatic fallback model when XGBoost is unavailable.

## Repository Structure

- [notebooks](notebooks): cleaning, baseline/simple modeling, and tuned boosted-model analysis.
- [pipeline](pipeline): reusable cleaning script and cleaning documentation.
- [app](app): Streamlit application and app dependencies.
- [pipeline/data](pipeline/data): local input/output area for pipeline runs.

## Quickstart

From the repository root:

1. Install dependencies

> [!NOTE]
> `app/requirements.txt` is intended for running the Streamlit app locally. The cleaning pipeline also requires `rapidfuzz`.

```bash
python -m pip install -r app/requirements.txt
python -m pip install rapidfuzz
```

2. Place raw CSV at:

- pipeline/data/input/realtor-data.csv

3. Run the cleaning pipeline:

```bash
python pipeline/cleaning_pipeline.py
```

4. Expected pipeline output:

- pipeline/data/output/cleaned_ny_listings.csv

5. Run the Streamlit app:

The app reads the cleaned dataset from `pipeline/data/output/cleaned_ny_listings.csv`.

```bash
streamlit run app/streamlit_app.py
```

## AI Usage

AI-assisted tools were used during this project to help with:
- Debugging

- Documentation cleanup

- Initial creation of the reusable cleaning pipeline in [pipeline/cleaning_pipeline.py](pipeline/cleaning_pipeline.py).

All AI suggestions were manually verified through several means. Debugging and Documentation was verified through human review, ensuring consistency and project alignment with my own goals.

Pipeline creation was validated by taking a small batch of the dataset and running it through the pipeline, ensuring the output matched cleaning/transformations done in the [manual cleaning notebook](notebooks/cleaning-eda.ipynb). Manually review was also conducted in the [pipeline file](pipeline/cleaning_pipeline.py) for additional validation.

## Acknowledgements
Data was collected from https://www.realtor.com/ - A real estate listing website operated by the News Corp subsidiary Move, Inc. and based in Santa Clara, California. It is the second most visited real estate listing website in the United States as of 2024, with over 100 million monthly active users.
