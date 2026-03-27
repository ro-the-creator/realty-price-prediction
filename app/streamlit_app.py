from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGB_IMPORT_ERROR = None
except Exception as exc:
    XGBRegressor = None
    XGB_IMPORT_ERROR = exc

st.set_page_config(page_title="Realty Price Prediction", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_ny_listings.csv"
FEATURE_COLS = ["city_te", "house_size", "acre_lot", "bath", "bed"]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    wape = (np.abs(y_true - y_pred).sum() / np.clip(np.abs(y_true).sum(), 1, None)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WAPE": wape}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def build_artifacts(df: pd.DataFrame) -> dict:
    base_cols = ["price", "city", "house_size", "acre_lot", "bath", "bed"]
    model_df = df[base_cols].dropna().copy()

    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)

    city_te_map = train_df.groupby("city")["price"].mean()
    global_price_mean = train_df["price"].mean()

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["city_te"] = train_df["city"].map(city_te_map).fillna(global_price_mean)
    test_df["city_te"] = test_df["city"].map(city_te_map).fillna(global_price_mean)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["price"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["price"]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Use XGBoost when available; otherwise fall back to sklearn's GradientBoostingRegressor.
    if XGBRegressor is not None:
        model_name = "XGBoost Regressor"
        model = XGBRegressor(
            n_estimators=3000,
            early_stopping_rounds=80,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=10,
            gamma=0.3,
            subsample=0.70,
            colsample_bytree=0.70,
            reg_alpha=0.6,
            reg_lambda=3.5,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model_name = "GradientBoostingRegressor (fallback)"
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.7,
            random_state=42,
        )
        model.fit(X_fit, y_fit)

    fit_pred = model.predict(X_fit)
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "fit": regression_metrics(y_fit, fit_pred),
        "validation": regression_metrics(y_val, val_pred),
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred),
    }

    overfit_ratio_val_fit = metrics["validation"]["RMSE"] / max(metrics["fit"]["RMSE"], 1e-9)
    overfit_ratio_test_fit = metrics["test"]["RMSE"] / max(metrics["fit"]["RMSE"], 1e-9)

    if overfit_ratio_test_fit <= 1.15:
        fit_label = "Low overfitting risk"
    elif overfit_ratio_test_fit <= 1.35:
        fit_label = "Moderate overfitting risk"
    else:
        fit_label = "High overfitting risk"

    residuals = y_test - test_pred
    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    feature_ranges = {
        "house_size": (float(model_df["house_size"].quantile(0.01)), float(model_df["house_size"].quantile(0.99))),
        "acre_lot": (float(model_df["acre_lot"].quantile(0.01)), float(model_df["acre_lot"].quantile(0.99))),
        "bath": (float(model_df["bath"].quantile(0.01)), float(model_df["bath"].quantile(0.99))),
        "bed": (float(model_df["bed"].quantile(0.01)), float(model_df["bed"].quantile(0.99))),
    }

    return {
        "model": model,
        "model_name": model_name,
        "xgb_available": XGBRegressor is not None,
        "xgb_import_error": None if XGB_IMPORT_ERROR is None else str(XGB_IMPORT_ERROR),
        "city_te_map": city_te_map,
        "global_price_mean": float(global_price_mean),
        "metrics": metrics,
        "fit_label": fit_label,
        "overfit_ratio_val_fit": float(overfit_ratio_val_fit),
        "overfit_ratio_test_fit": float(overfit_ratio_test_fit),
        "importance_df": importance_df,
        "y_test": y_test,
        "test_pred": test_pred,
        "residuals": residuals,
        "feature_ranges": feature_ranges,
        "row_counts": {
            "train": len(train_df),
            "test": len(test_df),
            "fit": len(X_fit),
            "validation": len(X_val),
        },
    }


def page_intro(df: pd.DataFrame, artifacts: dict) -> None:
    st.title("Realty Price Prediction")
    st.write("This app uses a boosted tree regressor to estimate home prices.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Train Rows", f"{artifacts['row_counts']['train']:,}")
    col3.metric("Test Rows", f"{artifacts['row_counts']['test']:,}")

    st.subheader("Model Setup")
    st.write(f"Model in use: {artifacts['model_name']}")
    st.write("Target: price")
    st.write("Features: city_te, house_size, acre_lot, bath, bed")
    st.write("City target encoding is built from training data only to prevent leakage.")
    if not artifacts["xgb_available"]:
        st.info(
            "Running in compatibility mode: XGBoost is unavailable on this machine, "
            "so the app is using a backup model. This is expected on managed computers."
        )
        if artifacts["xgb_import_error"]:
            with st.expander("Technical details (optional)"):
                st.text(artifacts["xgb_import_error"])

    st.subheader("Model Feature Customization")
    st.write("Adjust inputs below. Prediction and chart update automatically.")

    city_list = sorted(artifacts["city_te_map"].index.tolist())
    selected_city = st.selectbox("City", city_list, key="intro_city")

    city_default = float(
        artifacts["city_te_map"].get(selected_city, artifacts["global_price_mean"])
    )
    use_city_default = st.checkbox(
        "Use selected city target encoding", value=True, key="intro_use_city_default"
    )

    city_te_value = st.number_input(
        "city_te",
        min_value=0.0,
        value=city_default,
        step=1000.0,
        disabled=use_city_default,
        key="intro_city_te",
    )
    if use_city_default:
        city_te_value = city_default

    house_min, house_max = artifacts["feature_ranges"]["house_size"]
    lot_min, lot_max = artifacts["feature_ranges"]["acre_lot"]
    bath_min, bath_max = artifacts["feature_ranges"]["bath"]
    bed_min, bed_max = artifacts["feature_ranges"]["bed"]

    house_min_i = int(max(100, np.floor(house_min)))
    house_max_i = int(np.ceil(house_max))
    bath_min_i = int(max(1, np.floor(bath_min)))
    bath_max_i = int(max(bath_min_i, np.ceil(bath_max)))
    bed_min_i = int(max(1, np.floor(bed_min)))
    bed_max_i = int(max(bed_min_i, np.ceil(bed_max)))

    c1, c2 = st.columns(2)
    with c1:
        house_size = st.slider(
            "house_size",
            min_value=house_min_i,
            max_value=house_max_i,
            value=min(max(1800, house_min_i), house_max_i),
            step=10,
            key="intro_house_size",
        )
        acre_lot = st.slider(
            "acre_lot",
            min_value=float(max(0.0, lot_min)),
            max_value=float(max(lot_max, max(0.0, lot_min) + 0.01)),
            value=float(min(max(0.50, max(0.0, lot_min)), max(lot_max, max(0.0, lot_min) + 0.01))),
            step=0.01,
            key="intro_acre_lot",
        )
    with c2:
        bath = st.slider(
            "bath",
            min_value=bath_min_i,
            max_value=bath_max_i,
            value=min(max(2, bath_min_i), bath_max_i),
            step=1,
            key="intro_bath",
        )
        bed = st.slider(
            "bed",
            min_value=bed_min_i,
            max_value=bed_max_i,
            value=min(max(3, bed_min_i), bed_max_i),
            step=1,
            key="intro_bed",
        )

    input_row = {
        "city_te": float(city_te_value),
        "house_size": float(house_size),
        "acre_lot": float(acre_lot),
        "bath": float(bath),
        "bed": float(bed),
    }
    input_df = pd.DataFrame([input_row])
    pred_price = float(artifacts["model"].predict(input_df)[0])

    st.metric("Predicted Price", f"${pred_price:,.0f}")

    st.subheader("Dynamic Prediction Chart")
    sweep_feature = st.selectbox(
        "Feature to vary",
        FEATURE_COLS,
        index=1,
        key="intro_sweep_feature",
    )

    if sweep_feature == "city_te":
        city_values = artifacts["city_te_map"].values
        sweep_min = float(np.quantile(city_values, 0.05))
        sweep_max = float(np.quantile(city_values, 0.95))
    elif sweep_feature == "house_size":
        sweep_min, sweep_max = float(house_min_i), float(house_max_i)
    elif sweep_feature == "acre_lot":
        sweep_min = float(max(0.0, lot_min))
        sweep_max = float(max(lot_max, sweep_min + 0.01))
    elif sweep_feature == "bath":
        sweep_min, sweep_max = float(bath_min_i), float(bath_max_i)
    else:
        sweep_min, sweep_max = float(bed_min_i), float(bed_max_i)

    sweep_values = np.linspace(sweep_min, sweep_max, 60)
    sweep_df = pd.DataFrame([input_row] * len(sweep_values))
    sweep_df[sweep_feature] = sweep_values
    sweep_preds = artifacts["model"].predict(sweep_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sweep_values, sweep_preds, linewidth=2)
    ax.scatter([input_row[sweep_feature]], [pred_price], color="red", s=60, zorder=3)
    ax.set_title(f"Predicted Price vs {sweep_feature}")
    ax.set_xlabel(sweep_feature)
    ax.set_ylabel("Predicted Price")
    ax.grid(alpha=0.25)
    st.pyplot(fig)


def page_residual_diagnostics(artifacts: dict) -> None:
    st.title("Residual/Diagnostics")

    residuals = artifacts["residuals"]
    test_pred = artifacts["test_pred"]
    y_test = artifacts["y_test"]

    st.subheader("Residual Summary")
    st.write(f"Mean residual: {residuals.mean():,.2f}")
    st.write(f"Residual std: {residuals.std():,.2f}")
    st.write(f"Median absolute residual: {np.median(np.abs(residuals)):,.2f}")
    st.write(
        f"Corr(|residuals|, prediction): {np.corrcoef(np.abs(residuals), test_pred)[0, 1]:.3f}"
    )
    _, normality_p = stats.normaltest(residuals)
    st.write(f"Residual normality p-value: {normality_p:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=test_pred, y=residuals, alpha=0.35, ax=axes[0])
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs Predicted")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual")

    sns.histplot(residuals, bins=40, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot of Residuals")
    st.pyplot(fig2)

    st.subheader("Feature Importance")
    st.dataframe(artifacts["importance_df"], use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=artifacts["importance_df"], x="importance", y="feature", orient="h", ax=ax3)
    ax3.set_title("Model Feature Importance")
    ax3.set_xlabel("Importance")
    ax3.set_ylabel("Feature")
    st.pyplot(fig3)


def page_model_summary(artifacts: dict) -> None:
    st.title("Model Summary")

    metric_rows = []
    for split_name in ["fit", "validation", "train", "test"]:
        row = {"split": split_name}
        row.update(artifacts["metrics"][split_name])
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Overfitting Check")
    st.write(f"Validation/Fit RMSE ratio: {artifacts['overfit_ratio_val_fit']:.3f}")
    st.write(f"Test/Fit RMSE ratio: {artifacts['overfit_ratio_test_fit']:.3f}")
    st.write(f"Interpretation: {artifacts['fit_label']}")


# App bootstrap
df = load_data()
artifacts = build_artifacts(df)

pages = {
    "Intro Page": page_intro,
    "Residual/Diagnostics": page_residual_diagnostics,
    "Model Summary": page_model_summary,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))
if selected_page == "Intro Page":
    page_intro(df, artifacts)
else:
    pages[selected_page](artifacts)
