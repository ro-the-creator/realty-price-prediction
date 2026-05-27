from pathlib import Path
import base64

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BANNER_PATH = PROJECT_ROOT / "app" / "assets" / "saunders.png"
DATA_PATH_CANDIDATES = [
    PROJECT_ROOT / "pipeline" / "data" / "output" / "cleaned_ny_listings.csv",
    PROJECT_ROOT / "data" / "cleaned_ny_listings.csv",
]
FEATURE_COLS = ["city_te", "house_size", "acre_lot", "bath", "bed"]
FEATURE_LABELS = {
    "city_te": "Average Listing Price in City",
    "house_size": "House Square Footage",
    "acre_lot": "Property Acreage",
    "bath": "Bathroom Count",
    "bed": "Bedroom Count",
}


def resolve_data_path() -> Path:
    for candidate in DATA_PATH_CANDIDATES:
        if candidate.exists():
            return candidate

    searched_paths = "\n".join(f"- {path}" for path in DATA_PATH_CANDIDATES)
    raise FileNotFoundError(
        "Could not find cleaned dataset. Checked:\n"
        f"{searched_paths}"
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    wape = (np.abs(y_true - y_pred).sum() / np.clip(np.abs(y_true).sum(), 1, None)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WAPE": wape}


def friendly_feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def render_banner() -> None:
    if not BANNER_PATH.exists():
        return

    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        banner_base64 = base64.b64encode(BANNER_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <a href="https://www.hamptonsrealestate.com/eng" target="_blank" rel="noopener noreferrer">
                <img
                    src="data:image/png;base64,{banner_base64}"
                    alt="Saunders banner"
                    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
                />
            </a>
            """,
            unsafe_allow_html=True,
        )


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp,
        .stApp p,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp label,
        .stApp span,
        .stApp div,
        .stApp li,
        .stApp a,
        .stApp caption,
        .stApp small {
            color: #412d1b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path())


@st.cache_resource
def build_artifacts(df: pd.DataFrame) -> dict:
    if "city_te" not in df.columns:
        city_price_map = df.groupby("city")["price"].mean()
        df = df.copy()
        df["city_te"] = df["city"].map(city_price_map).fillna(df["price"].mean())

    base_cols = ["price", "city", "city_te", "house_size", "acre_lot", "bath", "bed"]
    model_df = df[base_cols].dropna().copy()

    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)

    city_te_map = train_df.groupby("city")["price"].mean()
    city_count_map = train_df.groupby("city")["price"].count()
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
    importance_df["feature"] = importance_df["feature"].map(friendly_feature_name)

    feature_ranges = {
        "house_size": (float(model_df["house_size"].quantile(0.01)), float(model_df["house_size"].quantile(0.99))),
        "acre_lot": (float(model_df["acre_lot"].quantile(0.01)), float(model_df["acre_lot"].quantile(0.99))),
        "bath": (float(model_df["bath"].quantile(0.01)), float(model_df["bath"].quantile(0.99))),
        "bed": (float(model_df["bed"].quantile(0.01)), float(model_df["bed"].quantile(0.99))),
        "city_te": (float(model_df["city_te"].quantile(0.01)), float(model_df["city_te"].quantile(0.99))),
    }

    feature_defaults = {
        "house_size": int(round(model_df["house_size"].median())),
        "acre_lot": float(round(model_df["acre_lot"].median(), 2)),
        "bath": int(round(model_df["bath"].median())),
        "bed": int(round(model_df["bed"].median())),
        "city_te": float(round(model_df["city_te"].median(), 2)),
    }

    return {
        "model": model,
        "model_name": model_name,
        "xgb_available": XGBRegressor is not None,
        "xgb_import_error": None if XGB_IMPORT_ERROR is None else str(XGB_IMPORT_ERROR),
        "city_te_map": city_te_map,
        "city_count_map": city_count_map,
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
        "feature_defaults": feature_defaults,
        "row_counts": {
            "train": len(train_df),
            "test": len(test_df),
            "fit": len(X_fit),
            "validation": len(X_val),
        },
    }


def page_estimator(df: pd.DataFrame, artifacts: dict) -> None:
    del df

    st.markdown(
        "<h1 style='text-align: center;'>Home Value Estimator</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Enter the property details below to estimate a home value.</p>",
        unsafe_allow_html=True,
    )

    feature_defaults = artifacts["feature_defaults"]
    feature_ranges = artifacts["feature_ranges"]

    def _bounded_int(value: int, lower: int, upper: int) -> int:
        return int(min(max(value, lower), upper))

    def _bounded_float(value: float, lower: float, upper: float) -> float:
        return float(min(max(value, lower), upper))

    house_min, house_max = feature_ranges["house_size"]
    lot_min, lot_max = feature_ranges["acre_lot"]
    bath_min, bath_max = feature_ranges["bath"]
    bed_min, bed_max = feature_ranges["bed"]
    city_min, city_max = feature_ranges["city_te"]

    if "city_price_mode" not in st.session_state:
        st.session_state["city_price_mode"] = "dropdown"
    if "predicted_home_value" not in st.session_state:
        st.session_state["predicted_home_value"] = None
    if "predicted_input_row" not in st.session_state:
        st.session_state["predicted_input_row"] = None

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
            background-color: #70b8d1;
            color: #412d1b;
            border: 1px solid #70b8d1;
            padding: 0.7rem 1.6rem;
            border-radius: 0.45rem;
            font-size: 1.05rem;
            font-weight: 600;
            min-width: 220px;
            display: block;
            margin: 0 auto;
        }
        div[data-testid="stButton"] > button:hover {
            background-color: #63a9bf;
            color: #412d1b;
            border-color: #63a9bf;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    form_left, form_center, form_right = st.columns([0.7, 2.6, 0.7])
    with form_center:
        city_list = sorted(artifacts["city_te_map"].index.tolist())
        selected_city = st.selectbox("City", city_list)
        city_default = float(
            artifacts["city_te_map"].get(selected_city, artifacts["global_price_mean"])
        )
        selected_city_rows = int(artifacts["city_count_map"].get(selected_city, 0))

        st.caption(f"{selected_city_rows:,} listing row(s) used for this city's average.")

        manual_city_value = st.toggle(
            "Enter exact city value",
            value=st.session_state["city_price_mode"] == "manual",
        )
        st.session_state["city_price_mode"] = "manual" if manual_city_value else "dropdown"

        if st.session_state["city_price_mode"] == "dropdown":
            st.number_input(
                FEATURE_LABELS["city_te"],
                value=city_default,
                step=1000.0,
                format="%.2f",
                disabled=True,
                help="This value is automatically filled from the selected city.",
            )
            city_te_value = city_default
        else:
            city_te_value = st.number_input(
                FEATURE_LABELS["city_te"],
                min_value=float(max(0.0, city_min)),
                max_value=float(max(city_max, max(0.0, city_min) + 1.0)),
                value=_bounded_float(
                    feature_defaults["city_te"],
                    float(max(0.0, city_min)),
                    float(max(city_max, max(0.0, city_min) + 1.0)),
                ),
                step=1000.0,
                format="%.2f",
                help="Type an exact city average value when you want to override the dropdown.",
            )

        house_size = st.number_input(
            FEATURE_LABELS["house_size"],
            min_value=int(max(100, np.floor(house_min))),
            max_value=int(np.ceil(house_max)),
            value=_bounded_int(
                feature_defaults["house_size"],
                int(max(100, np.floor(house_min))),
                int(np.ceil(house_max)),
            ),
            step=1,
        )

        acre_lot = st.number_input(
            FEATURE_LABELS["acre_lot"],
            min_value=float(max(0.0, lot_min)),
            max_value=float(max(lot_max, max(0.0, lot_min) + 0.01)),
            value=_bounded_float(
                feature_defaults["acre_lot"],
                float(max(0.0, lot_min)),
                float(max(lot_max, max(0.0, lot_min) + 0.01)),
            ),
            step=0.01,
            format="%.2f",
        )

        bath = st.number_input(
            FEATURE_LABELS["bath"],
            min_value=int(max(1, np.floor(bath_min))),
            max_value=int(max(1, np.ceil(bath_max))),
            value=_bounded_int(
                feature_defaults["bath"],
                int(max(1, np.floor(bath_min))),
                int(max(1, np.ceil(bath_max))),
            ),
            step=1,
        )

        bed = st.number_input(
            FEATURE_LABELS["bed"],
            min_value=int(max(1, np.floor(bed_min))),
            max_value=int(max(1, np.ceil(bed_max))),
            value=_bounded_int(
                feature_defaults["bed"],
                int(max(1, np.floor(bed_min))),
                int(max(1, np.ceil(bed_max))),
            ),
            step=1,
        )

    button_left, button_center, button_right = st.columns([0.7, 2.6, 0.7])
    with button_center:
        predict_clicked = st.button("**Predict**", use_container_width=True)

    input_row = {
        "city_te": float(city_te_value),
        "house_size": float(house_size),
        "acre_lot": float(acre_lot),
        "bath": float(bath),
        "bed": float(bed),
    }

    if predict_clicked:
        st.session_state["predicted_input_row"] = input_row.copy()
        st.session_state["predicted_home_value"] = float(
            artifacts["model"].predict(pd.DataFrame([input_row]))[0]
        )

    if st.session_state["predicted_home_value"] is not None:
        pred_left, pred_center, pred_right = st.columns([0.7, 2.6, 0.7])
        with pred_center:
            pred_price = st.session_state["predicted_home_value"]
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.15rem; font-weight: 600; color: #4b5563; margin-bottom: 0.25rem;">
                        Estimated Home Value
                    </div>
                    <div style="font-size: 4rem; font-weight: 800; line-height: 1.05; color: #111827;">
                        ${pred_price:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chart_input_row = st.session_state["predicted_input_row"] or input_row
            feature_option_map = {
                FEATURE_LABELS["house_size"]: "house_size",
                FEATURE_LABELS["acre_lot"]: "acre_lot",
                FEATURE_LABELS["bath"]: "bath",
                FEATURE_LABELS["bed"]: "bed",
                FEATURE_LABELS["city_te"]: "city_te",
            }
            selected_chart_feature_label = st.selectbox(
                "Estimated value trend by",
                list(feature_option_map.keys()),
                index=0,
            )
            selected_chart_feature = feature_option_map[selected_chart_feature_label]

            chart_min, chart_max = feature_ranges[selected_chart_feature]
            if selected_chart_feature in {"bath", "bed"}:
                x_values = np.arange(int(np.floor(chart_min)), int(np.ceil(chart_max)) + 1)
                x_values = x_values[x_values >= 1]
                if len(x_values) == 0:
                    x_values = np.array([1])
            else:
                x_values = np.linspace(chart_min, chart_max, 80)

            chart_df = pd.DataFrame([chart_input_row] * len(x_values))
            chart_df[selected_chart_feature] = x_values
            y_values = artifacts["model"].predict(chart_df)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(x_values, y_values, linewidth=2.2, color="#70b8d1")
            ax.scatter(
                [chart_input_row[selected_chart_feature]],
                [pred_price],
                color="#412d1b",
                s=60,
                zorder=3,
            )
            ax.set_title("Estimated Value Trend")
            ax.set_xlabel(selected_chart_feature_label)
            ax.set_ylabel("Estimated Home Value")
            ax.grid(alpha=0.25)
            st.pyplot(fig)


    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.75rem; margin-bottom: 1rem; max-width: 960px; margin-left: auto; margin-right: auto;">
            <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.35rem;">Acknowledgements</div>
            <div style="font-size: 0.95rem;">
                This project was created
                as a demo for demonstating real estate prediction techniques to prospective employers, and is not officially affiliated with <a href="https://www.hamptonsrealestate.com/eng" target="_blank" rel="noopener noreferrer">Saunders & Associates</a>.
                The model is trained on publicly available real estate listing data and is intended for demo/educational use only.
                Predictions may not reflect current market conditions and should not be used for financial decisions.
                Data was collected from <a href="https://www.realtor.com/" target="_blank" rel="noopener noreferrer">Realtor.com</a>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_model_details(artifacts: dict) -> None:
    st.title("Model Details")

    if not artifacts["xgb_available"]:
        st.info(
            "XGBoost is unavailable on this machine, so the app is using a backup model."
        )
        if artifacts["xgb_import_error"]:
            with st.expander("Technical details"):
                st.text(artifacts["xgb_import_error"])

    st.subheader("Model Summary")
    metric_rows = []
    for split_name in ["fit", "validation", "train", "test"]:
        row = {"split": split_name}
        row.update(artifacts["metrics"][split_name])
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Generalization Check")
    st.write(f"Validation/Fit RMSE ratio: {artifacts['overfit_ratio_val_fit']:.3f}")
    st.write(f"Test/Fit RMSE ratio: {artifacts['overfit_ratio_test_fit']:.3f}")
    st.write(f"Interpretation: {artifacts['fit_label']}")

    residuals = artifacts["residuals"]
    test_pred = artifacts["test_pred"]

    st.subheader("Residual Diagnostics")
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


# App bootstrap
render_global_styles()
render_banner()

df = load_data()
artifacts = build_artifacts(df)

pages = {
    "Home Value Estimator": page_estimator,
    "Technical Specifications": page_model_details,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))
if selected_page == "Home Value Estimator":
    page_estimator(df, artifacts)
else:
    page_model_details(artifacts)
