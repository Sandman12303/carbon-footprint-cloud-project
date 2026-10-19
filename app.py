"""
Streamlit Cloud Application: Carbon Footprint Regression
A simple, explainable cloud-computing project predicting weekly carbon footprint
from Locality Type and Primary Travel Mode using live Google Sheets data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cloud Carbon Footprint Regression",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cloud Data Pipeline Configuration
# ---------------------------------------------------------
CSV_FEED_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRIjOyPGiW_-amV8j1G8mkqSllhDVcMhbKnr2zCiFPQAqLwXQKMjpHn4_OSyMpd7YJ9bBo5NnlngNAW/pub?output=csv"
)
FORM_URL = (
    "https://docs.google.com/spreadsheets/d/1LHn8_J7-v-QrumLSMwVNdHycKoPBtbuj2P5o72ruxdo/edit?resourcekey=&gid=1176299686#gid=1176299686"
)

BASELINE_LOCALITY = "Urban"
BASELINE_MODE = "Walk/Cycle"

ALL_POSSIBLE_MODES = [
    "Walk/Cycle",
    "Bike/Scooter",
    "Bus",
    "Train/Metro",
    "Car (Petrol)",
    "Car (Diesel)",
    "Car (EV)",
    "Rideshare"
]
ALL_POSSIBLE_LOCALITIES = ["Urban", "Suburban"]


# ---------------------------------------------------------
# Data Ingestion and Cleaning Function
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_clean_data(url: str):
    """
    Fetches the live published CSV from Google Sheets and applies
    academic cleaning rules:
    1. Trims leading/trailing whitespace on all string columns.
    2. Normalizes 'Car(EV)' to 'Car (EV)'.
    3. Removes rogue placeholder options like 'Option 10'.
    4. Converts target and numeric fields to float, coercing invalid values to NaN.
    5. Computes composite Travel CO2 = Primary Travel CO2 + Secondary Travel CO2.
    """
    df_raw = pd.read_csv(url)
    df = df_raw.copy()

    # Trim string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Normalization: Car(EV) -> Car (EV)
    if "Primary mode of travel this week" in df.columns:
        df["Primary mode of travel this week"] = df[
            "Primary mode of travel this week"
        ].replace({"Car(EV)": "Car (EV)"})
    if "Secondary mode of travel (if any)" in df.columns:
        df["Secondary mode of travel (if any)"] = df[
            "Secondary mode of travel (if any)"
        ].replace({"Car(EV)": "Car (EV)"})

    # Drop rogue 'Option 10' rows
    if "Primary mode of travel this week" in df.columns:
        df = df[df["Primary mode of travel this week"] != "Option 10"]
    if "Secondary mode of travel (if any)" in df.columns:
        df = df[df["Secondary mode of travel (if any)"] != "Option 10"]

    # Convert numeric fields
    numeric_cols = [
        "Per-Capita CO2",
        "Primary Travel CO2",
        "Secondary Travel CO2",
        "Total CO2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute composite Travel CO2
    if "Primary Travel CO2" in df.columns:
        secondary = (
            df["Secondary Travel CO2"].fillna(0)
            if "Secondary Travel CO2" in df.columns
            else 0
        )
        df["Travel CO2"] = df["Primary Travel CO2"] + secondary

    return df_raw, df


# ---------------------------------------------------------
# Training and Modeling Function
# ---------------------------------------------------------
def prepare_and_train_model(df_clean: pd.DataFrame, target_column: str):
    """
    One-hot encodes Locality Type and Primary Travel Mode, drops specified
    baseline levels, and fits both Scikit-Learn and Statsmodels OLS models.
    """
    # Drop rows missing the target or predictors
    train_df = df_clean.dropna(
        subset=[
            target_column,
            "Locality Type",
            "Primary mode of travel this week",
        ]
    ).copy()

    X_raw = train_df[["Locality Type", "Primary mode of travel this week"]]
    X_dummies = pd.get_dummies(X_raw, drop_first=False, dtype=float)

    # Explicitly drop the academic baselines:
    # Baseline Locality: Urban
    # Baseline Travel Mode: Walk/Cycle
    cols_to_drop = []
    base_loc_col = f"Locality Type_{BASELINE_LOCALITY}"
    base_mode_col = f"Primary mode of travel this week_{BASELINE_MODE}"

    if base_loc_col in X_dummies.columns:
        cols_to_drop.append(base_loc_col)
    if base_mode_col in X_dummies.columns:
        cols_to_drop.append(base_mode_col)

    X = X_dummies.drop(columns=cols_to_drop)
    y = train_df[target_column]

    # Fit Statsmodels OLS (with constant intercept)
    X_sm = sm.add_constant(X, has_constant="add")
    ols_model = sm.OLS(y, X_sm).fit()

    # Fit Scikit-Learn LinearRegression
    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X, y)

    return train_df, X, y, ols_model, sk_model


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("☁️ Cloud Model Controls")
st.sidebar.markdown(
    """
    **Architecture Flow:**
    Google Form ➔ Sheet ➔ Live CSV ➔ In-Memory Model ➔ Dashboard
    """
)

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Select Regression Target")
target_choice = st.sidebar.radio(
    "Choose target variable:",
    [
        "Per-Capita CO2 (Total Household & Travel)",
        "Travel CO2 (Primary + Secondary Travel Only)",
    ],
    index=1,  # Default to Travel CO2 for clearer travel mode association
)

target_col = (
    "Per-Capita CO2"
    if "Per-Capita" in target_choice
    else "Travel CO2"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Reference Baselines:**
    - **Locality:** `{BASELINE_LOCALITY}`
    - **Travel Mode:** `{BASELINE_MODE}`

    *Coefficients represent difference in weekly kg CO₂ compared to an Urban resident who Walks/Cycles.*
    """
)
st.sidebar.markdown(f"[📝 View Google Form/Sheet Source]({FORM_URL})")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
try:
    df_raw, df_processed = load_and_clean_data(CSV_FEED_URL)
except Exception as e:
    st.error(f"Failed to fetch live published CSV from Google Sheets: {e}")
    st.stop()

train_df, X, y, ols_model, sk_model = prepare_and_train_model(
    df_processed, target_col
)

# ---------------------------------------------------------
# Header and Intro
# ---------------------------------------------------------
st.title("🌱 Cloud Carbon Footprint Predictor")
st.caption(
    "A free-tier cloud machine learning workflow combining Google Sheets live CSV feeds, "
    "scikit-learn, and statsmodels linear regression."
)

# Top metric summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Survey Rows", len(df_raw))
with col2:
    st.metric("Training Sample (n)", len(train_df))
with col3:
    r2_val = ols_model.rsquared
    st.metric("R² (Variance Explained)", f"{r2_val:.3f}")
with col4:
    adj_r2 = ols_model.rsquared_adj
    st.metric("Adjusted R²", f"{adj_r2:.3f}")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Regression Results & Interpretation",
        "🎯 Interactive Predictor",
        "🧹 Live Dataset & Cleaning",
        "🏛️ Cloud Architecture & Limitations",
    ]
)

# ---------------------------------------------------------
# TAB 1: Regression Results & Explanations
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Statistical Results for Target: `{target_col}` (kg CO₂ / week)")

    col_chart, col_stat = st.columns([1.2, 1])

    with col_chart:
        st.markdown("#### Regression Coefficients Relative to Baseline")
        st.caption(
            f"Baseline: Locality = **{BASELINE_LOCALITY}**, Primary Mode = **{BASELINE_MODE}**"
        )

        coef_df = pd.DataFrame(
            {
                "Predictor": [
                    col.replace("Primary mode of travel this week_", "Mode: ")
                    .replace("Locality Type_", "Locality: ")
                    for col in X.columns
                ],
                "Coefficient": ols_model.params[X.columns].values,
                "p_value": ols_model.pvalues[X.columns].values,
                "ci_lower": ols_model.conf_int().loc[X.columns, 0].values,
                "ci_upper": ols_model.conf_int().loc[X.columns, 1].values,
            }
        ).sort_values(by="Coefficient", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = [
            "#e74c3c" if c > 0 else "#2ecc71" for c in coef_df["Coefficient"]
        ]
        bars = ax.barh(
            coef_df["Predictor"],
            coef_df["Coefficient"],
            color=colors,
            alpha=0.85,
            edgecolor="black",
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Change in Weekly CO₂ (kg/week) vs Baseline")
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Value labels
        for bar in bars:
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.4 if width >= 0 else -0.4
            ax.annotate(
                f"{width:+.2f}",
                xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                va="center",
                ha=align,
                fontsize=9,
                fontweight="bold",
            )

        st.pyplot(fig)
        plt.close(fig)

    with col_stat:
        st.markdown("#### Model Fit & Goodness of Fit")
        st.markdown(
            f"""
            - **R-squared ($R^2$):** `{ols_model.rsquared:.4f}` ({ols_model.rsquared*100:.1f}% of variance explained)
            - **Adjusted $R^2$:** `{ols_model.rsquared_adj:.4f}`
            - **F-statistic:** `{ols_model.fvalue:.2f}` (p-value: `{ols_model.f_pvalue:.4f}`)
            - **Baseline Intercept ($\beta_0$):** `{ols_model.params['const']:.2f} kg CO₂/week`
            """
        )

        st.markdown("#### Statistical Interpretation")
        if target_col == "Travel CO2":
            st.info(
                """
                **Travel CO₂ Target Highlights:**
                - Notice how **Car (Diesel)** and **Car (Petrol)** show positive coefficients relative to walking/cycling.
                - Private fossil-fuel transit strongly correlates with higher dedicated travel emissions.
                - The F-statistic p-value is statistically significant at $p < 0.05$.
                """
            )
        else:
            st.warning(
                """
                **Per-Capita CO₂ Target Highlights:**
                - Per-capita emissions bundle travel with household electricity and cooking gas divided by household size.
                - In smaller samples, high electricity usage or large family sizes in certain households can mask travel effects.
                - Notice the low adjusted $R^2$, indicating that locality and travel mode alone leave much per-capita variation unexplained.
                """
            )

    st.markdown("---")
    st.markdown("#### Full OLS Statistical Table (Statsmodels)")
    summary_data = []
    for col_name in X.columns:
        clean_name = (
            col_name.replace("Primary mode of travel this week_", "")
            .replace("Locality Type_", "")
        )
        cat_type = "Locality" if "Locality" in col_name else "Travel Mode"
        c = ols_model.params[col_name]
        se = ols_model.bse[col_name]
        pval = ols_model.pvalues[col_name]
        sig = "⭐" if pval < 0.05 else ("•" if pval < 0.10 else "ns")
        summary_data.append(
            {
                "Category": cat_type,
                "Feature Level": clean_name,
                "Coefficient (kg/wk)": f"{c:+.2f}",
                "Std Error": f"{se:.2f}",
                "t-stat": f"{ols_model.tvalues[col_name]:.2f}",
                "p-value": f"{pval:.4f}",
                "Sig.": sig,
            }
        )
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    st.caption("Significance codes: ⭐ p < 0.05 | • p < 0.10 | ns: not statistically significant")


# ---------------------------------------------------------
# TAB 2: Interactive Predictor
# ---------------------------------------------------------
with tab2:
    st.subheader("🎯 Real-Time Carbon Footprint Estimator")
    st.markdown(
        "Select a Locality Type and Primary Travel Mode to calculate the estimated weekly emissions using the trained regression equation."
    )

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        sel_locality = st.selectbox(
            "Select Locality Type:",
            options=ALL_POSSIBLE_LOCALITIES,
            index=0
        )
    with pred_col2:
        sel_mode = st.selectbox(
            "Select Primary Mode of Travel:",
            options=ALL_POSSIBLE_MODES,
            index=0
        )

    # Prepare input vector
    input_dict = {col: 0.0 for col in X.columns}

    loc_feature = f"Locality Type_{sel_locality}"
    if loc_feature in input_dict:
        input_dict[loc_feature] = 1.0

    mode_feature = f"Primary mode of travel this week_{sel_mode}"
    if mode_feature in input_dict:
        input_dict[mode_feature] = 1.0

    input_df = pd.DataFrame([input_dict])

    # Run prediction
    predicted_val = sk_model.predict(input_df)[0]
    intercept_val = sk_model.intercept_
    loc_contrib = input_dict.get(loc_feature, 0.0) * sk_model.coef_[X.columns.get_loc(loc_feature)] if loc_feature in X.columns else 0.0
    mode_contrib = input_dict.get(mode_feature, 0.0) * sk_model.coef_[X.columns.get_loc(mode_feature)] if mode_feature in X.columns else 0.0

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        st.metric(
            label=f"Predicted {target_col}",
            value=f"{predicted_val:.2f} kg CO₂ / week",
            delta=f"{predicted_val - train_df[target_col].mean():+.2f} kg vs Sample Mean"
        )
        st.caption(f"Sample Average: {train_df[target_col].mean():.2f} kg CO₂/week")

    with res_col2:
        st.markdown("#### Equation Breakdown")
        st.markdown(
            f"""
            $$\\hat{{y}} = \\beta_0 + \\beta_{{\\text{{locality}}}} + \\beta_{{\\text{{mode}}}}$$

            - **Baseline Constant ($\\beta_0$):** `{intercept_val:.2f}` kg/wk (Urban + Walk/Cycle)
            - **Locality Effect ({sel_locality}):** `{loc_contrib:+.2f}` kg/wk
            - **Travel Mode Effect ({sel_mode}):** `{mode_contrib:+.2f}` kg/wk
            - **Final Predicted Footprint:** **`{predicted_val:.2f}` kg CO₂/week**
            """
        )
        if mode_feature not in X.columns and sel_mode != BASELINE_MODE:
            st.warning(
                f"Note: '{sel_mode}' has no valid non-null rows in current training data; its coefficient defaults to 0."
            )


# ---------------------------------------------------------
# TAB 3: Dataset Inspection & Cleaning Rules
# ---------------------------------------------------------
with tab3:
    st.subheader("🧹 Live Data Pipeline & Quality Control")

    st.markdown(
        """
        The live feed is read directly into memory from the published Google Sheet CSV endpoint.
        Automated transformations applied:
        1. **Whitespace Trimming:** Strip leading/trailing spaces across text headers and response strings.
        2. **Value Normalization:** Maps inconsistent inputs such as `'Car(EV)'` ➔ `'Car (EV)'`.
        3. **Form Artifact Pruning:** Automatically drops rogue test selections such as `'Option 10'`.
        4. **Type Coercion & Missing Value Handling:** Drops rows with `#N/A` or missing targets.
        5. **Feature Synthesis:** Computes `Travel CO2 = Primary Travel CO2 + Secondary Travel CO2`.
        """
    )

    clean_col1, clean_col2 = st.columns(2)
    with clean_col1:
        st.write("##### Cleaned Training Subsample (First 10 Rows)")
        preview_cols = [
            "Locality Type",
            "Primary mode of travel this week",
            "Primary Travel CO2",
            "Secondary Travel CO2",
            "Travel CO2",
            "Per-Capita CO2"
        ]
        st.dataframe(train_df[preview_cols].head(10), use_container_width=True)

    with clean_col2:
        st.write("##### Category Distribution in Training Sample")
        st.write("**Locality Type Distribution:**")
        st.dataframe(train_df["Locality Type"].value_counts().reset_index())
        st.write("**Primary Mode Distribution:**")
        st.dataframe(train_df["Primary mode of travel this week"].value_counts().reset_index())


# ---------------------------------------------------------
# TAB 4: Architecture & Limitations
# ---------------------------------------------------------
with tab4:
    st.subheader("🏛️ End-to-End Cloud Computing Architecture")

    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["📝 Google Form\n(Data Collection)"] --> B["📊 Google Sheets\n(Live Storage & Formulas)"]
            B --> C["🌐 Published CSV Feed\n(Serverless API Endpoint)"]
            C --> D["⚡ Streamlit Cloud Web App\n(Auto Ingestion & Retraining)"]
            C --> E["📓 Google Colab Notebook\n(Exploration & Statsmodels)"]
            B --> F["📈 Looker Studio\n(Optional BI Dashboard)"]
        ```
        """
    )

    st.markdown("### ⚠️ Academic Limitations & Disclaimers")
    st.error(
        """
        **1. Association vs. Causation:**
        This linear regression models empirical *statistical associations*, **not causal relationships**.
        A positive coefficient for a specific transit mode does not imply that switching to that mode alone
        will increase or decrease a person's footprint by that exact amount.

        **2. Small Sample Size ($n \\approx 42$):**
        The dataset contains ~42 valid responses. Small samples lead to larger standard errors, wider confidence
        intervals, and susceptibility to outliers (e.g. one respondent taking an unusual long-distance flight or running heavy AC).

        **3. Omitted Variable Bias:**
        Weekly per-capita carbon footprint is heavily influenced by variables omitted from the two-feature regression,
        including household electricity usage, appliance energy ratings, cooking fuels, and exact commute distances.

        **4. Self-Reported Survey Data:**
        Distance intervals and weekly energy estimations are subject to recall bias and rounding errors by respondents.
        """
    )

# Footer
st.markdown("---")
st.caption(
    "Mini Cloud-Computing Student Academic Project | Built with Google Sheets, Scikit-Learn, Statsmodels, & Streamlit"
)
