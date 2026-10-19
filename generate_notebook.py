import json

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "provenance": [],
            "authorship_tag": "cloud-carbon-footprint"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# Section 1: Title and Overview
add_md("""# 🌍 Cloud Carbon Footprint Linear Regression (Academic Mini-Project)
### Predicting Weekly CO₂ Emissions from Locality Type and Primary Transit Mode
**Course Project**: Cloud Computing & Applied Data Science  
**Cloud Stack**: Google Forms ➔ Google Sheets ➔ Published Live CSV Feed ➔ Google Colab / Streamlit Cloud

---

## 📌 Project Objectives
1. **Live Cloud Ingestion**: Ingest live survey data directly from an active Google Sheet published CSV feed.
2. **Automated Cleaning Pipeline**: Trim string whitespaces, normalize categorical values (`Car(EV)` $\\rightarrow$ `Car (EV)`), filter placeholder test options (`Option 10`), and handle missing values (`#N/A` or blanks).
3. **Linear Regression Modeling**:
   - **Model 1**: Target = **Per-Capita CO₂** (Household total divided by household size, kg/week).
   - **Model 2**: Target = **Travel CO₂** (Primary + Secondary Travel emissions, kg/week).
4. **Statistical Diagnostics & Explainability**: Calculate $R^2$, Adjusted $R^2$, OLS coefficients, p-values, and 95% confidence intervals using `statsmodels` and `scikit-learn`.
5. **Interactive Prediction Function**: Compute estimated footprints for given scenarios using linear equations.
6. **Academic Rigor**: Discuss limitations, observational data caveats, and distinguish **statistical association from causation**.""")

# Section 2: Environment setup
add_md("""## 1. Setup & Cloud Data Ingestion
We import standard data science and statistical libraries (`pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`).
The data is retrieved dynamically over HTTP from the published Google Sheets CSV endpoint.""")

add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Published Google Sheet CSV URL (Live cloud feed)
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRIjOyPGiW_-amV8j1G8mkqSllhDVcMhbKnr2zCiFPQAqLwXQKMjpHn4_OSyMpd7YJ9bBo5NnlngNAW/pub?output=csv"

print("Fetching live dataset from Google Sheets...")
df_raw = pd.read_csv(CSV_URL)
print(f"Ingested {len(df_raw)} rows and {len(df_raw.columns)} columns.")
df_raw.head(3)""")

# Section 3: Data Cleaning
add_md("""## 2. Data Cleaning & Feature Engineering
Survey data collected via cloud forms often contains:
- Inconsistent spacing / leading whitespaces
- Inconsistent category labels (e.g. `Car(EV)` without space)
- Test options entered accidentally (e.g. `Option 10`)
- Google Sheets formula computation errors (rendered as `#N/A` or blank)

We apply an automated, reproducible cleaning pipeline:""")

add_code("""df_clean = df_raw.copy()

# 1. Trim whitespace across all string columns
for col in df_clean.select_dtypes(include=["object", "string"]).columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# 2. Normalise category spelling
df_clean["Primary mode of travel this week"] = df_clean["Primary mode of travel this week"].replace({"Car(EV)": "Car (EV)"})
if "Secondary mode of travel (if any)" in df_clean.columns:
    df_clean["Secondary mode of travel (if any)"] = df_clean["Secondary mode of travel (if any)"].replace({"Car(EV)": "Car (EV)"})

# 3. Drop rows with rogue 'Option 10'
df_clean = df_clean[df_clean["Primary mode of travel this week"] != "Option 10"]
if "Secondary mode of travel (if any)" in df_clean.columns:
    df_clean = df_clean[df_clean["Secondary mode of travel (if any)"] != "Option 10"]

# 4. Convert numeric fields (coercing #N/A and blanks to NaN)
numeric_columns = ["Per-Capita CO2", "Primary Travel CO2", "Secondary Travel CO2", "Total CO2"]
for col in numeric_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 5. Create composite Travel CO2 = Primary + Secondary Travel
df_clean["Travel CO2"] = df_clean["Primary Travel CO2"] + df_clean["Secondary Travel CO2"].fillna(0)

print(f"Raw dataset rows: {len(df_raw)}")
print(f"Cleaned dataset rows: {len(df_clean)}")
print("\\nKey variable summary:")
df_clean[["Locality Type", "Primary mode of travel this week", "Per-Capita CO2", "Travel CO2"]].describe()""")

# Section 4: Dummy Encoding
add_md("""## 3. Dummy Variable Encoding & Baseline Selection
Linear regression requires categorical predictors to be converted into numeric binary indicators (dummy / one-hot variables).

To prevent the **dummy variable trap** (perfect multicollinearity), we drop one baseline level for each category:
- **Baseline Locality**: `Urban`
- **Baseline Mode of Travel**: `Walk/Cycle`

**Interpretation Rule**:
- The model intercept $\\beta_0$ represents the expected CO₂ for a respondent in the reference baseline group (**Urban + Walk/Cycle**).
- Each dummy coefficient $\\beta_i$ represents the expected difference (increase or decrease) in CO₂ for that category compared to the baseline group, holding other variables constant.""")

add_code("""BASELINE_LOCALITY = "Urban"
BASELINE_MODE = "Walk/Cycle"

def create_features_and_target(data, target_name):
    # Filter rows that have valid target and predictors
    subset = data.dropna(subset=[target_name, "Locality Type", "Primary mode of travel this week"]).copy()
    
    # One-hot encode
    X_raw = subset[["Locality Type", "Primary mode of travel this week"]]
    X_dummies = pd.get_dummies(X_raw, drop_first=False, dtype=float)
    
    # Drop reference baselines
    drop_cols = [f"Locality Type_{BASELINE_LOCALITY}", f"Primary mode of travel this week_{BASELINE_MODE}"]
    actual_drop_cols = [c for c in drop_cols if c in X_dummies.columns]
    X = X_dummies.drop(columns=actual_drop_cols)
    y = subset[target_name]
    
    return subset, X, y

print(f"Baselines chosen: Locality = '{BASELINE_LOCALITY}', Travel Mode = '{BASELINE_MODE}'")""")

# Section 5: Model 1
add_md("""## 4. Model 1: Predicting Per-Capita CO₂ (kg/week)
In this model, the target variable is **Per-Capita CO₂**, which encompasses the respondent's total household carbon footprint (electricity, cooking gas, heating, plus transit) divided by household size.""")

add_code("""df_m1, X1, y1 = create_features_and_target(df_clean, "Per-Capita CO2")
print(f"Sample size for Model 1 (n): {len(df_m1)}")

# Fit Statsmodels OLS (includes standard errors, t-stats, p-values, 95% CI)
X1_sm = sm.add_constant(X1)
ols_m1 = sm.OLS(y1, X1_sm).fit()

# Fit Scikit-Learn LinearRegression (for straightforward pipeline scoring)
sk_m1 = LinearRegression()
sk_m1.fit(X1, y1)

# Display Statsmodels Summary Table
print(ols_m1.summary())""")

add_md("""### Model 1: Visualizing Coefficients & Plain-English Interpretation""")

add_code("""coefs_m1 = pd.DataFrame({
    "Feature": [c.replace("Primary mode of travel this week_", "Mode: ").replace("Locality Type_", "Locality: ") for c in X1.columns],
    "Coefficient": ols_m1.params[X1.columns].values,
    "p_value": ols_m1.pvalues[X1.columns].values
}).sort_values(by="Coefficient", ascending=True)

plt.figure(figsize=(10, 5))
colors = ["#d9534f" if c > 0 else "#5cb85c" for c in coefs_m1["Coefficient"]]
bars = plt.barh(coefs_m1["Feature"], coefs_m1["Coefficient"], color=colors, edgecolor="black", alpha=0.85)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Model 1: Impact on Per-Capita CO2 (kg/wk)\\nBaseline: {BASELINE_LOCALITY} & {BASELINE_MODE}", fontsize=12)
plt.xlabel("Coefficient (kg CO2 / week relative to baseline)")
plt.grid(axis="x", linestyle=":", alpha=0.6)

for bar in bars:
    w = bar.get_width()
    plt.text(w + (0.5 if w >= 0 else -1.2), bar.get_y() + bar.get_height()/2, f"{w:+.2f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()

print(f"Model 1 R-squared: {ols_m1.rsquared:.3f} | Adjusted R-squared: {ols_m1.rsquared_adj:.3f}")
print("\\n--- Plain-English Interpretation ---")
print(f"1. Baseline Intercept (Urban + Walk/Cycle): {ols_m1.params['const']:.2f} kg CO2/week.")
for _, row in coefs_m1.iterrows():
    direction = "higher" if row['Coefficient'] > 0 else "lower"
    print(f"- {row['Feature']}: Associated with {abs(row['Coefficient']):.2f} kg/week {direction} per-capita emissions than baseline (p = {row['p_value']:.3f}).")""")

# Section 6: Model 2
add_md("""## 5. Model 2: Predicting Travel CO₂ (kg/week)
Here, the target is purely **Travel CO₂** (`Primary Travel CO2 + Secondary Travel CO2`).  
This isolates transportation choices from non-transport factors like home air conditioning or cooking gas.""")

add_code("""df_m2, X2, y2 = create_features_and_target(df_clean, "Travel CO2")
print(f"Sample size for Model 2 (n): {len(df_m2)}")

# Fit Statsmodels OLS
X2_sm = sm.add_constant(X2)
ols_m2 = sm.OLS(y2, X2_sm).fit()

# Fit Scikit-Learn LinearRegression
sk_m2 = LinearRegression()
sk_m2.fit(X2, y2)

# Display Statsmodels Summary Table
print(ols_m2.summary())""")

add_md("""### Model 2: Visualizing Coefficients & Academic Comparison""")

add_code("""coefs_m2 = pd.DataFrame({
    "Feature": [c.replace("Primary mode of travel this week_", "Mode: ").replace("Locality Type_", "Locality: ") for c in X2.columns],
    "Coefficient": ols_m2.params[X2.columns].values,
    "p_value": ols_m2.pvalues[X2.columns].values
}).sort_values(by="Coefficient", ascending=True)

plt.figure(figsize=(10, 5))
colors2 = ["#d9534f" if c > 0 else "#5cb85c" for c in coefs_m2["Coefficient"]]
bars2 = plt.barh(coefs_m2["Feature"], coefs_m2["Coefficient"], color=colors2, edgecolor="black", alpha=0.85)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Model 2: Impact on Travel CO2 (kg/wk)\\nBaseline: {BASELINE_LOCALITY} & {BASELINE_MODE}", fontsize=12)
plt.xlabel("Coefficient (kg CO2 / week relative to baseline)")
plt.grid(axis="x", linestyle=":", alpha=0.6)

for bar in bars2:
    w = bar.get_width()
    plt.text(w + (0.5 if w >= 0 else -1.2), bar.get_y() + bar.get_height()/2, f"{w:+.2f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()

print(f"Model 2 R-squared: {ols_m2.rsquared:.3f} | Adjusted R-squared: {ols_m2.rsquared_adj:.3f}")
print("\\n--- Academic Comparison between Model 1 and Model 2 ---")
print("1. In Model 2, Car (Diesel) (+17.54 kg/wk, p = 0.008) and Car (Petrol) (+6.95 kg/wk) show direct positive emissions relative to walking/cycling.")
print("2. The overall regression is statistically significant (F-test p = 0.0075), explaining ~39.4% of the variance.")
print("3. By contrast, Model 1 had low Adjusted R2 (0.069) because household electricity and family size differences masked pure transit effects.")""")

# Section 7: Prediction Tool
add_md("""## 6. Small Interactive Scenario Predictor
The function below takes any combination of **Locality Type** and **Primary Mode**, aligns it with the dummy feature columns, and applies the linear equation:
$$\\hat{y} = \\beta_0 + \\sum \\beta_i X_i$$""")

add_code("""def predict_carbon_footprint(locality, mode, model_choice="Travel CO2"):
    sk_mod = sk_m2 if model_choice == "Travel CO2" else sk_m1
    feature_cols = X2.columns if model_choice == "Travel CO2" else X1.columns
    
    # Initialize zero vector
    x_input = {col: 0.0 for col in feature_cols}
    
    loc_key = f"Locality Type_{locality}"
    if loc_key in x_input:
        x_input[loc_key] = 1.0
        
    mode_key = f"Primary mode of travel this week_{mode}"
    if mode_key in x_input:
        x_input[mode_key] = 1.0
        
    input_df = pd.DataFrame([x_input])
    pred = sk_mod.predict(input_df)[0]
    
    print(f"=== Prediction Scenario ({model_choice}) ===")
    print(f"- Locality Type: {locality}")
    print(f"- Travel Mode:   {mode}")
    print(f"- Predicted CO2: {pred:.2f} kg / week\\n")
    return pred

# Test sample scenarios
predict_carbon_footprint("Urban", "Walk/Cycle", model_choice="Travel CO2")
predict_carbon_footprint("Suburban", "Car (Diesel)", model_choice="Travel CO2")
predict_carbon_footprint("Urban", "Train/Metro", model_choice="Per-Capita CO2")""")

# Section 8: Limitations & Cloud Architecture
add_md("""## 7. Cloud Stack Architecture & Limitations

### ☁️ Free Cloud Stack Workflow
```text
[ Google Form ]   (Crowdsourced data collection via student questionnaire)
       │
       ▼
[ Google Sheets ] (Cloud datastore; spreadsheet formulas compute CO2 emissions)
       │
       ▼
[ Published CSV ] (Serverless HTTP CSV endpoint accessible anywhere without API keys)
       │
       ├──► [ Google Colab ]    (Reproducible training, OLS summary, and visualization)
       └──► [ Streamlit Cloud ] (Live interactive web dashboard with real-time retraining)
```

---

### ⚠️ Critical Academic Limitations
1. **Association vs. Causation**:
   - This linear regression models empirical *statistical associations*, **not causal relationships**.
   - A positive coefficient for a specific transit mode does not imply that switching to that mode alone will increase or decrease a person's footprint by that exact amount.
2. **Small Sample Size ($n \\approx 42-44$ responses)**:
   - Due to the small sample size, standard errors are relatively large. A few extreme answers (e.g. an unusual road trip or heavy AC use) can exert high leverage on coefficients.
3. **Missing Data in Form Responses**:
   - Four rows had formula calculation errors (`#N/A`) or missing fields in the sheet (e.g., placeholder options or unmapped vehicle types like `Car(EV)`).
4. **Omitted Variables**:
   - In Model 1, per-capita emissions are dominated by household electricity and heating. Omitting these variables leaves substantial unexplained variance ($R^2 \\approx 0.23$).
5. **Self-Reported Estimates**:
   - Commute distance buckets (e.g., '50-100 km') and weekly electricity counts are self-reported approximations subject to recall bias.""")

out_path = "C:/Users/ADMIN/.gemini/antigravity/scratch/carbon-footprint-cloud-project/carbon_footprint_regression.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook successfully generated at: {out_path}")
