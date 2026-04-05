# Phase 1: Exploratory Data Analysis & Preprocessing Pipeline

## Overview
This phase is the bedrock of the AuraCart MLOps lifecycle. Before we can train a model, we must transform raw, messy transactional data into a mathematically robust format that a machine learning algorithm can interpret. This notebook covers everything from initial data auditing to building a production-ready Scikit-Learn preprocessing pipeline.

### Learning Objectives
- **Data Auditing**: How to programmatically identify missing values, duplicates, and data type inconsistencies.
- **Manual Feature Engineering**: Extracting hidden signals from timestamps and geographic strings using regular expressions.
- **Scikit-Learn Pipelines**: Building a reusable `ColumnTransformer` to automate imputation, scaling, and encoding.
- **Persistence**: Saving preprocessing artifacts to ensure that future "live" data is treated exactly like our training data.

---

## Cell-by-Cell Breakdown

### Cell 1: Environment Setup & Constants
**Code Logic:**
- Imports `pandas` for data manipulation, `seaborn`/`matplotlib` for plotting, and `joblib` for model serialization.
- Defines `DROP_ID_COLS` and `DROP_TEXT_COLS`.
**Strategic Rationale:**
- We identify high-cardinality features (like unique `order_id`) and raw text (like `shipping_address`) early. These columns contain too much unique noise for simple encoders and must be either dropped or processed via feature extraction.

### Cell 2: Data Loading
**Code Logic:**
- Fetches the dataset directly from a remote HuggingFace repository using `pd.read_csv()`.
**Strategic Rationale:**
- Using a remote URL ensures reproducibility. Anyone running this notebook will always fetch the exact same version of the source data.

### Cell 3: Initial Data Audit
**Code Logic:**
- Creates a summary table showing data types, missing value counts, and uniqueness.
**Strategic Rationale:**
- A "Data Audit" is different from just looking at the data. It helps us decide our strategy: if a column has 0% missing values, we might choose a different imputation strategy than if it had 20%.

### Cell 4: Feature Extraction (The "Hidden" Signals)
**Code Logic:**
- **Time Deltas**: Calculates `shipping_delay_days` by subtracting `order_date` from `shipping_date`.
- **Cyclical Features**: Extracts month, day, and hour. Crucially, it creates an `is_weekend_order` flag.
- **Regex Extraction**: Uses a Regular Expression (`geo_pattern`) to pull `city` and `state` out of long address strings.
- **Popularity**: Calculates how many times a `product_id` appears to create a frequency-based feature.
**Strategic Rationale:**
- Raw timestamps are useless to a model. By converting them into "Days of Delay" or "Weekend Flags," we provide the model with features that likely correlate with customer behavior (e.g., VIPs might order more on weekends).

### Cell 6-9: Exploratory Data Analysis (EDA)
**Code Logic:**
- Generates KDE plots (distribution), Boxplots (outliers), and a Correlation Heatmap.
**Strategic Rationale:**
- We look for **Skewness**: If `price` is heavily skewed, our model might struggle unless we scale it.
- We look for **Correlation**: If two features are perfectly correlated, we might be providing redundant information.

### Cell 10-11: Building the Scikit-Learn Pipeline
**Code Logic:**
- Defines three distinct groups:
    1. **Numerical**: `SimpleImputer(median)` -> `StandardScaler()`.
    2. **Categorical**: `SimpleImputer(most_frequent)` -> `OneHotEncoder()`.
    3. **Ordinal**: `OrdinalEncoder` for `customer_segment` (VIP > Returning > New).
**Strategic Rationale:**
- **Why Impute?** Algorithms like `SGDRegressor` fail if they encounter a `NaN`.
- **Why Scale?** It ensures that `price` (thousands) doesn't "overpower" `quantity` (small integers) during gradient descent.
- **Why Ordinal?** It preserves the logical rank in segments which One-Hot encoding would destroy.

### Cell 12-13: Artifact Persistence
**Code Logic:**
- Saves the fitted `preprocessor` as a `.joblib` file and the `df_clean` as a `.csv`.
**Strategic Rationale:**
- This is the "MLOps" part. We don't just want a cleaned CSV; we want the *logic* used to clean it. By saving the `preprocessor`, we can ensure that when the model is deployed on the cloud (Phase 4), it processes new data using the same means and standard deviations calculated here.

---

## Technical Summary
| Library | Purpose |
| :--- | :--- |
| `pd.to_datetime` | Converts raw strings into navigable Python objects for math. |
| `ColumnTransformer` | Ensures that different preprocessing rules are applied to the correct columns simultaneously. |
| `StandardScaler` | Centers data around 0 with a standard deviation of 1. |
| `joblib` | High-performance serialization for large NumPy-based objects (our pipeline). |
