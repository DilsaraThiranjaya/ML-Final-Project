# Phase 1: Exploratory Data Analysis & Preprocessing Pipeline - Technical Documentation

This document provides a line-by-line, cell-by-cell architectural breakdown of the logic implemented in the `1_eda_and_preprocessing.ipynb` notebook. It is strictly procedural to fulfill explicit guidelines and is intended for ML architects to understand the data ingestion, manipulation, and Scikit-learn pipeline engineering strategies employed for AuraCart.

---

### Cell 1: Environment Initialization & Data Ingestion
**Code Focus**: Imports and pandas `read_csv`.
**Reasoning**: 
- `pandas` and `numpy` form the foundational data structures necessary for tabular operations.
- `matplotlib.pyplot` and `seaborn` are imported to generate analytical visualizations like distribution densities and correlation heatmaps. 
- The target dataset is pulled dynamically from the HuggingFace URL avoiding static local file dependencies.
- `sns.set_theme(style="whitegrid", palette="muted")` is called to ensure that all generated graphs adhere to professional, high-contrast visual standards.

### Cell 2: Feature Dropping and Temporal Extraction
**Code Focus**: Structuring time data and dropping metadata.
**Reasoning**:
- ID columns (`order_id`, `customer_id`, `product_id`) and arbitrary text strings like addresses act as noise in deterministic predictive models. They offer zero generalized significance and exponentially expand vector dimension size, thus they are scrubbed via `drop()`.
- Time-series data is useless if left as continuous string stamps. By utilizing `pd.to_datetime()`, we access `.dt` accessor methods to surgically extract cyclical features: `order_hour` and `order_month`, giving our regression models a structural understanding of seasonality.
- We produce the critical `shipping_duration_days` numerical metric derived mathematically by subtracting order dates from shipping dates. Missing values are filled with zeroes using `fillna(0)` reflecting non-shipped pending items.

### Cell 3: Continuous Statistical Visualization
**Code Focus**: `sns.histplot` targeting Price and Quantity.
**Reasoning**:
- Continuous inputs often reflect significant real-world skews. Our histograms visualize this right-skewness specifically in the financial `price` variable. We build this plot to definitively prove the necessity of utilizing `StandardScaler` during our pipeline phase, minimizing the disruptive effect of massive financial values updating gradients inappropriately compared to singular quantities like `quantity` or `order_hour`.

### Cell 4: Multicollinearity Detection
**Code Focus**: `.select_dtypes(include=[np.number])` and `.corr(method='pearson')` overlaid as an `sns.heatmap`.
**Reasoning**:
- Models such as Multiple Linear Regression explicitly rely on features maintaining statistical independence. The Pearson correlation matrix checks the mathematical overlap between variables. Based on the output, we visually verify that none of the features break the 0.9 correlation threshold horizontally, mitigating the need for strict dimensionality reduction steps like PCA.

### Cell 5: Class Imbalance Diagnostics
**Code Focus**: `sns.countplot` on Classification objectives.
**Reasoning**:
- E-commerce targets inherently lack parity. Visually displaying `delivery_status` and `customer_segment` allows us to diagnose exact majority/minority distribution scales. For example, 'Delivered' orders vastly outweigh 'Returned'. This direct evidence mandates the eventual architectural usage of `imblearn` Synthetic Minority Over-sampling Technique (SMOTE) to penalize frequency bias across algorithm weight calculations during subsequent classification training.

### Cell 6: Scikit-learn Preprocessing Pipeline Assembly
**Code Focus**: Instantiating pipelines utilizing `ColumnTransformer`, `OrdinalEncoder`, `OneHotEncoder`, and `StandardScaler`.
**Reasoning**:
- We prevent real-world data leakage by bundling all pre-treatment logic inside a scikit-learn architectural pipeline object. Variables are isolated by type:
  - **Numeric Transformers**: Scaled logically using standard scaling to control variance widths.
  - **Nominal Transformers**: Flat categories (payment method, channel) encoded via robust one-hot mechanics, avoiding numerical implication of "value".
  - **Ordinal Transformers**: In strict adherence to project guidelines, the `Customer Segment` feature explicitly utilizes `OrdinalEncoder` explicitly instructed via custom array mapping (`['New', 'Returning', 'VIP']`) establishing an exact hierarchical progression recognized natively by linear models.

### Cell 7: Sub-System Serialization to Disk
**Code Focus**: `joblib.dump` and `.to_csv`.
**Reasoning**:
- Reproducibility relies on downstream machine learning pipelines observing the identical data signatures witnessed dynamically during EDA. The raw preprocessing architecture is exported via `joblib` into an isolated `artifacts/` folder, freezing the blueprint. The fully sanitized structural dataframe is concurrently saved locally as a CSV so future notebook clusters load from an identically cleansed temporal state without redundant preprocessing latency.
