# Phase 2: Supervised Predictive Modeling & Experiment Tracking - Technical Documentation

This document explains the procedural methodology engineered inside `2_supervised_modeling.ipynb`. It translates complex mathematical choices regarding Gradient Descent, Multinomial Softmax thresholds, and MLflow pipeline bindings into plain logistical business reasoning.

---

### Cell 1 & 2: Environment Registration and MLflow Architecture
**Code Focus**: Loading libraries, initializing pandas DataFrames, and binding `mlflow.set_tracking_uri()`.
**Reasoning**:
- The notebook ingests the sanitized `ecommerce_cleaned.csv` and frozen `base_preprocessor.joblib` created dynamically during Phase 1. 
- Setting `mlflow.set_tracking_uri("../mlruns")` forces MLflow to write experiment histories directly to local storage for immediate programmatic tracking, fulfilling the explicit guideline requirement for MLOps artifact management early.

### Cell 3 & 4: Continuous Revenue Prediction via SGDRegressor
**Code Focus**: Scikit-Learn `SGDRegressor`, `ImbPipeline`, and `mlflow.start_run()`.
**Reasoning**:
- The target vector natively separates `price` for continuous predictive mapping. 
- We employ `SGDRegressor` explicitly to gain access to learning parameters (`learning_rate`, `max_iter`, `eta0`). 
- Using `mlflow.start_run()`, we explicitly capture the execution context, utilizing `log_params()` to log static parameters simulating our epochs and batch control schemas. 
- Loss functions (MSE and MAE) are calculated and logged dynamically (`log_metric()`). The notebook explicitly contrasts the sensitivity of Mean Squared Error to immense mathematical deviations versus the conservative linearity of Mean Absolute Error. 

### Cell 5 & 6: K-Fold Statistical Validation Limit Testing
**Code Focus**: Scikit-learn `KFold` and `cross_val_score`.
**Reasoning**:
- A single train-test split fundamentally masks dataset variance. By partitioning identical parameters iteratively over 5 dynamically disjoint sets of data chunks (`cv=5`), the `cross_val_score` method verifies that our algorithm isn't simply memorizing specific order data. An aggressively tight standard deviation output fundamentally rejects the presence of overfitting (high variance) in our model.

### Cell 7 & 8: Multinomial Softmax Delivery Probability (Classification Task I)
**Code Focus**: `LogisticRegression(multi_class='multinomial')` integrated explicitly with `SMOTE`.
**Reasoning**:
- Traditional Logistic Regression maps binary 0/1 splits via sigmoid curves. Multinomial logistic regression forces probabilities to sum to 1.0 against exclusively distinct labels.
- Recognizing the fundamental risk diagnosed in Phase 1 regarding class imbalances, the framework utilizes `imblearn.pipeline.Pipeline` instead of `sklearn.pipeline`. This permits Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance class volume vectors solely during the `fit()` operation, preventing class leakage entirely.
- MLflow logs the Categorical Cross-Entropy (Log-Loss) metric explicitly. Lower loss represents the algorithm asserting high confidence in mathematically sound label estimations. 

### Cell 9 & 10: Multi-class Evaluation, Tradeoffs & Asymmetric Risk Execution
**Code Focus**: `confusion_matrix`, `classification_report`, Seaborn heatmaps.
**Reasoning**:
- Overall systemic accuracy routinely hides minoritarian class failure. The visual confusion matrix dynamically highlights misclassified matrices mapping specific prediction boundaries.
- The `classification_report` output specifically focuses analytical insight on **Precision, Recall, and F1-score**.
- **Asymmetric Risk Translation**: A system misidentifying a normal shipment as "Returned" (False Positive) imposes minimal structural damage compared to completely missing warning variables for actual returned shipments (False Negative - Poor Recall). The code implicitly argues mapping lowered probabilistic acceptance thresholds targeting exactly the "Returned" class to minimize fatal logistical blind spots.

### Cell 11 & 12: Vertex AI Champion Component Export (Classification Task II)
**Code Focus**: Pipeline locking and `joblib.dump()`.
**Reasoning**:
- Addressing the mandate in Phase 4.2 specifically ("deploy final model trained for customer segment"), we create a secondary structural instance of our Softmax/SMOTE framework but substitute `customer_segment` directly as the primary label objective variable.
- It executes entirely on the global schema (no train/test withholding). The fully fused execution chain (Raw Features -> StandardScaler/OrdinalEncoders -> SMOTE Handling -> Softmax Estimation) commits irreversibly to disk precisely as `model.joblib`. This monolithic asset eliminates operational friction moving toward GCP containerization.
