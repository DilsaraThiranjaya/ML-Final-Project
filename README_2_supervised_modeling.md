# Phase 2: Supervised Modeling & Experiment Tracking

## Overview
Phase 2 transforms our processed data into predictive power. We move beyond simple data manipulation into the realm of **Probabilistic Learning**. This notebook demonstrates how to predict both continuous values (Revenue) and discrete categories (Customer Segments) while maintaining a strict audit trail using **MLflow**.

### Learning Objectives
- **Stochastic Gradient Descent (SGD)**: Understanding how models learn iteratively.
- **Handling Class Imbalance**: Implementing **SMOTE** to ensure minority classes (like 'Returned' orders) aren't ignored.
- **Hyperparameter Optimization**: Using **Optuna** to automatically find the best settings for our algorithms.
- **The Model Registry**: Moving from a local script to a centralized MLOps "Champion" model management system.

---

## Cell-by-Cell Breakdown

### Cell 1-2: Connectivity & Artifact Retrieval
**Code Logic:**
- Connects to an SQLite-backed MLflow server.
- Loads the `ecommerce_cleaned.csv` and the `base_preprocessor.joblib` created in Phase 1.
**Strategic Rationale:**
- **Why SQLite for MLflow?** For beginners, using a local database ensures that your experiment history is saved even if you close the notebook.

### Cell 3: Preprocessor Alignment
**Code Logic:**
- Defines `get_aligned_preprocessor()`.
**Strategic Rationale:**
- This is a vital "Production" trick. If we decide to train a model that doesn't use all the original features, a standard Scikit-Learn pipeline would crash due to a shape mismatch. This logic allows us to "drop" transformers for missing columns dynamically.

### Cell 4: Revenue Regression (SGD)
**Code Logic:**
- Uses `SGDRegressor` to predict the `price` column.
- Logs MSE (Mean Squared Error) and MAE (Mean Absolute Error) to MLflow.
**Strategic Rationale:**
- **Why SGD?** Unlike standard Linear Regression, SGD is highly scalable and allows us to control the "Learning Rate"—the size of the steps the model takes toward the local minimum of error.

### Cell 6: Classification & SMOTE
**Code Logic:**
- Predicts `delivery_status`.
- Integrates `SMOTE` (Synthetic Minority Over-sampling Technique).
**Strategic Rationale:**
- **The Imbalance Problem**: Most orders are "Delivered." Very few are "Returned." If we didn't use SMOTE, the model would simply guess "Delivered" every time and achieve 90% accuracy while failing 100% of the time on the class we actually care about (Returns).

### Cell 7: Performance Audit
**Code Logic:**
- Generates a **Confusion Matrix**.
- Performs **Threshold Calibration**.
**Strategic Rationale:**
- We don't just care about "Accuracy." We care about **Recall**. Threshold Calibration allows a business manager to decide: "I'd rather have a few false alarms for returns (Low Precision) than miss a single return (High Recall)."

### Cell 8: The Champion Pipeline (Optuna + MLflow)
**Code Logic:**
- **Optuna Objective**: Iteratively tests different values for `C` (Regularization) and `solver`.
- **Nested Runs**: Every trial is saved under the main Optuna run in MLflow.
- **Registry Promotion**: The best model is labeled "CHAMPION" in the MLflow Model Registry.
**Strategic Rationale:**
- **Why Optuna?** Manual tuning is slow and prone to human error. Optuna uses Bayesian optimization to find the "needle in the haystack" of hyperparameters.
- **Why provide a Unified Joblib?** The final `model.joblib` contains BOTH the preprocessor and the classifier. A developer only needs to load **one** file to get the system running in a web app.

---

## Key Terms for Beginners
| Term | Meaning |
| :--- | :--- |
| **Log Loss** | A metric that punishes the model more heavily if it is confidently wrong than if it is hesitantly wrong. |
| **Hyperparameters** | The "knobs" on the algorithm (like learning rate) that we set BEFORE training starts. |
| **SMOTE** | Creates "fake" data points for minority classes to balance the training process. |
| **Experiment Tracking** | The practice of saving every result so you can compare today's model with one you built last week. |
