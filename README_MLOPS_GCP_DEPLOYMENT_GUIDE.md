# AuraCart Retail Analytics: Production-Grade MLOps & GCP Deployment Guide

This document provides a holistic architectural overview of the AuraCart machine learning ecosystem. It details the end-to-end lifecycle from local experimentation tracking with MLflow to enterprise-scale deployment on Google Cloud Vertex AI.

---

## 1. MLOps Strategy: Experiment Tracking with MLflow

AuraCart identifies experimental reproducibility as a critical operational pillar. We utilize **MLflow** to eliminate "black box" modeling.

### Tracking Architecture
- **Tracking URI**: Every experiment is bound to a local or remote MLflow tracking server using `mlflow.set_tracking_uri()`. This ensures that metrics are persisted beyond the execution of a single notebook.
- **Parametric Logging**: We log every hyperparameter (Learning Rate, Epochs, Batch Size) via `mlflow.log_params()`. This allows architects to compare "Champion" vs "Challenger" models.
- **Metric Versioning**: Key performance indicators (MSE, MAE, Log-Loss, F1-Score) are logged via `mlflow.log_metric()`. This enables team-wide visibility into model decay or improvement over time.
- **Artifact Locking**: We serialize our unified Scikit-Learn pipelines as `.joblib` binary artifacts, ensuring that the exact preprocessing logic used during training is what powers the production endpoint.

---

## 2. Google Cloud Platform (GCP) Infrastructure

The system transitions from local development to cloud production using the following Google Cloud services.

### Phase 2.1: Artifact Storage (Google Cloud Storage)
- **Role**: GCS acts as the central repository for model binaries.
- **Process**:
    1. A storage bucket is programmatically instantiated (e.g., `gs://your-project-id-ml-artifacts/`).
    2. The champion `model.joblib` and its `requirements.txt` dependency file are uploaded.
    3. Vertex AI uses this specific URI as the source-of-truth for its Model Registry.

### Phase 2.2: Vertex AI Model Registry
- **Role**: Centralized version control for all production models.
- **Process**:
    - We initialize the `aiplatform` SDK.
    - We "Upload" the model to the registry, specifically mapping it to a **Pre-built Prediction Container** (e.g., `sklearn-cpu.1-3`). This removes the need for manual Docker maintenance.

### Phase 2.3: Vertex AI Endpoints (Production Inference)
- **Role** Provides a scalable, high-availability REST API for real-time traffic.
- **Process**:
    - The registered model is "Deployed" to an endpoint resource.
    - We select the machine type (e.g., `n1-standard-2`) based on AuraCart's latency requirements.
    - Once "Live", the endpoint exposes a unique resource URI that accepts JSON payloads.

---

## 3. Operational Integration: Querying the System

AuraCart's frontend microservices interact with the ML system via standard HTTP POST requests.

### JSON Payload Structure
To query the endpoint, frontends send a JSON array representing the transaction features:
```json
{
  "instances": [
    [2, 14, 4, 3, "Electronics", "Credit Card", "Mobile", "Organic"]
  ]
}
```

### Result Processing
The Vertex AI endpoint returns a probability distribution or the absolute class index (e.g., `0` for 'New'). The frontend then maps this back to business-readable strings (e.g., 'VIP') to trigger hyper-personalized campaign logic.

---

## 4. Maintenance and Future Scalability

- **Retraining Loops**: As new transactional data accumulates, Phase 2 notebooks can be re-executed to update the `model.joblib` artifact in GCS.
- **Model Monitoring**: Vertex AI Model Monitoring can be enabled to detect "Feature Drift" (changes in customer behavior over time) and trigger automated retraining alerts.
- **CI/CD Integration**: These notebooks can be converted to Python scripts and integrated into GitHub Actions or Google Cloud Build for automated model promotion.
