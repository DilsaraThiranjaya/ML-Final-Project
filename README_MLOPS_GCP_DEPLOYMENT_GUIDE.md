# AuraCart MLOps & GCP Deployment Guide

This document provides the definitive setup instructions for the AuraCart production machine learning infrastructure, utilizing **MLflow** for local experiment tracking and **Google Cloud Vertex AI** for cloud-scale deployment.

---

## 1. Local Experiment Tracking (MLflow & Optuna)

AuraCart uses **MLflow** with a SQLite backend for persistent tracking and **Optuna** for Bayesian hyperparameter optimization.

### **Launching the MLflow UI**
To visualize your model trials, parallel coordinate plots, and metrics, run the following command in your terminal from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### **Accessing the Dashboard**
1.  Open your browser and navigate to `http://127.0.0.1:5000`.
2.  Select the experiment: **`AuraCart_Market_Segmentation_Tuning`**.
3.  **Nested Runs**: Click the `+` icon next to a parent run to see the individual Optuna trials.
4.  **Comparison**: Select multiple trials and click **Compare** to view the trade-offs between parameters like `C` and `solver` against the `VIP_Recall` metric.

---

## 2. Google Cloud Platform (GCP) Preparation

Before running `4_mlops_deployment.ipynb`, ensure your cloud environment is correctly configured.

### **Phase A: Project Identification**
*   **Project ID**: `dilsara`
*   **Location/Region**: `asia-southeast1`

### **Phase B: Enabling Required APIs**
Navigate to the [GCP API Console](https://console.cloud.google.com/apis/library) and enable the following:
1.  **Vertex AI API** (`aiplatform.googleapis.com`)
2.  **Cloud Storage API** (`storage-api.googleapis.com`)

### **Phase C: IAM & Service Account Setup**
1.  **Create Service Account**: Go to `IAM & Admin` > `Service Accounts` and create a new account (e.g., `auracart-deployer`).
2.  **Assign Roles**:
    *   `Vertex AI User`
    *   `Storage Admin`
3.  **Generate Key**: Click on the service account > `Keys` > `Add Key` > `Create New Key` (JSON). 
4.  **Local Authentication**: Save this file and set the environment variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
    ```

---

## 3. Vertex AI Deployment Workflow

The following steps are automated within `notebooks/4_mlops_deployment.ipynb`:

### **Step 1: Artifact Preparation**
We package the Scikit-learn champion model into a standardized `model.joblib` and generate a `requirements.txt` to align the Vertex AI container environment with our local training state.

### **Step 2: Cloud Storage (GCS) Uplink**
The system creates a bucket named `dilsara-auracart-ml-artifacts` and uploads the artifacts. Vertex AI requires models to be served from a GCS location.

### **Step 3: Model Registry Ingestion**
The model is uploaded to the **Vertex AI Model Registry**. We use the pre-built Scikit-Learn prediction container image:
`us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest`

### **Step 4: Endpoint Instantiation**
The model is deployed to a **Vertex AI Managed Endpoint**. This provides a REST interface for real-time predictions. 
*   **Machine Type**: `n1-standard-2` (Recommended for balanced performance/cost).

---

## 4. Querying the Production API
Once the endpoint is `READY`, you can send JSON payloads. The system expects **20 input features** (Aligned with the Phase 1 Preprocessing schema):

```json
{
  "instances": [
    [155.5, 2, 4, 12, 6, 4, 15, 2, 3, 1, 85, "Home & Kitchen", "Delivered", "Credit Card", "Social Media", "San Jose", "CA", "San Jose", "CA", "Mobile"]
  ]
}
```

> [!TIP]
> Use the "Online Prediction" tab in the Vertex AI Console for a visual interface to test your live model behavior.
