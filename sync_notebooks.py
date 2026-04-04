import json
import os

base_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks'

# --- 1. Fix Supervised Modeling ---
path2 = os.path.join(base_path, '2_supervised_modeling.ipynb')
with open(path2, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    if cell['cell_type'] == 'code' and "champ_pipeline = ImbPipeline" in "".join(cell['source']):
        source = "".join(cell['source'])
        # New logic to handle the Pipeline-wrapped preprocessor
        new_logic = [
            "# Utilizing Customer Segment as Target. \n",
            "X_champ = df.drop(columns=['customer_segment'])\n",
            "y_champ = df['customer_segment']\n",
            "\n",
            "# Access the inner transformer since 'preprocessor' is now a Pipeline step\n",
            "inner_transformer = preprocessor.named_steps['preprocessor']\n",
            "modified_transformer = clone(inner_transformer).set_params(\n",
            "    transformers=[t for t in inner_transformer.transformers if t[0] != 'ord']\n",
            ")\n",
            "\n",
            "champ_pipeline = ImbPipeline(steps=[\n",
            "    ('preprocessor', modified_transformer),\n",
            "    ('smote', SMOTE(random_state=42)),\n",
            "    ('classifier', LogisticRegression(solver='lbfgs', max_iter=2000))\n",
            "])\n",
            "\n",
            "with mlflow.start_run(run_name=\"champion_vertex_deployment_model\"):\n",
            "    champ_pipeline.fit(X_champ, y_champ) # Model trains on full operational schema for total coverage\n",
            "    \n",
            "    # Registering standard accuracy metrics for tracking logs\n",
            "    preds = champ_pipeline.predict(X_champ)\n",
            "    f1_weighted = classification_report(y_champ, preds, output_dict=True)['weighted avg']['f1-score']\n",
            "    mlflow.log_metric(\"final_f1_score\", f1_weighted)\n",
            "    \n",
            "    # Serialize the Unified Binary Artifact (Requirement 4.2)\n",
            "    FINAL_ARTIFACT_PATH = '../artifacts/model.joblib'\n",
            "    joblib.dump(champ_pipeline, FINAL_ARTIFACT_PATH)\n",
            "    print(f\"Success! Unified predictive engine and preprocessing transformer fused into single serialization blob.\")\n",
            "    print(f\"Champion Artifact localized at: {FINAL_ARTIFACT_PATH}\")"
        ]
        cell['source'] = new_logic

with open(path2, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=4)

# --- 2. Fix Unsupervised Clustering ---
path3 = os.path.join(base_path, '3_unsupervised_clustering.ipynb')
with open(path3, 'r', encoding='utf-8') as f:
    nb3 = json.load(f)

for cell in nb3['cells']:
    content = "".join(cell['source'])
    # Fix the mean() calculation and the scatter plot
    if "shipping_duration_days" in content:
        cell['source'] = [line.replace("shipping_duration_days", "shipping_delay_days") for line in cell['source']]
    
    # Also update the display selection if needed
    if "cluster_analysis = df.groupby('cluster')" in content:
        cell['source'] = [
            "# Aggregate numeric features by cluster\n",
            "cluster_analysis = df.groupby('cluster')[['price', 'quantity', 'shipping_delay_days', 'order_hour', 'product_popularity']].mean()\n",
            "\n",
            "print(\"--- Cluster Centroid Analysis ---\")\n",
            "display(cluster_analysis)"
        ]

with open(path3, 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=4)

# --- 3. Fix MLOps Deployment ---
path4 = os.path.join(base_path, '4_mlops_deployment.ipynb')
with open(path4, 'r', encoding='utf-8') as f:
    nb4 = json.load(f)

for cell in nb4['cells']:
    if cell['cell_type'] == 'code' and "test_instance =" in "".join(cell['source']):
        cell['source'] = [
            "import pandas as pd\n",
            "\n",
            "# Simulate input features for a New Transaction using all 21 input features\n",
            "test_data = {\n",
            "    'price': [150.0], 'quantity': [2], 'shipping_delay_days': [4], 'product_popularity': [15],\n",
            "    'order_month': [4], 'order_day': [15], 'order_dayofweek': [2], 'order_hour': [14],\n",
            "    'shipping_month': [4], 'shipping_day': [19], 'shipping_dayofweek': [6], 'shipping_hour': [10],\n",
            "    'category': ['Electronics'], 'payment_method': ['Credit Card'], 'device_type': ['Mobile'],\n",
            "    'channel': ['Direct'], 'shipping_city': ['Austin'], 'shipping_state': ['TX'],\n",
            "    'billing_city': ['Austin'], 'billing_state': ['TX'], 'is_weekend_order': [0]\n",
            "}\n",
            "\n",
            "test_instance_df = pd.DataFrame(test_data)\n",
            "\n",
            "print(\"Querying endpoint with the enhanced 21-feature schema...\")\n",
            "# Note: In a real Vertex query, this df would be converted to a list for the 'instances' key\n",
            "prediction = endpoint.predict(instances=test_instance_df.values.tolist())\n",
            "\n",
            "print(f\"Prediction Response Structure: {prediction}\")\n",
            "print(f\"Assigned Customer Segment: {prediction.predictions[0]}\")"
        ]

with open(path4, 'w', encoding='utf-8') as f:
    json.dump(nb4, f, indent=4)

print("Synchronized all notebooks successfully.")
