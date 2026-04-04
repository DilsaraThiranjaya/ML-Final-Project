import json
import os

path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\2_supervised_modeling.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "inner_transformer = preprocessor.named_steps['preprocessor']" in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = [
            "# Utilizing Customer Segment as Target. \n",
            "X_champ = df.drop(columns=['customer_segment'])\n",
            "y_champ = df['customer_segment']\n",
            "\n",
            "# Robust access: account for Pipeline or raw ColumnTransformer types\n",
            "if hasattr(preprocessor, 'named_steps'):\n",
            "    inner_transformer = preprocessor.named_steps['preprocessor']\n",
            "else:\n",
            "    inner_transformer = preprocessor\n",
            "\n",
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
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated with robustness logic.")
