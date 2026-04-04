import json
import os

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1 (Imports)
# We find the cell that contains 'import pandas as pd'
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'import pandas as pd' in "".join(cell['source']):
        source = cell['source']
        # Add DROP columns at the end of the imports
        new_source = []
        for line in source:
            new_source.append(line)
        
        # Ensure we don't duplicate
        if 'DROP_ID_COLS' not in "".join(new_source):
            new_source.append("\n")
            new_source.append("# Constants for feature removal\n")
            new_source.append("DROP_ID_COLS = [\"order_id\", \"customer_id\", \"product_id\"]\n")
            new_source.append("DROP_TEXT_COLS = [\"shipping_address\", \"billing_address\"]\n")
        
        cell['source'] = new_source

# Update Cell 5 (Initial Data Cleaning - Code Cell 3)
# We find the cell that starts with "# Drop non-predictive identifiers"
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '# Drop non-predictive identifiers' in "".join(cell['source']):
        cell['source'] = [
            "# Rationale: Advanced feature extraction from temporal streams.\n",
            "data = df.copy()\n",
            "\n",
            "data[\"order_date\"] = pd.to_datetime(data[\"order_date\"], errors=\"coerce\")\n",
            "data[\"shipping_date\"] = pd.to_datetime(data[\"shipping_date\"], errors=\"coerce\")\n",
            "\n",
            "data[\"shipping_delay_days\"] = (\n",
            "    (data[\"shipping_date\"] - data[\"order_date\"]).dt.total_seconds() / 86400\n",
            ")\n",
            "\n",
            "data[\"order_month\"] = data[\"order_date\"].dt.month\n",
            "data[\"order_day\"] = data[\"order_date\"].dt.day\n",
            "data[\"order_dayofweek\"] = data[\"order_date\"].dt.dayofweek\n",
            "data[\"order_hour\"] = data[\"order_date\"].dt.hour\n",
            "\n",
            "data[\"shipping_month\"] = data[\"shipping_date\"].dt.month\n",
            "data[\"shipping_day\"] = data[\"shipping_date\"].dt.day\n",
            "data[\"shipping_dayofweek\"] = data[\"shipping_date\"].dt.dayofweek\n",
            "data[\"shipping_hour\"] = data[\"shipping_date\"].dt.hour\n",
            "\n",
            "data[\"is_weekend_order\"] = (data[\"order_date\"].dt.dayofweek >= 5).astype(int)\n",
            "\n",
            "drop_cols = DROP_ID_COLS + DROP_TEXT_COLS + [\"order_date\", \"shipping_date\"]\n",
            "df_clean = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')\n",
            "\n",
            "print(\"Feature extraction successful. Current columns:\")\n",
            "print(df_clean.columns.tolist())"
        ]

# Update the Preprocessor Cell
# We find the cell that defines numeric_features
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'numeric_features =' in "".join(cell['source']):
        new_source = []
        for line in cell['source']:
            if 'numeric_features =' in line:
                # Update with the new extraction columns
                new_source.append("numeric_features = ['quantity', 'order_month', 'order_day', 'order_dayofweek', 'order_hour', 'shipping_month', 'shipping_day', 'shipping_dayofweek', 'shipping_hour', 'shipping_delay_days', 'is_weekend_order']\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
