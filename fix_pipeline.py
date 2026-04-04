import json
import os

path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new preprocessing code
preprocessor_code = [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder\n",
    "\n",
    "# 1. Define Column Categorization\n",
    "numeric_features = [\n",
    "    'price', 'quantity', 'shipping_delay_days', 'product_popularity',\n",
    "    'order_month', 'order_day', 'order_dayofweek', 'order_hour',\n",
    "    'shipping_month', 'shipping_day', 'shipping_dayofweek', 'shipping_hour'\n",
    "]\n",
    "\n",
    "nominal_features = [\n",
    "    'category', 'payment_method', 'device_type', 'channel', \n",
    "    'shipping_state', 'billing_state', 'is_weekend_order'\n",
    "]\n",
    "\n",
    "ordinal_features = ['customer_segment']\n",
    "# Setting correct hierarchy: New < Returning < VIP\n",
    "customer_segment_order = [['New', 'Returning', 'VIP']]\n",
    "\n",
    "# 2. Build the ColumnTransformer\n",
    "preprocessor_transformer = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', StandardScaler(), numeric_features),\n",
    "        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features),\n",
    "        ('ord', OrdinalEncoder(categories=customer_segment_order, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)\n",
    "    ],\n",
    "    remainder='drop'\n",
    ")\n",
    "\n",
    "# 3. Wrap in a Modular Scikit-learn Pipeline (Targeting Task Guidelines)\n",
    "base_preprocessor = Pipeline(steps=[\n",
    "    ('preprocessor', preprocessor_transformer)\n",
    "])\n",
    "\n",
    "# 4. Fit and Serialize\n",
    "base_preprocessor.fit(df_clean)\n",
    "joblib.dump(base_preprocessor, '../artifacts/base_preprocessor.joblib')\n",
    "\n",
    "print(\"✅ Unified Preprocessing Pipeline built and serialized.\")\n",
    "print(f\"Feature footprint size: {base_preprocessor.transform(df_clean).shape[1]} columns.\")"
]

# Find the cell that creates the preprocessor (Cell 37 or based on content)
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "ColumnTransformer" in "".join(cell['source']):
        cell['source'] = preprocessor_code
        cell['outputs'] = []
        found = True
        break

if not found:
    # If for some reason we didn't find it, append it
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": preprocessor_code
    })

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
