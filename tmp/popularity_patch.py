import json
import os

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 5 (Initial Data Cleaning)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Advanced feature extraction' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'drop_cols =' in line:
                new_source.append("\n")
                new_source.append("# Extract Product Popularity (Frequency Count) before dropping product_id\n")
                new_source.append("data['product_popularity'] = data['product_id'].map(data['product_id'].value_counts())\n")
                new_source.append("\n")
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

# Update Cell 6 (Preprocessing Architecture)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'numeric_features =' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'numeric_features =' in line:
                # Add product_popularity to numeric_features for scaling
                new_source.append("numeric_features = ['quantity', 'order_month', 'order_day', 'order_dayofweek', 'order_hour', 'shipping_month', 'shipping_day', 'shipping_dayofweek', 'shipping_hour', 'shipping_delay_days', 'is_weekend_order', 'product_popularity']\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Product Popularity featured and pipeline successfully updated.")
