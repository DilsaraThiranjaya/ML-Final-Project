import json
import os

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the Preprocessor Cell
# We find the cell that defines numeric_features (it's around line 404 in the previously viewed file)
target_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'numeric_features =' in "".join(cell['source']):
        new_source = []
        for line in cell['source']:
            if 'numeric_features =' in line:
                # Update with the new extraction columns
                # We also include 'is_weekend_order' and ensure 'shipping_delay_days' is used
                new_source.append("numeric_features = ['quantity', 'order_month', 'order_day', 'order_dayofweek', 'order_hour', 'shipping_month', 'shipping_day', 'shipping_dayofweek', 'shipping_hour', 'shipping_delay_days', 'is_weekend_order']\n")
                target_found = True
            else:
                new_source.append(line)
        cell['source'] = new_source

if target_found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook preprocessor aligned successfully.")
else:
    print("Could not find numeric_features definition.")
