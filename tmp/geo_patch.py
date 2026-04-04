import json
import os

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 5 (Initial Data Cleaning)
# Target cell with "Advanced feature extraction" comment
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Advanced feature extraction' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'drop_cols =' in line:
                new_source.append("\n")
                new_source.append("# Extract geographic features (City and State) before dropping raw text\n")
                new_source.append("geo_pattern = r',\\s*([^,]+),\\s*([a-zA-Z\\s]+)\\s+\\d+$'\n")
                new_source.append("data[['shipping_city', 'shipping_state']] = data['shipping_address'].str.extract(geo_pattern)\n")
                new_source.append("data[['billing_city', 'billing_state']] = data['billing_address'].str.extract(geo_pattern)\n")
                new_source.append("\n")
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

# Update Cell 6 (Preprocessing Architecture)
# Target cell with "nominal_features ="
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'nominal_features =' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'nominal_features =' in line:
                # Add States to nominal features for OneHotEncoding
                new_source.append("nominal_features = ['category', 'payment_method', 'channel', 'shipping_state', 'billing_state']\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook geographic extraction and pipeline successfully updated.")
