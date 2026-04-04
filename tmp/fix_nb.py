import json
import os
import re

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\2_supervised_modeling.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cells
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = "".join(cell['source'])
    
    # Imports
    if 'preprocessor = joblib.load' in source:
        if 'from sklearn.base import clone' not in source:
            new_lines = []
            for line in cell['source']:
                if 'from sklearn.model_selection' in line:
                    new_lines.append(line)
                    new_lines.append("from sklearn.base import clone\n")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
    
    # Task 3.2 & 3.3: Dropping columns
    if 'X_reg = df.drop' in source or 'X_class1 = df.drop' in source:
        new_lines = []
        for line in cell['source']:
            new_lines.append(line.replace(", 'customer_segment'", ""))
        cell['source'] = new_lines

    # Task 4.2: Champion model
    if "champ_pipeline = ImbPipeline(steps=[" in source:
        new_source = source.replace(
            "    ('preprocessor', preprocessor),",
            "    # Since customer_segment is now the target, we must exclude it from the preprocessor's mapping\n    # to avoid a KeyError. We create a copy of the preprocessor without the 'ord' transformer.\n    ('preprocessor', clone(preprocessor).set_params(transformers=[t for t in preprocessor.transformers if t[0] != 'ord'])),"
        )
        cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]

# Note: The set_params approach above is better than manual transformers list update on fitted ColumnTransformers, 
# although here it's likely un-fitted.

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
