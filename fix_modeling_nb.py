import json

path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\2_supervised_modeling.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Helper to inject synchronization logic
sync_logic = [
    "\n",
    "# Synchronization Logic: Align preprocessor with available features in X\n",
    "def get_aligned_preprocessor(base_preprocessor, X_cols):\n",
    "    new_transformers = []\n",
    "    for name, transformer, columns in base_preprocessor.transformers:\n",
    "        # Filter columns to only those that exist in X\n",
    "        available_cols = [c for c in columns if c in X_cols]\n",
    "        if available_cols:\n",
    "            new_transformers.append((name, transformer, available_cols))\n",
    "    \n",
    "    # Create a modified version of the preprocessor\n",
    "    aligned_preprocessor = clone(base_preprocessor)\n",
    "    aligned_preprocessor.transformers = new_transformers\n",
    "    return aligned_preprocessor\n",
    "\n"
]

for i, cell in enumerate(cells):
    source = "".join(cell['source'])
    
    # 1. Inject the helper in the setup cell
    if 'preprocessor = joblib.load' in source:
        cell['source'].append("\n")
        cell['source'].append("# Synchronize the preprocessor logic with the available features in X\n")
        cell['source'].append("def get_aligned_preprocessor(base_preprocessor, X_cols):\n")
        cell['source'].append("    new_transformers = []\n")
        cell['source'].append("    for name, transformer, columns in base_preprocessor.transformers:\n")
        cell['source'].append("        available_cols = [c for c in columns if c in X_cols]\n")
        cell['source'].append("        if available_cols:\n")
        cell['source'].append("            new_transformers.append((name, transformer, available_cols))\n")
        cell['source'].append("    aligned_preprocessor = clone(base_preprocessor)\n")
        cell['source'].append("    aligned_preprocessor.transformers = new_transformers\n")
        cell['source'].append("    return aligned_preprocessor\n")

    # 2. Update Regression Pipeline
    if 'regression_pipeline = ImbPipeline' in source:
        new_source = []
        for line in cell['source']:
            if 'regression_pipeline = ImbPipeline' in line:
                new_source.append("    ('preprocessor', get_aligned_preprocessor(preprocessor, X_train_reg.columns)),\n")
            elif "('preprocessor', preprocessor)," in line:
                # Skip the original line
                continue
            else:
                new_source.append(line)
        cell['source'] = new_source

    # 3. Update Classification Pipeline
    if 'deliver_status_pipeline = ImbPipeline' in source:
        new_source = []
        for line in cell['source']:
            if 'deliver_status_pipeline = ImbPipeline' in line:
                 # This is the line before steps
                 new_source.append(line)
            elif "('preprocessor', preprocessor)," in line:
                 new_source.append("    ('preprocessor', get_aligned_preprocessor(preprocessor, X_tr_cl1.columns)),\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

    # 4. Update the Champion Vertex step (if present)
    if 'champ_pipeline = ImbPipeline' in source:
        new_source = []
        for line in cell['source']:
             if "('preprocessor', preprocessor)," in line or "('preprocessor', modified_transformer)," in line:
                 new_source.append("    ('preprocessor', get_aligned_preprocessor(preprocessor, X_champ.columns)),\n")
             else:
                 new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("✅ Supervised Modeling notebook updated with Preprocessor Column Alignment logic.")
