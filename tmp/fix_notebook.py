import json
import os

path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\2_supervised_modeling.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell['source']
        if any('customer_segmentation_optuna_study' in line for line in source):
            new_source = []
            added = False
            for i, line in enumerate(source):
                new_source.append(line)
                # Look for the closing parenthesis of train_test_split
                if 'train_test_split(' in "".join(source[max(0, i-5):i+1]) and ')' in line and not added:
                    # Make sure we're in the right context
                    if 'X_tr_seg' in "".join(source[max(0, i-5):i+1]):
                        new_source.append('preprocessor_seg = get_aligned_preprocessor(preprocessor, X_tr_seg.columns)\n')
                        new_source.append('\n')
                        added = True
            cell['source'] = new_source
            if added:
                print("Successfully updated the Optuna cell.")
            else:
                print("Found cell but could not match line for insertion.")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
