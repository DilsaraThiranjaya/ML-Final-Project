import json
import os

notebook_path = r'c:\Users\ACER\Downloads\ML Test 4\notebooks\1_eda_and_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1 (Imports)
# We find the cell that contains 'from datasets import load_dataset'
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'datasets' in "".join(cell['source']):
        cell['source'] = [
            "# Rationale: Import all required libraries for data manipulation, visualization, and ML preprocessing.\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "import os\n",
            "\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder\n",
            "\n",
            "# Set visualization style\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Dataset source URL\n",
            "DATA_URL = \"https://huggingface.co/datasets/millat/e-commerce-orders/raw/main/ecommerce_orders_clean.csv\"\n",
            "\n",
            "# Create artifacts directory for saving the pipeline later\n",
            "os.makedirs('../artifacts', exist_ok=True)"
        ]

# Update Cell 3 (Loading)
# We find the cell that contains 'load_dataset('
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'load_dataset(' in "".join(cell['source']):
        cell['source'] = [
            "# Rationale: Load the data from HuggingFace URL and inspect the structure.\n",
            "print(\"Loading dataset from remote URL...\")\n",
            "df = pd.read_csv(DATA_URL)\n",
            "\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "display(df.head())"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
