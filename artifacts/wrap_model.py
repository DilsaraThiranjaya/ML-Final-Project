import joblib
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Define the Column Restorer Class locally for the script
class DataFrameRestorer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Convert NumPy array/List back into a named DataFrame
        # Vertex AI passes data as lists/arrays, so we must restore names for ColumnTransformer
        return pd.DataFrame(X, columns=self.columns)

# Exact order of features from training (df_clean in Phase 1)
TRAINING_COLUMNS = [
    'category', 'price', 'quantity', 'delivery_status', 'payment_method', 
    'device_type', 'channel', 'shipping_delay_days', 'order_month', 
    'order_day', 'order_dayofweek', 'order_hour', 'shipping_month', 
    'shipping_day', 'shipping_dayofweek', 'shipping_hour', 'is_weekend_order', 
    'shipping_city', 'shipping_state', 'billing_city', 'billing_state', 'product_popularity'
]

MODEL_PATH = "c:/Users/ACER/Downloads/ML Test 4/artifacts/model.joblib"

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    old_model = joblib.load(MODEL_PATH)
    
    # Construct the wrapped pipeline
    wrapped_model = Pipeline([
        ('restorer', DataFrameRestorer(columns=TRAINING_COLUMNS)),
        ('original_pipeline', old_model)
    ])
    
    # Overwrite the artifact
    joblib.dump(wrapped_model, MODEL_PATH)
    print("Success: Champion model has been wrapped with DataFrameRestorer and re-saved.")
else:
    print(f"Error: Could not find model artifact at {MODEL_PATH}")
