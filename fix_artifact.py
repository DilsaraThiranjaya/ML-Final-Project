import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Load the cleaned data
df_clean = pd.read_csv(r'c:\Users\ACER\Downloads\ML Test 4\artifacts\ecommerce_cleaned.csv')

# Re-define the column categorization from the updated notebook
numeric_features = [
    'price', 'quantity', 'shipping_delay_days', 'product_popularity',
    'order_month', 'order_day', 'order_dayofweek', 'order_hour',
    'shipping_month', 'shipping_day', 'shipping_dayofweek', 'shipping_hour'
]

nominal_features = [
    'category', 'payment_method', 'device_type', 'channel', 
    'shipping_state', 'billing_state', 'is_weekend_order'
]

ordinal_features = ['customer_segment']
customer_segment_order = [['New', 'Returning', 'VIP']]

# Re-build the transformer
preprocessor_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features),
        ('ord', OrdinalEncoder(categories=customer_segment_order, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
    ],
    remainder='drop'
)

# Wrap in the Pipeline (as per guidelines)
base_preprocessor = Pipeline(steps=[
    ('preprocessor', preprocessor_transformer)
])

# Fit on the current data
base_preprocessor.fit(df_clean)

# Save the unified artifact
joblib.dump(base_preprocessor, r'c:\Users\ACER\Downloads\ML Test 4\artifacts\base_preprocessor.joblib')

print("✅ Artifact base_preprocessor.joblib updated to <class 'sklearn.pipeline.Pipeline'>.")
