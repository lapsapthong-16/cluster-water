import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import joblib

# === Custom Transformer to Add Time Features ===
class TimeFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='Timestamp'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df['Hour'] = df[self.time_column].dt.hour
        df['DayOfYear'] = df[self.time_column].dt.dayofyear
        df['Month'] = df[self.time_column].dt.month
        return df.drop(columns=[self.time_column])

# === Custom Transformer to Drop Quality & Non-Feature Columns ===
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# === Custom Transformer to Remove Outliers ===
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if input is a pandas DataFrame or a NumPy array
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            z = np.abs((df - df.mean()) / df.std())
            filtered_df = df[(z < self.threshold).all(axis=1)]
            return filtered_df.reset_index(drop=True)
        else:
            # Handle NumPy array
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            z_scores = np.abs((X - X_mean) / (X_std + 1e-10))  # Add small epsilon to avoid division by zero
            mask = (z_scores < self.threshold).all(axis=1)
            return X[mask]

# === List of columns to drop based on your notebook ===
columns_to_drop = [
    'Record number', 'Chlorophyll [quality]', 'Temperature [quality]',
    'Dissolved Oxygen [quality]', 'Dissolved Oxygen (%Saturation) [quality]',
    'pH [quality]', 'Salinity [quality]', 'Specific Conductance [quality]',
    'Turbidity [quality]'
]

# === Full Pipeline ===
training_pipeline = Pipeline([
    ('drop_columns', DropColumns(columns_to_drop=columns_to_drop)),
    ('add_time_features', TimeFeaturesAdder()),
    ('impute_missing', SimpleImputer(strategy='median')),
    ('remove_outliers', OutlierRemover(threshold=3)),  # ✅ KEEP THIS HERE
    ('scaling', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

inference_pipeline = Pipeline([
    ('drop_columns', DropColumns(columns_to_drop=columns_to_drop)),
    ('add_time_features', TimeFeaturesAdder()),
    ('impute_missing', SimpleImputer(strategy='median')),
    # ❌ No outlier remover
    ('scaling', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

joblib.dump(training_pipeline, 'pipeline_training.pkl')
joblib.dump(inference_pipeline, 'pipeline_inference.pkl')

# # Load trained clustering model and inference pipeline
# pipeline = joblib.load('pipeline_inference.pkl')
# model = joblib.load('trained_model.pkl')

# # Get user input
# user_input = pd.DataFrame([user_input_dict])

# # Transform input
# X_transformed = pipeline.transform(user_input)

# # Predict cluster
# predicted_cluster = model.predict(X_transformed)