import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrialPredictionModel')

class TrialPredictionModel:
    """
    Model to predict expected proceeds for trial users.
    """
    
    def __init__(self, product_dim_df=None):
        """
        Initialize the model with product dimension data.
        
        Args:
            product_dim_df: DataFrame containing product pricing information
        """
        self.model = None
        self.preprocessor = None
        self.features = [
            'report_date', 'channel_group', 'marketing_network_id', 'target_market',
            'eur_marketing_spend', 'impressions', 'clicks', 'signup_country_group',
            'signup_client_platform', 'is_trial_subscription', 'is_trial_autorenewal_on_d0',
            'started_content', 'n_content_starts', 'finished_content', 'n_content_finishes', 'plan_tier'
        ]
        self.product_dim_df = product_dim_df
        self.product_proceeds = {}
        self.country_product_proceeds = {}
        self.global_average_proceeds = 0.0
        
    def _calculate_historical_proceeds(self, training_df):
        """
        Calculate average proceeds for different products and country groups.
        
        Args:
            training_df: DataFrame with historical data
        """
        # Not sure how logging in dbt works, but we might not need all these logs below.
        print("Calculating historical proceeds...")
        
        # Filter to only include records with non-zero proceeds
        converted_df = training_df[training_df['eur_proceeds_d8'] > 0].copy()
        
        # Calculate global average proceeds
        self.global_average_proceeds = converted_df['eur_proceeds_d8'].mean()
        
        # Calculate average proceeds by product
        self.product_proceeds = converted_df.groupby('product_name')['eur_proceeds_d8'].mean().to_dict()
        
        # Calculate average proceeds by product and country group
        for (country, product), group in converted_df.groupby(['signup_country_group', 'product_name']):
            if len(group) >= 10:  # Only use combinations with enough data
                self.country_product_proceeds[(country, product)] = group['eur_proceeds_d8'].mean()
    
    def _get_average_proceeds(self, user_row):
        """
        Get the average proceeds for a user based on their country group and product.
        
        Args:
            user_row: Series containing user data
            
        Returns:
            Average proceeds value
        """
        # Check if we have product-specific data
        if 'product_name' not in user_row:
            # If product_name is missing, use global average
            return self.global_average_proceeds
            
        product = user_row['product_name']
        if product in self.product_proceeds:
            # Check if we have country-product specific data
            if ('signup_country_group' in user_row and 
                (user_row['signup_country_group'], product) in self.country_product_proceeds):
                return self.country_product_proceeds[(user_row['signup_country_group'], product)]
            
            # Fall back to product-specific data
            return self.product_proceeds[product]
        
        # Fall back to global average
        return self.global_average_proceeds
        
    def fit(self, training_d8_df):
        """
        Train the model using historical data.
        
        Args:
            training_d8_df: DataFrame containing d8 training data
            training_d100_df: DataFrame containing d100 training data
        """
        print("Starting trial model fitting...")
        
        # Calculate historical proceeds
        self._calculate_historical_proceeds(training_d8_df)
        
        # Prepare training data
        X = training_d8_df[self.features]
        
        # For trial users, we want to predict if they converted to a paid plan (binary target)
        # 1 if they have proceeds > 0, 0 otherwise
        y = (training_d8_df['eur_proceeds_d8'] > 0).astype(int)
        
        # Log target values
        print(f"Target distribution: 0s: {sum(y==0)}, 1s: {sum(y==1)}")
                
        # Define preprocessing for numerical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # not sure this strategy makes sense if e.g. channel_group or marketing_network_id is missing . I feel we shoudld then just drop the row (in data import step)
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Transform the training data
        print("Transforming training data...")
        X_processed = self.preprocessor.fit_transform(X)
        
        # Create and train the model
        print("Creating XGBClassifier...")
        self.model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=42
        )
        
        # Train the model with binary target
        print(f"Training model with X shape: {X_processed.shape}, y shape: {y.shape}")
        self.model.fit(X_processed, y)
        print("Model training completed")
    
    def predict(self, inference_df):
        """
        Predict expected proceeds for trial users.
        
        Args:
            inference_df: DataFrame containing trial users for inference
            
        Returns:
            DataFrame with predictions
        """
        print("Starting trial model prediction...")
        
        # Make a copy of the inference data
        result_df = inference_df.copy()  # don't think this is actually needed, especially when running in dbt
        
        # Extract features for prediction
        X = result_df[self.features]
        print(f"Inference data shape: {X.shape}")
        
        # Preprocess features
        print("Preprocessing inference features...")
        X_processed = self.preprocessor.transform(X)
        
        # Predict conversion probability
        print("Predicting conversion probabilities...")
        conversion_probs = self.model.predict_proba(X_processed)[:, 1]
        
        # Calculate expected proceeds
        print("Calculating expected proceeds...")
        expected_proceeds = []
        for i, prob in enumerate(conversion_probs):
            # Get average proceeds for this user's country group and product
            avg_proceeds = self._get_average_proceeds(result_df.iloc[i])
            # Expected proceeds = probability of conversion * average proceeds if converted
            expected_proceeds.append(prob * avg_proceeds)
        
        # Add predictions to result DataFrame
        result_df['expected_proceeds_d8'] = expected_proceeds
        
        # For trial users, d100 is the same as d8
        result_df['expected_proceeds_d100'] = expected_proceeds
        
        # Ensure no negative values in predictions
        result_df[['expected_proceeds_d8', 'expected_proceeds_d100']] = result_df[['expected_proceeds_d8', 'expected_proceeds_d100']].clip(lower=0)
        
        print(f"Prediction completed for {len(result_df)} trial users")
        return result_df
