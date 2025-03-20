import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
    Simplified model to predict expected_proceeds_d8 for trial users.
    
    The approach is to predict trial conversion probability and calculate
    expected proceeds using a fixed average value.
    """
    
    def __init__(self, product_dim_df=None):
        """
        Initialize the model with optional product dimension data.
        
        Args:
            product_dim_df: DataFrame containing product pricing information
        """
        self.model = None
        self.preprocessor = None
        # Simplified feature list
        self.features = [
            'channel_group', 'marketing_network_id', 'target_market',
            'signup_country_group', 'signup_client_platform', 
            'is_trial_autorenewal_on_d0', 'started_content', 'n_content_starts'
        ]
        self.product_dim_df = product_dim_df
        # Fixed average proceeds value
        self.average_proceeds = 70.0
        logger.info("TrialPredictionModel initialized with fixed average proceeds of %.2f", self.average_proceeds)
        
    def fit(self, training_d8_df: pd.DataFrame, training_d100_df: pd.DataFrame):
        """
        Train the model using historical data.
        
        Args:
            training_d8_df: DataFrame containing d8 training data
            training_d100_df: DataFrame containing d100 training data
        """
        logger.info("Starting trial model fitting...")
        
        # Prepare training data - use only features that exist in the DataFrame
        available_features = [f for f in self.features if f in training_d8_df.columns]
        if len(available_features) < len(self.features):
            missing_features = set(self.features) - set(available_features)
            logger.warning("Missing features in training data: %s", missing_features)
        
        X = training_d8_df[available_features]
        
        # For trial users, we want to predict if they converted to a paid plan (binary target)
        # 1 if they have proceeds > 0, 0 otherwise
        y = (training_d8_df['eur_proceeds_d8'] > 0).astype(int)
        
        # Log target values
        logger.info("Target distribution: 0s: %d, 1s: %d", sum(y==0), sum(y==1))
        
        # Define preprocessing for numerical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        logger.info("Numerical features: %s", numerical_features)
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info("Categorical features: %s", categorical_features)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create and fit the preprocessor
        logger.info("Fitting preprocessor...")
        self.preprocessor = preprocessor.fit(X)
        
        # Transform the training data
        logger.info("Transforming training data...")
        X_processed = self.preprocessor.transform(X)
        
        # Create and train the model
        logger.info("Creating XGBClassifier...")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train the model with binary target
        logger.info("Training model with X shape: %s, y shape: %s", X_processed.shape, y.shape)
        self.model.fit(X_processed, y)
        logger.info("Model training completed")
    
    def predict(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected proceeds for trial users.
        
        Args:
            inference_df: DataFrame containing trial users for inference
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting trial model prediction...")
        
        # Make a copy of the inference data
        result_df = inference_df.copy()
        
        # Extract features for prediction - use only features that exist in the DataFrame
        available_features = [f for f in self.features if f in result_df.columns]
        if len(available_features) < len(self.features):
            missing_features = set(self.features) - set(available_features)
            logger.warning("Missing features in inference data: %s", missing_features)
        
        X = result_df[available_features]
        logger.info("Inference data shape: %s", X.shape)
        
        # Preprocess features
        logger.info("Preprocessing inference features...")
        X_processed = self.preprocessor.transform(X)
        
        # Predict conversion probability
        logger.info("Predicting conversion probabilities...")
        conversion_probs = self.model.predict_proba(X_processed)[:, 1]
        logger.info("Generated %d probability predictions", len(conversion_probs))
        logger.info("Probability range: %.4f to %.4f", conversion_probs.min(), conversion_probs.max())
        
        # Calculate expected proceeds using fixed average proceeds
        logger.info("Calculating expected proceeds using fixed average of %.2f", self.average_proceeds)
        expected_proceeds = conversion_probs * self.average_proceeds
        
        # Add predictions to result DataFrame
        result_df['expected_proceeds_d8'] = expected_proceeds
        
        # For trial users, d100 is the same as d8
        result_df['expected_proceeds_d100'] = result_df['expected_proceeds_d8']
        
        logger.info("Prediction completed for %d trial users", len(result_df))
        logger.info("Average expected proceeds: %.2f", expected_proceeds.mean())
        return result_df