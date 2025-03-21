import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple

class LagPurchasePredictionModel:
    """
    Model to predict expected_proceeds_d8 and expected_proceeds_d100 for users who haven't purchased yet.
    
    The approach is two-fold:
    1. Predict probability of purchasing each product within the timeframe
    2. Calculate expected proceeds based on historical data for each product
    """
    
    def __init__(self, product_dim_df=None):
        """
        Initialize the model with optional product dimension data.
        
        Args:
            product_dim_df: DataFrame containing product pricing information
        """
        self.product_dim_df = product_dim_df
        self.d8_product_model = None
        self.d100_product_model = None
        self.categorical_features = [
            'channel_group', 'marketing_network_id', 'target_market',
            'signup_country_group', 'signup_client_platform'
        ]
        self.numerical_features = [
            'eur_marketing_spend', 'impressions', 'clicks',
            'started_content', 'n_content_starts', 'finished_content', 
            'n_content_finishes', 'space_user'
        ]
        self.features = self.categorical_features + self.numerical_features
        self.d8_preprocessor = None
        self.d100_preprocessor = None
        self.d8_product_classes = None
        self.d100_product_classes = None
        self.d8_product_proceeds = {}
        self.d100_product_proceeds = {}
        
    def _create_preprocessor(self):
        """Create a preprocessing pipeline for the features."""
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        return ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ])
    
    def _prepare_product_target(self, df: pd.DataFrame, timeframe: str) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare target variable for product prediction.
        
        Args:
            df: Training DataFrame
            timeframe: 'd8' or 'd100'
            
        Returns:
            Tuple of (target array, product classes)
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Determine which product was purchased based on proceeds
        proceeds_col = f'eur_proceeds_{timeframe}'
        
        # For users who didn't purchase anything, set product_name to 'no_purchase'
        df_copy.loc[df_copy[proceeds_col] <= 0, 'product_name'] = 'no_purchase'
        
        # For users with missing product_name but positive proceeds, set to 'unknown_product'
        df_copy.loc[(df_copy[proceeds_col] > 0) & 
                    (df_copy['product_name'].isna()), 'product_name'] = 'unknown_product'
        
        # Fill any remaining NaN values with 'no_purchase'
        df_copy['product_name'].fillna('no_purchase', inplace=True)
        
        # Get unique product names including 'no_purchase'
        product_classes = sorted(df_copy['product_name'].unique())
        
        # Create target array - IMPORTANT: Convert product names to integer labels
        # This is the key fix - we need to use integer labels for classification
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_copy['product_name'].values)
        
        # Store the label encoder for later use
        if timeframe == 'd8':
            self.d8_label_encoder = label_encoder
        else:
            self.d100_label_encoder = label_encoder
        
        return y, product_classes
    
    def _calculate_product_proceeds(self, df: pd.DataFrame, timeframe: str, product_classes: List[str]) -> Dict:
        """
        Calculate average proceeds for each product.
        
        Args:
            df: Training DataFrame
            timeframe: 'd8' or 'd100'
            product_classes: List of product classes
            
        Returns:
            Dictionary mapping product names to average proceeds
        """
        proceeds_col = f'eur_proceeds_{timeframe}'
        product_proceeds = {}
        
        for product in product_classes:
            if product == 'no_purchase':
                product_proceeds[product] = 0.0
                continue
                
            # Get users who purchased this product
            product_users = df[(df['product_name'] == product) & (df[proceeds_col] > 0)]
            
            if len(product_users) > 0:
                # Calculate average proceeds for this product
                avg_proceeds = product_users[proceeds_col].mean()
                product_proceeds[product] = avg_proceeds
            elif self.product_dim_df is not None:
                # Try to get price from product dimension table
                product_match = self.product_dim_df[
                    self.product_dim_df['product_name'] == product
                ]
                if not product_match.empty:
                    product_proceeds[product] = product_match.iloc[0]['price']
                else:
                    # If product not found in dimension table, use global average
                    global_avg = df[df[proceeds_col] > 0][proceeds_col].mean()
                    product_proceeds[product] = global_avg if not np.isnan(global_avg) else 0.0
            else:
                # If no product dimension table, use global average
                global_avg = df[df[proceeds_col] > 0][proceeds_col].mean()
                product_proceeds[product] = global_avg if not np.isnan(global_avg) else 0.0
        
        return product_proceeds
        
    def fit(self, training_d8_df, training_d100_df):
        """
        Train the model on historical data.
        
        Args:
            training_d8_df: DataFrame with historical user data for d8 predictions
            training_d100_df: DataFrame with historical user data for d100 predictions
        """
        # Extract features
        X_d8 = training_d8_df[self.features]
        X_d100 = training_d100_df[self.features]
        
        # Prepare targets for product prediction
        y_d8_product, self.d8_product_classes = self._prepare_product_target(training_d8_df, 'd8')
        y_d100_product, self.d100_product_classes = self._prepare_product_target(training_d100_df, 'd100')
        
        # Create and fit preprocessors
        self.d8_preprocessor = self._create_preprocessor()
        self.d100_preprocessor = self._create_preprocessor()
        
        X_d8_processed = self.d8_preprocessor.fit_transform(X_d8)
        X_d100_processed = self.d100_preprocessor.fit_transform(X_d100)
        
        # Train product models using XGBoost instead of RandomForest
        self.d8_product_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )
        
        self.d100_product_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )
        
        self.d8_product_model.fit(X_d8_processed, y_d8_product)
        self.d100_product_model.fit(X_d100_processed, y_d100_product)
        
        # Calculate average proceeds for each product
        self.d8_product_proceeds = self._calculate_product_proceeds(training_d8_df, 'd8', self.d8_product_classes)
        self.d100_product_proceeds = self._calculate_product_proceeds(training_d100_df, 'd100', self.d100_product_classes)
        
    def _calculate_expected_proceeds(self, product_probs, product_proceeds):
        """
        Calculate expected proceeds based on product probabilities and average proceeds.
        
        Args:
            product_probs: Array of product probabilities
            product_proceeds: Dictionary mapping product names to average proceeds
            
        Returns:
            Expected proceeds
        """
        expected_proceeds = 0.0
        
        for i, product in enumerate(product_proceeds.keys()):
            if product != 'no_purchase':
                prob = product_probs[i]
                proceeds = product_proceeds[product]
                expected_proceeds += prob * proceeds
        
        return expected_proceeds
    
    def predict(self, inference_df):
        """
        Predict expected proceeds for lag purchase users.
        
        Args:
            inference_df: DataFrame containing lag purchase users for inference
            
        Returns:
            DataFrame with predictions
        """
        # Make a copy of the inference data
        result_df = inference_df.copy()
        
        # Extract features for prediction
        X = result_df[self.features]
        
        # D8 predictions
        if self.d8_preprocessor is not None and self.d8_product_model is not None:
            # Preprocess features
            X_d8_processed = self.d8_preprocessor.transform(X)
            
            # Predict product probabilities
            d8_product_probs = self.d8_product_model.predict_proba(X_d8_processed)
            
            # Calculate expected proceeds for d8
            d8_expected_proceeds = []
            for probs in d8_product_probs:
                # Map probabilities back to product classes using the indices
                expected_proceed = 0.0
                for i, prob in enumerate(probs):
                    # Get the product class for this index
                    if i < len(self.d8_product_classes):
                        product = self.d8_product_classes[i]
                        if product != 'no_purchase':
                            proceeds = self.d8_product_proceeds[product]
                            expected_proceed += prob * proceeds
                
                d8_expected_proceeds.append(expected_proceed)
            
            result_df['expected_proceeds_d8'] = d8_expected_proceeds
        else:
            raise ValueError("D8 model not trained. Call fit() before predict().")
        
        # D100 predictions
        if self.d100_preprocessor is not None and self.d100_product_model is not None:
            # Preprocess features
            X_d100_processed = self.d100_preprocessor.transform(X)
            
            # Predict product probabilities
            d100_product_probs = self.d100_product_model.predict_proba(X_d100_processed)
            
            # Calculate expected proceeds for d100
            d100_expected_proceeds = []
            for probs in d100_product_probs:
                # Map probabilities back to product classes using the indices
                expected_proceed = 0.0
                for i, prob in enumerate(probs):
                    # Get the product class for this index
                    if i < len(self.d100_product_classes):
                        product = self.d100_product_classes[i]
                        if product != 'no_purchase':
                            proceeds = self.d100_product_proceeds[product]
                            expected_proceed += prob * proceeds
                
                d100_expected_proceeds.append(expected_proceed)
            
            result_df['expected_proceeds_d100'] = d100_expected_proceeds
        else:
            raise ValueError("D100 model not trained. Call fit() before predict().")
        
        # Ensure no negative values in predictions
        result_df['expected_proceeds_d8'] = result_df['expected_proceeds_d8'].clip(lower=0)
        result_df['expected_proceeds_d100'] = result_df['expected_proceeds_d100'].clip(lower=0)
        
        # Validate: D8 should be >= D0
        if 'eur_proceeds_d0' in result_df.columns:
            d0_d8_violations = (result_df['expected_proceeds_d8'] < result_df['eur_proceeds_d0']).sum()
            if d0_d8_violations > 0:
                print(f"Lag model: Found {d0_d8_violations} cases where D8 < D0. Fixing...")
                result_df['expected_proceeds_d8'] = result_df[['eur_proceeds_d0', 'expected_proceeds_d8']].max(axis=1)
        
        # Validate: D100 should be >= D8
        d8_d100_violations = (result_df['expected_proceeds_d100'] < result_df['expected_proceeds_d8']).sum()
        if d8_d100_violations > 0:
            print(f"Lag model: Found {d8_d100_violations} cases where D100 < D8. Fixing...")
            result_df['expected_proceeds_d100'] = result_df[['expected_proceeds_d8', 'expected_proceeds_d100']].max(axis=1)
        
        return result_df
