import pandas as pd
import numpy as np

class DirectPurchasePredictionModel:
    """
    Model to predict expected_proceeds_d100 for direct purchasers using d8 to d100 multipliers.
    """
    
    def __init__(self):
        """Initialize the model for direct purchasers."""
        self.multipliers = None
        
    def fit(self, training_d100_df: pd.DataFrame):
        """Calculate d8 to d100 multipliers from training data."""
        # Calculate multipliers at product_name and signup_country_group level
        multipliers = training_d100_df.groupby(['product_name', 'signup_country_group']).agg({
            'eur_proceeds_d8': 'sum',
            'eur_proceeds_d100': 'sum'
        }).reset_index()
        
        # Calculate the ratio
        multipliers['d100_to_d8_ratio'] = multipliers['eur_proceeds_d100'] / multipliers['eur_proceeds_d8'].replace(0, np.nan)
        
        # Ensure multiplier is not less than 1
        multipliers['d100_to_d8_ratio'] = multipliers['d100_to_d8_ratio'].apply(lambda x: max(x, 1))
        
        # Handle cases with too few entries by falling back to product_name level
        min_entries = 100  # Define a threshold for minimum entries
        multipliers['entry_count'] = training_d100_df.groupby(['product_name', 'signup_country_group'])['eur_proceeds_d8'].transform('count')
        
        # Calculate product_name level multipliers
        product_level_multipliers = training_d100_df.groupby('product_name').agg({
            'eur_proceeds_d8': 'sum',
            'eur_proceeds_d100': 'sum'
        }).reset_index()
        product_level_multipliers['d100_to_d8_ratio'] = product_level_multipliers['eur_proceeds_d100'] / product_level_multipliers['eur_proceeds_d8'].replace(0, np.nan)
        
        # Ensure product level multiplier is not less than 1
        product_level_multipliers['d100_to_d8_ratio'] = product_level_multipliers['d100_to_d8_ratio'].apply(lambda x: max(x, 1))
        
        # Merge product level multipliers
        multipliers = multipliers.merge(product_level_multipliers[['product_name', 'd100_to_d8_ratio']], on='product_name', suffixes=('', '_product'))
        
        # Use product level multiplier if entry count is below threshold
        multipliers['d100_to_d8_ratio'] = np.where(
            multipliers['entry_count'] < min_entries,
            multipliers['d100_to_d8_ratio_product'],
            multipliers['d100_to_d8_ratio']
        )
        
        self.multipliers = multipliers[['product_name', 'signup_country_group', 'd100_to_d8_ratio']]
    
    def predict(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected proceeds for direct purchase users using d8 to d100 multipliers.
        
        Args:
            inference_df: DataFrame containing direct purchase users for inference
            
        Returns:
            DataFrame with predictions
        """
        # Create a copy of the inference dataframe to avoid modifying the original
        result_df = inference_df.copy()
        
        # Calculate average ratio across all products as fallback
        avg_ratio = self.multipliers['d100_to_d8_ratio'].mean()
        
        # Create product-level multipliers dictionary for faster lookup
        product_multipliers = self.multipliers.groupby('product_name')['d100_to_d8_ratio'].mean().to_dict()
        
        # Create a lookup dictionary for product and country group combinations
        product_country_multipliers = self.multipliers.set_index(['product_name', 'signup_country_group'])['d100_to_d8_ratio'].to_dict()
        
        # Apply the multipliers with fallback logic
        def get_multiplier(row):
            # First try product and country combination
            key = (row['product_name'], row['signup_country_group'])
            if key in product_country_multipliers:
                return product_country_multipliers[key]
            # Fall back to product level
            elif row['product_name'] in product_multipliers:
                return product_multipliers[row['product_name']]
            # Fall back to average ratio
            else:
                return avg_ratio
        
        # Add expected_proceeds_d8 column
        result_df['expected_proceeds_d8'] = result_df['eur_proceeds_d8']
        
        # Apply the multiplier function to get d100_to_d8_ratio
        result_df['d100_to_d8_ratio'] = result_df.apply(get_multiplier, axis=1)
        
        # Calculate expected_proceeds_d100
        result_df['expected_proceeds_d100'] = result_df['eur_proceeds_d8'] * result_df['d100_to_d8_ratio']
        
        return result_df