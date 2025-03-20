import pandas as pd
import numpy as np
from typing import List

class PredictionProcessor:
    """
    Class for processing and aggregating predictions at various levels.
    
    This class handles:
    1. Combining predictions from different sources
    2. Aggregating predictions by specified grouping columns
    3. Calculating additional metrics like ROI and CPA
    """
    
    def __init__(self):
        """
        Initialize the PredictionProcessor.
        """
        self.date_column = "report_date"
        
        self.grouping_columns = [
            "report_date", "channel_group", "marketing_network_id", "account_id",
            "campaign_name", "campaign_id", "adgroup_name", "adgroup_id", "target_market"
        ]
    
    def process_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and aggregate predictions.
        
        Args:
            predictions_df: DataFrame containing prediction data
            
        Returns:
            Aggregated DataFrame with calculated metrics
        """
        # Make a copy to avoid modifying the original
        df = predictions_df.copy()
        
        # Convert columns to appropriate types
        numeric_columns = ['eur_marketing_spend', 'eur_proceeds_d0', 'eur_proceeds_d8', 
                          'eur_proceeds_d100', 'expected_proceeds_d8', 'expected_proceeds_d100']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure all grouping columns exist in the DataFrame
        valid_grouping = [col for col in self.grouping_columns if col in df.columns]
        
        # Create aggregation dictionary
        agg_dict = {
            'user_id': 'count',
            'eur_marketing_spend': 'sum',
        }
        
        # Add proceeds columns if they exist
        if 'eur_proceeds_d0' in df.columns:
            agg_dict['eur_proceeds_d0'] = 'sum'
        if 'eur_proceeds_d8' in df.columns:
            agg_dict['eur_proceeds_d8'] = 'sum'
        if 'eur_proceeds_d100' in df.columns:
            agg_dict['eur_proceeds_d100'] = 'sum'
        
        # Add expected proceeds columns if they exist
        if 'expected_proceeds_d8' in df.columns:
            agg_dict['expected_proceeds_d8'] = 'sum'
        if 'expected_proceeds_d100' in df.columns:
            agg_dict['expected_proceeds_d100'] = 'sum'
        
        # Perform aggregation
        aggregated_df = df.groupby(valid_grouping).agg(agg_dict).reset_index()
        
        # Rename columns for clarity
        aggregated_df = aggregated_df.rename(columns={
            'user_id': 'total_users',
            'eur_marketing_spend': 'total_spend'
        })
        
        # Calculate ROI and CPA metrics
        if 'total_spend' in aggregated_df.columns:
            # Calculate ROI (Return on Investment) for expected proceeds
            if 'expected_proceeds_d8' in aggregated_df.columns:
                # Use pandas division with fill_value to handle division by zero
                aggregated_df['exp_roi_d8'] = (aggregated_df['expected_proceeds_d8'] / 
                                          aggregated_df['total_spend'].replace(0, np.nan)).fillna(0)
                
            if 'expected_proceeds_d100' in aggregated_df.columns:
                aggregated_df['exp_roi_d100'] = (aggregated_df['expected_proceeds_d100'] / 
                                            aggregated_df['total_spend'].replace(0, np.nan)).fillna(0)
            
            # Calculate ROI for actual proceeds if available
            if 'eur_proceeds_d8' in aggregated_df.columns:
                aggregated_df['actual_roi_d8'] = (aggregated_df['eur_proceeds_d8'] / 
                                                 aggregated_df['total_spend'].replace(0, np.nan)).fillna(0)
                
            if 'eur_proceeds_d100' in aggregated_df.columns:
                aggregated_df['actual_roi_d100'] = (aggregated_df['eur_proceeds_d100'] / 
                                                   aggregated_df['total_spend'].replace(0, np.nan)).fillna(0)
            
            # Calculate CPA (Cost Per Acquisition)
            aggregated_df['cpa'] = (aggregated_df['total_spend'] / 
                                   aggregated_df['total_users'].replace(0, np.nan)).fillna(aggregated_df['total_spend'])
        
        # Final check to ensure no NaN values
        for col in aggregated_df.columns:
            if aggregated_df[col].isna().any():
                print(f"Filling NaN values in {col}")
                aggregated_df[col] = aggregated_df[col].fillna(0)
        
        return aggregated_df
    