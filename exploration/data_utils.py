import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union

def split_data_by_date(
    df: pd.DataFrame,
    inference_date: Union[str, datetime],
    training_window_days: int = 365,
    date_column: str = 'report_date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into inference, training_d8, and training_d100 datasets based on dates.
    
    Args:
        df: Input DataFrame containing user data
        inference_date: Date for inference data (users who signed up on this date)
                       Can be a datetime object or a string in format 'YYYY-MM-DD'
        training_window_days: Number of days to include in training data (default: 365)
        date_column: Name of the column containing signup dates (default: 'report_date')
        
    Returns:
        Tuple of (inference_df, training_d8_df, training_d100_df)
    """
    # Convert string date to datetime if needed
    if isinstance(inference_date, str):
        inference_date = datetime.strptime(inference_date, '%Y-%m-%d')
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Calculate cutoff dates
    d8_cutoff = inference_date - timedelta(days=8)
    d100_cutoff = inference_date - timedelta(days=100)
    
    # Calculate training window start dates
    d8_training_start = d8_cutoff - timedelta(days=training_window_days)
    d100_training_start = d100_cutoff - timedelta(days=training_window_days)
    
    # Split data
    inference_df = df[df[date_column] == inference_date].copy()
    
    # Training data for d8 predictions: users who signed up between d8_training_start and d8_cutoff
    # These users have had at least 8 days to generate proceeds_d8 data
    training_d8_df = df[
        (df[date_column] >= d8_training_start) & 
        (df[date_column] < d8_cutoff)
    ].copy()
    
    # Training data for d100 predictions: users who signed up between d100_training_start and d100_cutoff
    # These users have had at least 100 days to generate proceeds_d100 data
    training_d100_df = df[
        (df[date_column] >= d100_training_start) & 
        (df[date_column] < d100_cutoff)
    ].copy()
    
    return inference_df, training_d8_df, training_d100_df

def split_data_for_model_training(
    df: pd.DataFrame,
    reference_date: Optional[Union[str, datetime]] = None,
    training_window_days: int = 365,
    date_column: str = 'report_date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training_d8 and training_d100 datasets for model training.
    
    Args:
        df: Input DataFrame containing user data
        reference_date: Reference date for calculating cutoffs (default: max date in DataFrame)
                       Can be a datetime object or a string in format 'YYYY-MM-DD'
        training_window_days: Number of days to include in training data (default: 365)
        date_column: Name of the column containing signup dates (default: 'report_date')
        
    Returns:
        Tuple of (training_d8_df, training_d100_df)
    """
    # Convert string date to datetime if needed
    if isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # If no reference date provided, use the maximum date in the DataFrame
    if reference_date is None:
        reference_date = df[date_column].max()
    
    # Calculate cutoff dates
    d8_cutoff = reference_date - timedelta(days=8)
    d100_cutoff = reference_date - timedelta(days=100)
    
    # Calculate training window start dates
    d8_training_start = d8_cutoff - timedelta(days=training_window_days)
    d100_training_start = d100_cutoff - timedelta(days=training_window_days)
    
    # Training data for d8 predictions: users who signed up between d8_training_start and d8_cutoff
    training_d8_df = df[
        (df[date_column] >= d8_training_start) & 
        (df[date_column] < d8_cutoff)
    ].copy()
    
    # Training data for d100 predictions: users who signed up between d100_training_start and d100_cutoff
    training_d100_df = df[
        (df[date_column] >= d100_training_start) & 
        (df[date_column] < d100_cutoff)
    ].copy()
    
    return training_d8_df, training_d100_df

def split_data_by_user_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into three groups:
    1. trial_users_df -> is_trial_subscription == 1
    2. day0_payers_df -> is_trial_subscription == 0 AND eur_proceeds_d0 > 0
    3. other_users_df -> is_trial_subscription == 0 AND eur_proceeds_d0 == 0
    
    Args:
        df: DataFrame containing user data
        
    Returns:
        Tuple of (trial_users_df, day0_payers_df, other_users_df)
    """
    # Check if we have the original column name or the EUR version
    proceeds_d0_col = 'proceeds_d0' if 'proceeds_d0' in df.columns else 'eur_proceeds_d0'
    
    # 1) trial_users_df -> is_trial_subscription == 1
    trial_users_df = df[df['is_trial_subscription'] == 1].copy()
    
    # 2) day0_payers_df -> is_trial_subscription == 0 AND proceeds_d0 > 0
    day0_payers_df = df[
        (df['is_trial_subscription'] == 0) &
        (df[proceeds_d0_col] > 0)
    ].copy()
    
    # 3) other_users_df -> is_trial_subscription == 0 AND proceeds_d0 == 0
    other_users_df = df[
        (df['is_trial_subscription'] == 0) &
        (df[proceeds_d0_col] == 0)
    ].copy()
    
    return trial_users_df, day0_payers_df, other_users_df 