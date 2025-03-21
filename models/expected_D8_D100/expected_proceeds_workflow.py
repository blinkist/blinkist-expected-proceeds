import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import logging

# Import custom modules
from data_utils import split_data_by_date, split_data_by_user_type, load_data
from models.expected_D8_D100.models.trial_predictions import TrialPredictionModel
from models.expected_D8_D100.models.direct_purchase_predictions import DirectPurchasePredictionModel
from models.expected_D8_D100.models.lag_purchase_predictions import LagPurchasePredictionModel

# Connect to Snowflake
from snowflake.snowpark.context import get_active_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ExpectedProceedsWorkflow')

# Constants
TRAINING_WINDOW_DAYS = 180  # Use 1/2 year of training data
OUTPUT_TABLE = "BLINKIST_DEV.DBT_MJAAMA.USER_LEVEL_EXPECTED_PROCEEDS_PREDICTIONS"


def train_models(trial_training_d8, trial_training_d100, 
                day0_payers_training_d8, day0_payers_training_d100, 
                other_training_d8, other_training_d100,
                product_df):
    """
    Train prediction models for different user types.
    
    Args:
        trial_training_*: Training data for trial users
        day0_payers_training_*: Training data for day0 payers
        other_training_*: Training data for other users
        product_df: Product dimension data
        
    Returns:
        Tuple of (trial_model, direct_model, lag_model)
    """
    # Train trial model
    logger.info("Training trial model")
    trial_model = TrialPredictionModel(product_dim_df=product_df)
    trial_model.fit(trial_training_d8, trial_training_d100)
    
    # Train direct purchase model
    logger.info("Training direct purchase model")
    direct_model = DirectPurchasePredictionModel()
    direct_model.fit(day0_payers_training_d100)
    
    # Train lag purchase model
    logger.info("Training lag purchase model")
    lag_model = LagPurchasePredictionModel(product_dim_df=product_df)
    lag_model.fit(other_training_d8, other_training_d100)
    
    return trial_model, direct_model, lag_model


def make_predictions(trial_model, direct_model, lag_model,
                    trial_inference, day0_payers_inference, other_inference):
    """
    Make predictions for different user types.
    
    Args:
        *_model: Trained models for different user types
        *_inference: Inference data for different user types
        
    Returns:
        Combined DataFrame with predictions
    """
    predictions = []
    
    # Trial user predictions
    if not trial_inference.empty:
        logger.info(f"Making predictions for {len(trial_inference)} trial users")
        trial_predictions = trial_model.predict(trial_inference)
        trial_predictions['user_type'] = 'trial'
        predictions.append(trial_predictions)
    
    # Direct purchase predictions
    if not day0_payers_inference.empty:
        logger.info(f"Making predictions for {len(day0_payers_inference)} direct purchasers")
        direct_predictions = direct_model.predict(day0_payers_inference)
        direct_predictions['user_type'] = 'day0_payer'
        predictions.append(direct_predictions)
    
    # Lag purchase predictions
    if not other_inference.empty:
        logger.info(f"Making predictions for {len(other_inference)} lag purchasers")
        lag_predictions = lag_model.predict(other_inference)
        lag_predictions['user_type'] = 'lag_payer'
        predictions.append(lag_predictions)
    
    # Combine predictions
    if not predictions:
        logger.warning("No predictions generated - inference data may be empty")
        return None
    
    return pd.concat(predictions, ignore_index=False)


def save_predictions(session, predictions_df, inference_date):
    """
    Save predictions to Snowflake.
    
    Args:
        session: Snowflake session
        predictions_df: DataFrame with predictions
        inference_date: Date of inference
    """
    # Add inference date column
    predictions_df['inference_date'] = inference_date
    
    # Convert to Snowpark DataFrame
    snowpark_df = session.create_dataframe(predictions_df)
    
    # Save to Snowflake table
    logger.info(f"Saving {len(predictions_df)} predictions to {OUTPUT_TABLE}")
    snowpark_df.write.mode("append").save_as_table(OUTPUT_TABLE)
    
    logger.info(f"Predictions for {inference_date} saved successfully")


def run_workflow(session, inference_date=None):
    """
    Run the expected proceeds prediction workflow
    
    Args:
        session: Snowflake session
        inference_date: Optional date string in format 'YYYY-MM-DD'
                       If None, uses yesterday's date
                       
    Returns:
        DataFrame with predictions
    """
    # Set inference date to yesterday if not provided
    if inference_date is None:
        inference_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    logger.info(f"Running workflow for inference date: {inference_date}")
    
    # 1. Load Data (country groups are added in the load_data function)
    logger.info("Loading data from Snowflake")
    input_df, product_df = load_data(session)
    
    # 2. Split Data by Date
    logger.info("Splitting data by date")
    inference_df, training_d8_df, training_d100_df = split_data_by_date(
        input_df,
        inference_date=inference_date,
        training_window_days=TRAINING_WINDOW_DAYS,
        date_column='report_date'
    )
    
    # 3. Split Data by User Type
    logger.info("Splitting data by user type")
    trial_inference, day0_payers_inference, other_inference = split_data_by_user_type(inference_df)
    trial_training_d8, day0_payers_training_d8, other_training_d8 = split_data_by_user_type(training_d8_df)
    trial_training_d100, day0_payers_training_d100, other_training_d100 = split_data_by_user_type(training_d100_df)
    
    # 4. Train Models
    trial_model, direct_model, lag_model = train_models(
        trial_training_d8, trial_training_d100,
        day0_payers_training_d8, day0_payers_training_d100,
        other_training_d8, other_training_d100,
        product_df
    )
    
    # 5. Make Predictions
    predictions_df = make_predictions(
        trial_model, direct_model, lag_model,
        trial_inference, day0_payers_inference, other_inference
    )
    
    if predictions_df is None:
        return None
    
    # 6. Save Predictions
    save_predictions(session, predictions_df, inference_date)
    
    return predictions_df


if __name__ == "__main__":
    # Get Snowflake session
    session = get_active_session()
    
    # Check if inference date is provided as command line argument
    inference_date = None
    if len(sys.argv) > 1:
        inference_date = sys.argv[1]
    
    # Run workflow
    run_workflow(session, inference_date) 