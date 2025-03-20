import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Import custom modules
from country_utils import add_signup_country_group
from data_utils import split_data_by_date, split_data_by_user_type
from trial_predictions import TrialPredictionModel
from direct_purchase_predictions import DirectPurchasePredictionModel
from lag_purchase_predictions import LagPurchasePredictionModel

# Connect to Snowflake
from snowflake.snowpark.context import get_active_session

def run_workflow(session, inference_date=None):
    """
    Run the expected proceeds prediction workflow
    
    Args:
        session: Snowflake session
        inference_date: Optional date string in format 'YYYY-MM-DD'
                       If None, uses yesterday's date
    """
    # Set inference date to yesterday if not provided
    if inference_date is None:
        inference_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    print(f"Running workflow for inference date: {inference_date}")
    
    # 1. Load Data
    input_query = """
        SELECT 
            *
        FROM BLINKIST_PRODUCTION.CORE_BUSINESS.EXP_PROCEEDS_INPUT
        """
        
    input_df = session.sql(input_query).to_pandas()
    
    product_query = """
        select sku as product_name, price 
        from BLINKIST_PRODUCTION.reference_tables.product_dim
        where is_purchasable;
        """
        
    product_df = session.sql(product_query).to_pandas()
    
    # 2. Add Country Groups
    input_df = add_signup_country_group(input_df)
    
    # 3. Split Data by Date
    training_window_days = 180  # Use 1/2 year of training data
    
    inference_df, training_d8_df, training_d100_df = split_data_by_date(
        input_df,
        inference_date=inference_date,
        training_window_days=training_window_days,
        date_column='report_date'
    )
    
    # 4. Split Data by User Type
    trial_inference, day0_payers_inference, other_inference = split_data_by_user_type(inference_df)
    trial_training_d8, day0_payers_training_d8, other_training_d8 = split_data_by_user_type(training_d8_df)
    trial_training_d100, day0_payers_training_d100, other_training_d100 = split_data_by_user_type(training_d100_df)
    
    # 5. Train Models
    print("Train trial model")
    trial_model = TrialPredictionModel(product_dim_df=product_df)
    trial_model.fit(trial_training_d8, trial_training_d100)
    
    print("Train direct purchase model")
    # Direct purchase model
    direct_model = DirectPurchasePredictionModel()
    direct_model.fit(day0_payers_training_d100)
    
    print("Train lag purchase model")
    # Lag purchase model
    lag_model = LagPurchasePredictionModel(product_dim_df=product_df)
    lag_model.fit(other_training_d8, other_training_d100)
    
    # 6. Make Predictions
    predictions = []
    
    if not trial_inference.empty:
        trial_predictions = trial_model.predict(trial_inference)
        trial_predictions['user_type'] = 'trial'
        predictions.append(trial_predictions)
    
    if not day0_payers_inference.empty:
        direct_predictions = direct_model.predict(day0_payers_inference)
        direct_predictions['user_type'] = 'day0_payer'
        predictions.append(direct_predictions)
    
    if not other_inference.empty:
        lag_predictions = lag_model.predict(other_inference)
        lag_predictions['user_type'] = 'lag_payer'
        predictions.append(lag_predictions)
    
    # 7. Process and Aggregate Predictions
    if not predictions:
        print("No predictions generated - inference data may be empty")
        return
    
    predictions_df = pd.concat(predictions, ignore_index=False)

    
    # Add inference date column
    predictions_df['inference_date'] = inference_date
    
    # Convert to Snowpark DataFrame
    snowpark_df = session.create_dataframe(predictions_df)
    
    # Save to Snowflake table
    table_name = "BLINKIST_DEV.DBT_MJAAMA.DAILY_EXPECTED_PROCEEDS_PREDICTIONS"
    
    # Append to existing table or create new one
    snowpark_df.write.mode("append").save_as_table(table_name)
    
    print(f"Predictions for {inference_date} saved to {table_name}")
    return predictions_df

if __name__ == "__main__":
    # Get Snowflake session
    session = get_active_session()
    
    # Check if inference date is provided as command line argument
    inference_date = '2025-03-12'
    if len(sys.argv) > 1:
        inference_date = sys.argv[1]
    
    # Run workflow
    run_workflow(session, inference_date) 