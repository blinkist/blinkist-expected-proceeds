import pandas as pd
import numpy as np
from typing import Dict

class CountryGrouper:
    """
    Utility class to assign country groups based on historical performance.
    """
    
    def __init__(self, max_proceeds=1000):
        """
        Initialize the CountryGrouper.
        
        Args:
            max_proceeds: Maximum proceeds value to consider (to filter outliers)
        """
        self.max_proceeds = max_proceeds
        
    def get_signup_country_clusters(self, df: pd.DataFrame) -> Dict:
        """
        Generate country clusters based on historical performance.
        
        Args:
            df: DataFrame containing user data with signup_country, us_state, and eur_proceeds_d8
            
        Returns:
            Dictionary mapping countries/states to country groups
        """
        temp_df = df.copy()
        temp_df["temp_country"] = np.where(temp_df.signup_country == "US",
                                           (temp_df.signup_country + "-" + temp_df.us_state), temp_df.signup_country)

        # base clustering on last 6 months of data
        six_months_ago = pd.to_datetime("today") - pd.Timedelta(180, "day")

        # US data
        us_df = temp_df.loc[(temp_df.report_date >= six_months_ago) & (temp_df.signup_country == "US") & (
            temp_df.eur_proceeds_d8 < self.max_proceeds)].groupby("temp_country", as_index=False).agg(
            mean_proceeds=pd.NamedAgg(column="eur_proceeds_d8", aggfunc="mean"),
            n=pd.NamedAgg(column="us_state", aggfunc="count"))

        us_df["signup_country_group"] = np.where(us_df.temp_country == "US-unknown", "us_other",
                                                 np.where(us_df.n < 10, "us_other",
                                                          np.where(us_df.mean_proceeds <= 3, "us_low", np.where(
                                                              (us_df.mean_proceeds > 3) & (us_df.mean_proceeds <= 5),
                                                              "us_med", np.where((us_df.mean_proceeds > 5) & (
                                                                      us_df.mean_proceeds <= 6.5), "us_mod",
                                                                                 np.where(us_df.mean_proceeds > 6.5,
                                                                                          "us_high",
                                                                                          "us_other"))))))

        us_dict = dict(zip(us_df.temp_country, us_df.signup_country_group))

        row_df = temp_df.loc[(temp_df.report_date >= six_months_ago) & (
            ~temp_df.signup_country.isin(["US", "DE", "AT", "CA", "AU", "GB", "IN", "A1", "ZZ", "WW"])) & (
            temp_df.eur_proceeds_d8 < self.max_proceeds)].groupby("temp_country", as_index=False).agg(
            mean_proceeds=pd.NamedAgg(column="eur_proceeds_d8", aggfunc="mean"),
            n=pd.NamedAgg(column="eur_proceeds_d8", aggfunc="count"))

        row_df["signup_country_group"] = np.where(row_df.n < 10, "other",
                                                  np.where(row_df.mean_proceeds <= 1.1, "row_verylow", np.where(
                                                      (row_df.mean_proceeds > 1.1) & (row_df.mean_proceeds <= 2.5),
                                                      "row_low", np.where(
                                                          (row_df.mean_proceeds > 2.5) & (row_df.mean_proceeds <= 4),
                                                          "row_mod", np.where(
                                                              (row_df.mean_proceeds > 4) & (row_df.mean_proceeds <= 7),
                                                              "row_high",
                                                              np.where(row_df.mean_proceeds > 7, "row_veryhigh",
                                                                       "other"))))))

        row_dict = dict(zip(row_df.temp_country, row_df.signup_country_group))

        country_clusters = {**us_dict, **row_dict}
        country_clusters.update(
            {"IN": "in", "DE": "de_at", "AT": "de_at", "GB": "gb_ca_au", "CA": "gb_ca_au", "AU": "gb_ca_au"})

        return country_clusters

    def assign_signup_country_group(self, df: pd.DataFrame, country_clusters: Dict = None) -> pd.DataFrame:
        """
        Assign country groups to a DataFrame based on country clusters.
        
        Args:
            df: DataFrame containing user data with signup_country and us_state
            country_clusters: Dictionary mapping countries/states to country groups
                             If None, will generate clusters from the data
                             
        Returns:
            DataFrame with added signup_country_group column
        """
        result_df = df.copy()
        
        # Generate country clusters if not provided
        if country_clusters is None:
            country_clusters = self.get_signup_country_clusters(result_df)
        
        # Create temporary column for mapping
        result_df["temp_country"] = np.where(result_df.signup_country == "US",
                                          (result_df.signup_country + "-" + result_df.us_state),
                                          result_df.signup_country)

        # Map country clusters
        result_df["signup_country_group"] = result_df.temp_country.map(country_clusters)
        
        # Fill remaining unknown countries with "other"
        result_df["signup_country_group"] = np.where((~result_df.user_id.isna()) & (result_df.signup_country_group.isna()),
                                              "other", result_df.signup_country_group)

        # Drop temporary column
        result_df = result_df.drop(columns="temp_country")

        return result_df

def add_signup_country_group(df: pd.DataFrame, reference_df: pd.DataFrame = None, max_proceeds: float = 1000) -> pd.DataFrame:
    """
    Add signup_country_group column to a DataFrame.
    
    Args:
        df: DataFrame to add signup_country_group to
        reference_df: Reference DataFrame to use for generating country clusters
                     If None, will use the input DataFrame
        max_proceeds: Maximum proceeds value to consider (to filter outliers)
        
    Returns:
        DataFrame with added signup_country_group column
    """
    country_grouper = CountryGrouper(max_proceeds=max_proceeds)
    
    # If reference DataFrame provided, use it to generate country clusters
    if reference_df is not None:
        country_clusters = country_grouper.get_signup_country_clusters(reference_df)
        return country_grouper.assign_signup_country_group(df, country_clusters)
    
    # Otherwise, generate clusters from the input DataFrame
    return country_grouper.assign_signup_country_group(df) 