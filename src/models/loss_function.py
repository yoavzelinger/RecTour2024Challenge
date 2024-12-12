from pandas import DataFrame

from src.data.csv_tools import csv_to_dataframe 

def evaluate(
        set_name: str, # train/ val
        results_df: DataFrame
 ) -> float:
    """
    Return MRR@10 score for a prediction

    Parameters:
        results_df (DataFrame): The predictions. Structure: accommodation_id,user_id,review_1,review_2,review_3,...,review_10

    Returns:
        float: MRR@10.
    """
    matches_df = csv_to_dataframe(set_name, "matches")

    assert len(results_df) == len(matches_df), "results and matches should have the same size."
    
    mrr = 0

    for rank, column_name in enumerate(results_df.columns[2: ], 1):
        column_matches_count = (matches_df.values == results_df[["accommodation_id", "user_id", column_name]].values).all(axis=1).sum()
        mrr += column_matches_count * (1.0 / rank)
    
    mrr = mrr / len(matches_df)
    
    return mrr