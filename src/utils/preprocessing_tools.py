from src.data.csv_tools import csv_to_dataframe, processed_to_csv
from src.data.pickle_tools import save_to_pickle

def concatenate_data(set_name: str) -> None:
    """
    Concatenate reviews and users dataframes for a given set_name.

    Parameters:
        set_name (str): The set name. It can be 'train' or 'val'
    """
    users_df = csv_to_dataframe(set_name, "users")
    matches_df = csv_to_dataframe(set_name, "matches")
    matches_df = matches_df.drop(columns="accommodation_id")
    processed_df = users_df.merge(matches_df, on="user_id")

    reviews_df = csv_to_dataframe(set_name, "reviews")
    # Drop accommodation_id column
    reviews_df = reviews_df.drop(columns="accommodation_id")
    processed_df = processed_df.merge(reviews_df, on="review_id")
    return processed_df

def create_concatenated_set(set_name: str) -> None:
    """
    Concatenate data for a given set_name.

    Parameters:
        set_name (str): The set name. It can be 'train' or 'val'
    """
    processed_set = concatenate_data(set_name)
    processed_to_csv(processed_set, set_name)
    print(f"{set_name} concatenated and saved")
    return processed_set

def create_accommodation_reviews(set_name) -> None:
    reviews_df = csv_to_dataframe(set_name, "reviews")
    accommodation_reviews = reviews_df.groupby("accommodation_id").agg({"review_id": " ".join}).reset_index()
    # Convert to dictionary
    accommodation_reviews = accommodation_reviews.to_dict(orient="records")
    accommodation_reviews = {accommodation_review["accommodation_id"]: accommodation_review["review_id"].split() for accommodation_review in accommodation_reviews}
    save_to_pickle(accommodation_reviews, f"{set_name}_reviews_grouped_by_accommodation")
    print(f"Accommodation reviews for {set_name} set aggregated")
    return accommodation_reviews
    
if __name__ == "__main__":
    for set_name in ["train", "val"]:
        concatenate_set(set_name)
        accommodation_reviews(set_name)
    accommodation_reviews("test")