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

def concatenate_train_val() -> None:
    processed_train = concatenate_data("train")
    processed_to_csv(processed_train, "train")
    print("train concatenated")
    
    processed_val = concatenate_data("val")
    processed_to_csv(processed_val, "val")
    print("val concatenated")

def accommodation_reviews(set_name) -> None:
    reviews_df = csv_to_dataframe(set_name, "reviews")
    accommodation_reviews = reviews_df.groupby("accommodation_id").agg({"review_id": " ".join}).reset_index()
    # Convert to dictionary
    accommodation_reviews = accommodation_reviews.to_dict(orient="records")
    accommodation_reviews = {accommodation_review["accommodation_id"]: accommodation_review["review_id"].split() for accommodation_review in accommodation_reviews}
    save_to_pickle(accommodation_reviews, f"{set_name}_reviews_grouped_by_accommodation")
    return accommodation_reviews
    
if __name__ == "__main__":
    concatenate_train_val()