from src.data.csv_tools import csv_to_dataframe, processed_to_csv

def concatenate_data(set_name: str) -> None:
    """
    Concatenate reviews and users dataframes for a given set_name.

    Parameters:
        set_name (str): The set name. It can be 'train' or 'val'
    """
    users_df = csv_to_dataframe(set_name, "users")
    matches_df = csv_to_dataframe(set_name, "matches")
    processed_df = users_df.merge(matches_df, on="user_id")

    reviews_df = csv_to_dataframe(set_name, "reviews")
    processed_df = processed_df.merge(reviews_df, on="review_id")

    return processed_df

def concatenate_train_val() -> None:
    processed_train = concatenate_data("train")
    processed_to_csv(processed_train, "train")
    print("train concatenated")
    
    processed_val = concatenate_data("val")
    processed_to_csv(processed_val, "val")
    print("val concatenated")

if __name__ == "__main__":
    concatenate_train_val()