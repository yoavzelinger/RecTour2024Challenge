from pandas import DataFrame, read_csv

from src.data.directories import get_raw_file_path, get_processed_file_path, RESULTS_FILE_PATH

files_dictionary = {    # {set_name: {file_type: DataFrame}}
    "train": {
        "processed": None,
        "raw": {
            "matches": None,
            "reviews": None,
            "users": None
        }
    },
    "val": {
        "processed": None,
        "raw": {
            "matches": None,
            "reviews": None,
            "users": None
        }
    },
    "test": {
        "raw": {
            "matches": None,
            "reviews": None
        }
    }
}

def csv_to_dataframe(
    set_name: str,
    file_type: str = None
    ) -> DataFrame:
    """
    Read a csv file and return a DataFrame.
    
    Parameters:
        set_name (str): The set name. It can be 'train', 'val' or 'test'.
        file_type (str): The file type. It can be 'matches' (not test), 'reviews' or 'users'. If None, the processed file is returned.
        
    Returns:
        DataFrame: The DataFrame with the data.
    """
    assert set_name in files_dictionary
    file_set = files_dictionary[set_name]
    if file_type is None:
        if file_set["processed"] is None:
            file_set["processed"] = read_csv(get_processed_file_path(set_name))
        return file_set["processed"]
    raw_file_set = file_set["raw"]
    assert file_type in raw_file_set
    if raw_file_set[file_type] is None:
        raw_file_set[file_type] = read_csv(get_raw_file_path(set_name, file_type))
    return raw_file_set[file_type]

def dataframe_to_csv(
    df: DataFrame, 
    path: str) -> None:
    """
    Save a DataFrame to a csv file.
    
    Parameters:
        df (DataFrame): The DataFrame to be saved.
        path (str): The path to save the file.
    """
    df.to_csv(path, 
              index=False)  # TODO - Check if necessary

def processed_to_csv(
    df: DataFrame, 
    set_name: str) -> None:
    """
    Save a processed DataFrame to a csv file.

    Parameters:
        df (DataFrame): The DataFrame to be saved.
        set_name (str): The set name. It can be 'train' or 'val'.
    """
    files_dictionary[set_name]["processed"] = df
    dataframe_to_csv(df, get_processed_file_path(set_name))

def save_submission(
    df: DataFrame) -> None:
    """
    Save a submission DataFrame to a csv file.
    
    Parameters:
        df (DataFrame): The DataFrame to be saved.
    """
    if 'ID' not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
    dataframe_to_csv(df, RESULTS_FILE_PATH)