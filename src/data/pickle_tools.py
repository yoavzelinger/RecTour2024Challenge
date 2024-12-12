import pickle

from pandas import DataFrame, read_csv

from src.data.directories import get_pickle_file_path

def save_to_pickle(obj: object, file_name: str) -> None:
    """
    Save an object to a pickle file.
    
    Parameters:
        obj (object): The object to be saved.
        file_name (str): The file name.
    """
    with open(get_pickle_file_path(file_name), "wb") as file:
        pickle.dump(obj, file)