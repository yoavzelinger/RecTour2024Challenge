from pandas import DataFrame, read_csv
from os.path import is_exists

FILE_EXTENSION = "csv"

files_dictionary = {    # {set_name: {file_type: DataFrame}}
    "train": {
        "matches": None,
        "reviews": None,
        "users": None
    },
    "val": {
        "matches": None,
        "reviews": None,
        "users": None
    },
    "test": {
        "reviews": None,
        "users": None
    }
}

def _load_file_dataframe(
        set_name: str,   # train/ val/ test
        file_type: str # matches (not test)/ reviews/ users
 ) -> str:
    # Get full file name
    file_name = f"{set_name}_{file_type}.{FILE_EXTENSION}"
    if is_exists(file_name):
        return read_csv(file_name)

def get_dataframe(
        set_name: str,   # train/ val/ test
        file_type: str # matches (not test)/ reviews/ users
 ) -> DataFrame:
    assert set_name in files_dictionary
    file_set = files_dictionary[set_name]
    assert file_type in file_set
    if file_set[file_type] is None:
        file_set[file_type] = _load_file_dataframe(set_name, file_type)
    return file_set[file_type]