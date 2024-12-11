from pandas import DataFrame
from os import getcwd, path as os_path

DEFAULT_FILE_NAME = "submission.csv"
DEFAULT_FILE_PATH = os_path.join(getcwd(), DEFAULT_FILE_NAME)

def create_submission(df: DataFrame, path=DEFAULT_FILE_PATH) -> None:
    if 'ID' not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
    try:
        df.to_csv(path, 
                index=False)  # TODO - Check if necessary
        print(f"Submission file created at {path}")
    except Exception as e:
        if path != DEFAULT_FILE_PATH:
            # Try again using the default path
            print(f"Error creating submission file at {path}. Trying again using the default path {DEFAULT_FILE_PATH}")
            create_submission(df)
            return
        raise e