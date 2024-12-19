from os import getcwd, mkdir
from os.path import join as os_path_join, exists as os_path_exists

WORKING_DIRECTORY = getcwd()

# INPUTS

_DATA_DIRECTORY_NAME = "data"
_RAW_DIRECTORY_NAME = "raw"
_PROCESSED_DIRECTORY_NAME = "processed"

DATA_DIRECTORY_PATH = os_path_join(WORKING_DIRECTORY, _DATA_DIRECTORY_NAME)
RAW_DIRECTORY_PATH = os_path_join(DATA_DIRECTORY_PATH, _RAW_DIRECTORY_NAME)
PROCESSED_DIRECTORY_PATH = os_path_join(DATA_DIRECTORY_PATH, _PROCESSED_DIRECTORY_NAME)

def get_raw_file_path(set_name: str, file_type: str) -> str:
    return os_path_join(RAW_DIRECTORY_PATH, f"{set_name}_{file_type}.csv")

def get_processed_file_path(file_name: str) -> str:
    if not os_path_exists(PROCESSED_DIRECTORY_PATH):
        mkdir(PROCESSED_DIRECTORY_PATH)
    return os_path_join(PROCESSED_DIRECTORY_PATH, f"{file_name}.csv")

# OUTPUTS

_OUTPUTS_DIRECTORY_NAME = "out"
_RESULTS_FILE_NAME = "submission.csv"

OUTPUTS_DIRECTORY_PATH = os_path_join(WORKING_DIRECTORY, _OUTPUTS_DIRECTORY_NAME)
RESULTS_FILE_PATH = os_path_join(OUTPUTS_DIRECTORY_PATH, _RESULTS_FILE_NAME)

PICKLE_EXTENSION = ".pickle"
KERAS_EXTENSION = ".keras"

def _get_output_file_path(file_name, file_extension):
    if not os_path_exists(OUTPUTS_DIRECTORY_PATH):
        mkdir(OUTPUTS_DIRECTORY_PATH)
    return os_path_join(OUTPUTS_DIRECTORY_PATH, f"{file_name}{file_extension}")

def get_pickle_file_path(file_name: str) -> str:
    return _get_output_file_path(file_name, PICKLE_EXTENSION)

def get_keras_file_path(file_name: str) -> str:
    return _get_output_file_path(file_name, KERAS_EXTENSION)