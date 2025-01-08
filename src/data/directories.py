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
_PICKELS_DIRECTORY_NAME = "pickels"
_MODELS_DIRECTORY_NAME = "models"
_SUBMISSIONS_DIRECTORY_NAME = "submissions"
_RESULTS_FILE_NAME = "submission"
_RESULTS_EXTENSION = ".csv"

_OUTPUTS_DIRECTORY_PATH = os_path_join(WORKING_DIRECTORY, _OUTPUTS_DIRECTORY_NAME)
_PICKELS_DIRECTORY_PATH = os_path_join(_OUTPUTS_DIRECTORY_PATH, _PICKELS_DIRECTORY_NAME)
_MODELS_DIRECTORY_PATH = os_path_join(_OUTPUTS_DIRECTORY_PATH, _MODELS_DIRECTORY_NAME)
_SUBMISSIONS_DIRECTORY_PATH = os_path_join(WORKING_DIRECTORY, _SUBMISSIONS_DIRECTORY_NAME)

_PICKLE_EXTENSION = ".pickle"
_KERAS_EXTENSION = ".h5"

def _get_file_path(file_name, file_directory, file_extension):
    if not os_path_exists(file_directory):
        mkdir(file_directory)
    return os_path_join(file_directory, f"{file_name}{file_extension}")

def get_submission_file_path(part_index: int = "") -> str:
    if part_index != "":
        part_index = '_' + str(part_index)
    return _get_file_path(f"{_RESULTS_FILE_NAME}{part_index}", _SUBMISSIONS_DIRECTORY_PATH, _RESULTS_EXTENSION)

def get_pickle_file_path(file_name: str) -> str:
    return _get_file_path(file_name, _PICKELS_DIRECTORY_PATH, _PICKLE_EXTENSION)

def get_keras_file_path(file_name: str) -> str:
    return _get_file_path(file_name, _MODELS_DIRECTORY_PATH, _KERAS_EXTENSION)