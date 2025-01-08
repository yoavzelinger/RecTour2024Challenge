# RecTour2024Challenge

## Structure:
* data - All data-related files. Contains:
    * raw - The raw input to the project (Dataset from Kaggle).
    * processed - Data preprocessed.
    Due to files sizes the folder doesn't contain the files, insert them the the data folders as described in it's README.
* notebooks - Jupiter Notebooks.
* out - all the processed files (pickles, models, etc.):
    * archive - old models.
    * models - the models files.
    * pickles - saved pickles files (mostly dictionaries) (cannot be uploaded due to size, will be generated during the notebooks runs).
    * top_accommodations.csv - accommodations with top items.
* src - Source code. Contains:
    * data - Scripts for data loading.
    * models - Code for building, training, evaluating, and saving models.
    * utils - Utility scripts.
* submissions - The submission files: