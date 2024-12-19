from tensorflow.keras.saving import save_model
from tensorflow.keras.models import load_model

from src.data.directories import get_keras_file_path

def save_keras_model(model, name):
    save_model(model, get_keras_file_path(name))

def load_keras_model(name):
    return load_model(get_keras_file_path(name))