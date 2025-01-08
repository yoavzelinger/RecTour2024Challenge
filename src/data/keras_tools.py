from src.data.directories import get_keras_file_path

def save_keras_model_weights(model, name):
    model.save_weights(f"{get_keras_file_path(name)}.h5", overwrite=True)

def load_keras_model_weights(model, name):
    model.load_weights(get_keras_file_path(name))