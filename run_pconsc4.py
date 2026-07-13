import os
import random
import numpy as np
import tensorflow as tf

# Explicit import to avoid ambiguity
from pconsc4.create_model import get_pconsc4
from pconsc4 import predict  # import the predict function


def setup_gpu():
    """Enable dynamic GPU memory allocation (works in TF1.x and TF2.x)."""
    try:
        # TF2.x way
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TF2.x GPU memory growth enabled.")
    except Exception:
        # TF1.x fallback
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)
        print("TF1.x GPU memory growth enabled.")

def pconsc4Prediction():
    setup_gpu()

    # Use the function from the package
    model = get_pconsc4()
    datasets = ["davis", "kiba"]
    for dataset in datasets:
        base_dir = os.path.join("data", dataset)
        aln_dir = os.path.join(base_dir, "hhfilter")
        output_dir = os.path.join(base_dir, "pconsc4")

        os.makedirs(output_dir, exist_ok=True)

        file_list = os.listdir(aln_dir)
        random.shuffle(file_list)

        for file in file_list:
            if not file.endswith(".a3m"):
                continue

            input_file = os.path.join(aln_dir, file)
            output_file = os.path.join(output_dir, file.replace('.a3m', '.npy'))

            if os.path.exists(output_file):
                print(output_file, 'exists. Skipping.')
                continue

            try:
                print('Processing', input_file)
                pred = predict(model, input_file)
                np.save(output_file, pred['cmap'])
                print(output_file, 'saved.')
            except Exception as e:
                print(output_file, 'error:', e)
            


if __name__ == '__main__':
    pconsc4Prediction()
