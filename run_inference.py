import os
import constants as CONST
from tensorflow.keras.models import load_model
from load_images import prep_and_load_images
from results_utils import video_write

# Ensure the results folder exists
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    # Load the trained model
    model_file = os.path.join(results_dir, 'trained_model.h5')
    model = load_model(model_file)
    print('Model loaded.')

    # Load and preprocess new validation/test data
    data, labels = prep_and_load_images()
    test_size = int(CONST.DATA_SIZE * (1 - CONST.SPLIT_RATIO))
    print('Total data:', len(data), 'Validation size:', test_size)

    test_data = data[-test_size:]
    test_images = test_data.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)

    video_write(model)
