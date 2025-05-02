import os
import pickle
import constants as CONST
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from load_images import prep_and_load_images
from model import get_model
from results_utils import video_write

# Ensure the results folder exists
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)

def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # One row, two columns

    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', color='blue', marker='o')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='green', marker='x')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(history['loss'], label='Train Loss', color='red', marker='o')
    axes[1].plot(history['val_loss'], label='Validation Loss', color='orange', marker='x')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'training_plots.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved to {plot_path}')


if __name__ == "__main__":
    # Unpack data and labels returned from prep_and_load_images
    data, labels = prep_and_load_images()

    train_size = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
    print('data size:', len(data), 'train size:', train_size)

    # Split the data into training and testing data
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    print('train data fetched..')

    test_data = data[train_size:]
    test_labels = labels[train_size:]
    print('test data fetched..')

    # Reshape data to match the expected input for the model
    train_images = train_data.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    test_images = test_data.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)

    # Now you can train the model
    model = get_model()
    print('training started...')
    log_dir = os.path.join("logs", "fit", "model")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train_images, train_labels, batch_size=50, epochs=3, verbose=1, validation_data=(test_images, test_labels), callbacks=[tensorboard])
    print('training done...')

    # Save the model
    model_file = os.path.join(results_dir, 'trained_model.h5')
    model.save(model_file)

    # Save training history
    history_file = os.path.join(results_dir, 'trained_model_history.pickle')
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    # Plot training history and create a video
    plotter(history_file)
    video_write(model)
