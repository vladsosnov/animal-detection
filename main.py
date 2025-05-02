import numpy as np
import os
import cv2
import pickle
import constants as CONST
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from data_prep import prep_and_load_data
from model import get_model
import copy

# Ensure the results folder exists
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)

def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
    
    # Accuracy plot
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
    plt.show()

    # Loss plot
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.show()

def video_write(model):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(os.path.join(results_dir, 'demo.mp4'), fourcc, 1.0, (400, 400))  # Save video in results folder
    val_map = {2: 'Horse', 1: 'Dog', 0: 'Cat'}  # Assuming you have 3 classes: Horse, Dog, and Cat

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)
    image_paths = image_paths[:100]  # Limiting to 100 for demo
    count = 0
    for img_path in image_paths:
        image, image_std = process_image(DIR, img_path)
        
        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
        pred = model.predict([image_std])
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        
        # Round the max_val to 2 decimal places
        confidence = f"{max_val[0] * 100:.2f}%"
        
        s = val_map[arg_max[0]] + ' - ' + confidence
        cv2.putText(image, s, location, font, fontScale, fontColor, lineType)
        
        frame = cv2.resize(image, (400, 400))
        out.write(frame)
        
        count += 1
        print(count)
    out.release()

def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    image_copy = copy.deepcopy(image)
    
    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float') / 255.0
    return image_copy, image_std


if __name__ == "__main__":
    # Unpack data and labels returned from prep_and_load_data
    data, labels = prep_and_load_data()

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
    history = model.fit(train_images, train_labels, batch_size=50, epochs=15, verbose=1, validation_data=(test_images, test_labels), callbacks=[tensorboard])
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
