import numpy as np
import os 
from random import shuffle
import constants as CONST 
import cv2

def label_img(name):
    # Extract the label from the image filename (e.g., 'cat.1.jpg')
    word_label = name.split('.')[0]
    
    if word_label == CONST.CAT:
        label = 0
    elif word_label == CONST.DOG:
        label = 1
    elif word_label == CONST.HORSE:
        label = 2
    else:
        raise ValueError(f"Unknown class label: {word_label}")
    
    # Create a one-hot encoded label (3 classes: cat, dog, horse)
    label_arr = np.zeros(3)  # For 3 classes: cat, dog, horse
    label_arr[label] = 1  # Set the correct class to 1
    return label_arr

def prep_and_load_images():
    DIR = CONST.TRAIN_DIR
    data = []   # List to store image data
    labels = [] # List to store the labels
    image_paths = os.listdir(DIR)
    shuffle(image_paths)
    count = 0

    print("Loading images from dataset...")

    for img_path in image_paths:
        # Skip files that are not images (optional, just to be safe)
        if not img_path.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        
        label = label_img(img_path)  # Get the label for the image
        path = os.path.join(DIR, img_path)
        image = cv2.imread(path)
        
        # Resize image to a fixed size (CONST.IMG_SIZE, CONST.IMG_SIZE)
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
        
        # Normalize the image
        image = image.astype('float') / 255.0 
        
        data.append(image)   # Append the image to the data list
        labels.append(label)  # Append the label to the labels list
        
        count += 1
        
        # Stop after reaching the specified size (CONST.DATA_SIZE)
        if count == CONST.DATA_SIZE:
            break

    # Shuffle the data and labels together to keep them aligned
    combined = list(zip(data, labels))
    shuffle(combined)
    data[:], labels[:] = zip(*combined)

    print(len(data))
    print('Images loaded successfully...')

    # Convert lists to NumPy arrays and return
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    prep_and_load_images()
    