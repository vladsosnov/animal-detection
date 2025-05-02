import numpy as np
import os
import cv2
import constants as CONST
import copy

# Ensure the results folder exists
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)

def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    image_copy = copy.deepcopy(image)
    
    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float') / 255.0
    return image_copy, image_std

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
