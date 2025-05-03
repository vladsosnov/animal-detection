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
    out = cv2.VideoWriter(os.path.join(results_dir, 'demo.mp4'), fourcc, 1.0, (400, 400))
    val_map = {2: 'Horse', 1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)            # Prediction label
    name_location = (20, 50)       # Image filename
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)
    image_paths = image_paths[:150]

    for count, img_path in enumerate(image_paths, 1):
        image, image_std = process_image(DIR, img_path)
        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
        
        pred = model.predict([image_std])
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        
        confidence = f"{max_val[0] * 100:.2f}%"
        label = val_map[arg_max[0]] + ' - ' + confidence

        # Draw prediction
        cv2.putText(image, label, location, font, fontScale, fontColor, lineType)

        # Draw image filename
        cv2.putText(image, f"File: {img_path}", name_location, font, fontScale, fontColor, lineType)

        frame = cv2.resize(image, (400, 400))
        out.write(frame)

        print(f"[{count}] Processed: {img_path}")
    
    out.release()
    print("Video saved to:", os.path.join(results_dir, 'demo.mp4'))

