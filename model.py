import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import constants as CONST

def get_model():
    model = tf.keras.Sequential()
    
    # First Convolution Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    # Second Convolution Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    # Third Convolution Block
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    # Fourth Convolution Block
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Fifth Convolution Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output Layer for 3 classes (cat, dog, horse)
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 neurons

    # Compile the model with categorical cross-entropy loss and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print('Model prepared...')
    return model
