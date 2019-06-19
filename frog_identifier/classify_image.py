from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import matplotlib.pylab as plt
import keras

# input image dimensions
img_rows, img_cols = 32, 32#256, 256
batch_size = 32#128
epochs = 10

def get_data_generator(data_path):
    # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
    datagen  = ImageDataGenerator()
    generator = datagen.flow_from_directory(
            data_path,
            target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
            batch_size=batch_size,
            class_mode='categorical')

    return generator

# you shall move you dataset to that file's directory 
train_generator = get_data_generator('./cifar10/train/')#'./imagenet/train/')
validation_generator = get_data_generator('./cifar10/validation/')#'./imagenet/validation/')

def create_model():
    # https://www.codeproject.com/Articles/4023566/Cat-or-Not-An-Image-Classifier-using-Python-and-Ke
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    
    return model

model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
          steps_per_epoch=(ceil(len(train_generator)/batch_size)),
          epochs=epochs,
          verbose=1,
          validation_data=validation_generator,
          validation_steps=(ceil(len(validation_generator)/batch_size)))

score = model.evaluate_generator(validation_generator, steps=(ceil(len(validation_generator)/batch_size)), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
