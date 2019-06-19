from __future__ import print_function
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import matplotlib.pylab as plt
import keras

# input image dimensions
img_rows, img_cols = 32, 32#256, 256
batch_size = 32#128
epochs = 15

# you shall move you dataset to that file's directory 
test_data_path = './cifar10/test/'#'./imagenet/validation/'
train_data_path = './cifar10/train/'#'./imagenet/train/'

# https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
            test_data_path,
            target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
            batch_size=batch_size,
            class_mode='categorical')

def get_data_generator(data_path, datagen, subset):
    # https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    generator = datagen.flow_from_directory(
            data_path,
            target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
            batch_size=batch_size,
            class_mode='categorical',
            subset=subset)
    return generator

train_datagen = ImageDataGenerator(horizontal_flip=False, validation_split=0.2)
train_generator = get_data_generator(train_data_path, train_datagen, 'training')
validation_generator = get_data_generator(train_data_path, train_datagen, 'validation')

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
# Model reconstruction from JSON file
#with open('frog_identifier_model_architecture.json', 'r') as f:
#    model = model_from_json(f.read())

# Load weights into the new model
#model.load_weights('frog_identifier_model_weights.h5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
          steps_per_epoch=(ceil(len(train_generator)/batch_size)),
          epochs=epochs,
          verbose=1,
          validation_data=validation_generator,
          validation_steps=(ceil(len(validation_generator)/batch_size)))

score = model.evaluate_generator(test_generator, steps=(ceil(len(test_generator)/batch_size)), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the weights
model.save_weights('frog_identifier_model_weights.h5')

# Save the model architecture
with open('frog_identifier_model_architecture.json', 'w') as f:
    f.write(model.to_json())
