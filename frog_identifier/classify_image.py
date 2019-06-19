from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt

# input image dimensions
img_rows, img_cols = 224, 224
batch_size = 128
num_classes = 2
epochs = 10

def get_data_generator(data_path):
    # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
    datagen  = ImageDataGenerator()
    generator = datagen.flow_from_directory(
            data_path,
            target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
            batch_size=32,
            class_mode='categorical')

    return generator

# you shall move you dataset to that file's directory 
train_generator = get_data_generator('./imagenet/train/')
validation_generator = get_data_generator('./imagenet/validation/')

def create_model():
    # https://www.codeproject.com/Articles/4023566/Cat-or-Not-An-Image-Classifier-using-Python-and-Ke
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
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

model = createModel()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()
model.fit(train_generator,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=validation_generator,
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
