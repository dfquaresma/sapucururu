from __future__ import print_function
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import matplotlib.pyplot as plt
import keras, sys

# input image dimensions
img_rows, img_cols = 256, 256
batch_size = 128
epochs = 1500

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
    #model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))

    return model

def get_data_generator(data_path, datagen, subset):
    # https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    generator = datagen.flow_from_directory(
        data_path,
        target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset
    )
    
    return generator

def get_data_generators(test_data_path, train_data_path):
    # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(img_rows, img_cols), # The target_size is the size of your input images,every image will be resized to this size
        batch_size=batch_size,
        class_mode='categorical'
    )

    train_datagen = ImageDataGenerator(horizontal_flip=False, validation_split=0.2)
    train_generator = get_data_generator(train_data_path, train_datagen, 'training')
    validation_generator = get_data_generator(train_data_path, train_datagen, 'validation')

    return test_generator, train_generator, validation_generator

def train_model(test_data_path, train_data_path, model_weights_path, model_architecture_path):
    model = create_model()
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    test_generator, train_generator, validation_generator = get_data_generators(test_data_path, train_data_path)    

    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1500, min_delta=0.001, baseline=0.0001)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=(ceil(len(train_generator) / batch_size)),
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=(ceil(len(validation_generator) / batch_size)),
        callbacks=[callback]
    )

    score = model.evaluate_generator(test_generator, steps=(ceil(len(test_generator)/batch_size)), verbose=0)
    # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # https://jovianlin.io/saving-loading-keras-models/
    # Save the weights
    model.save_weights(model_weights_path)

    # Save the model architecture
    with open(model_architecture_path, 'w') as f:
        f.write(model.to_json())
    
    return history

def plot(history, model_accuracy_path, model_loss_path):
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(model_accuracy_path)
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(model_loss_path)

if __name__ == '__main__':
    test_data_path =  sys.argv[1] # input("Enter the TEST dataset path: ") # './imagenet-to-identification/test/'
    train_data_path = sys.argv[2] # input("Enter the TRAIN dataset path: ") # './imagenet-to-identification/train/'
    model_weights_path = sys.argv[3] # input("Enter the model's weights name and path: ") # './trained-models/frog_identifier_imagenet256-final_model_weights.h5'
    model_architecture_path = sys.argv[4] # input("Enter the model's architecture name and path: ") # './trained-models/frog_identifier_imagenet256-final_model_architecture.json'
    model_accuracy_path = sys.argv[5] # input("Enter the model's accuracy plot name and path: ") # './plotted-models/frog_identifier_imagenet256-final_accuracy.png'
    model_loss_path = sys.argv[6] # input("Enter the model's loss plot name and path: ") # './plotted-models/frog_identifier_imagenet256-final_loss.png'
    [img_rows, img_cols] = sys.argv[7].split(",") # input("Enter the image's row and col separated by comma (row,col): ").split(",") # '256,256'
    img_rows, img_cols = int(img_rows), int(img_cols)    

    plot(
        train_model(test_data_path, train_data_path, model_weights_path, model_architecture_path), 
        model_accuracy_path, 
        model_loss_path
    )
