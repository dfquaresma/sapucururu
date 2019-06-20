from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import keras

# input image dimensions
batch_size = 128

def get_pretrained_model(architecture_path, weights_path):
    # https://jovianlin.io/saving-loading-keras-models/
    model = None
    # Model reconstruction from JSON file
    with open(architecture_path, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weights_path)
    return model

def get_test_data_generator(test_data_path, img_rows_cols_tuple):
    # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_data_path, # './imagenet/test/'
        target_size=(int(img_rows_cols_tuple[0]), int(img_rows_cols_tuple[1])), # The target_size is the size of your input images,every image will be resized to this size
        batch_size=batch_size,
        class_mode='categorical'
    )

    return test_generator

def evaluate_model(model, test_generator):
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    ) 
    score = model.evaluate_generator(test_generator, steps=(ceil(len(test_generator)/batch_size)), verbose=0)
    # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    model_architecture_path = input("Enter the model architecture file path: ") # './trained-models/frog_identifier_cifar10_model_architecture.json'
    model_weights_path = input("Enter the model weights file path: ") # './trained-models/frog_identifier_cifar10_model_weights.h5'
    test_data_path = input("Enter the TEST dataset path: ") # './imagenet/test/'
    img_rows_cols_tuple = input("Enter the image's row and col separated by comma (row,col): ").split(",") # 256,256

    model = get_pretrained_model(model_architecture_path, model_weights_path)
    test_generator = get_test_data_generator(test_data_path, img_rows_cols_tuple)
    evaluate_model(model, test_generator)
