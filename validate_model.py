from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import keras, sys

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
        target_size=img_rows_cols_tuple, # The target_size is the size of your input images,every image will be resized to this size
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
    return score[0], score[1]
 
if __name__ == '__main__':
    model_architecture_path = sys.argv[1] # input("Enter the model architecture file path: ") # './trained-models/frog_identifier_cifar10_model_architecture.json'
    model_weights_path = sys.argv[2] # input("Enter the model weights file path: ") # './trained-models/frog_identifier_cifar10_model_weights.h5'
    test_data_path = sys.argv[3] # input("Enter the TEST dataset path: ") # './imagenet-to-identification/test/'
    img_rows_cols_tuple = sys.argv[4].split(",") # input("Enter the image's row and col separated by comma (row,col): ").split(",") # '256,256'
    
    repetitions = 10
    loss_sum, accuracy_sum, count = 0, 0, 0
    for i in range(repetitions):
        model = get_pretrained_model(model_architecture_path, model_weights_path)
        test_generator = get_test_data_generator(test_data_path, (int(img_rows_cols_tuple[0]), int(img_rows_cols_tuple[1])))
        loss, accuracy = evaluate_model(model, test_generator)
        count, loss_sum, accuracy_sum = count + 1, loss_sum + loss, accuracy_sum + accuracy
        print('Count: %d, test loss: %f, test accuracy: %f' % (count, (loss_sum / count), (accuracy_sum / count)))

    final_loss = loss_sum / repetitions
    final_accuracy = accuracy_sum / repetitions
    # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    print('Final test loss:', final_loss)
    print('Final test accuracy:', final_accuracy)
