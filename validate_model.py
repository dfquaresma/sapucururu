from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import sys
from sklearn.metrics import confusion_matrix, classification_report
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

batch_size=32

def get_pretrained_model(architecture_path, weights_path):
    # https://jovianlin.io/saving-loading-keras-models/
    model = None
    # Model reconstruction from JSON file
    with open(architecture_path, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weights_path)
    return model

def get_test_data_generator(test_data_path, model_image_size):
    # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_data_path, # './imagenet/test/'
        target_size=model_image_size, # The target_size is the size of your input images,every image will be resized to this size
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator

def confusion_matrix(model, test_generator):
    Y_pred = np.array(model.predict_generator(test_generator, steps=len(test_generator), verbose=0))
    y_pred = np.array(np.argmax(Y_pred, axis=1))
    print('Classification Report')
    target_names = ['Frog', 'NotFrog']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
    
    cm = ConfusionMatrix(test_generator.classes, y_pred)
    print('Confusion Matrix')
    print(cm)
    return cm
 
if __name__ == '__main__':
    model_architecture_path = sys.argv[1] # input("Enter the model architecture file path: ") # './trained-models/frog_identifier_cifar10_model_architecture.json'
    model_weights_path = sys.argv[2] # input("Enter the model weights file path: ") # './trained-models/frog_identifier_cifar10_model_weights.h5'
    test_data_path = sys.argv[3] # input("Enter the TEST dataset path: ") # './imagenet-to-identification/test/'
    model_image_size = sys.argv[4].split(",") # input("Enter the image's row and col separated by comma (row,col): ").split(",") # '256,256'
    plot_title = sys.argv[5]

    model = get_pretrained_model(model_architecture_path, model_weights_path)
    test_generator = get_test_data_generator(test_data_path, (int(model_image_size[0]), int(model_image_size[1])))

    cm = confusion_matrix(model, test_generator)
    ax = cm.plot(normalized=True)
    ax.set_title(plot_title)
    ax.set_xticklabels(['Frog', 'NotFrog'])
    ax.set_yticklabels(['Frog', 'NotFrog'])
    plt.show()