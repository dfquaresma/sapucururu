from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import cv2, urllib, os
import numpy as np

def get_pretrained_model(architecture_path, weights_path):
    # https://jovianlin.io/saving-loading-keras-models/
    model = None
    # Model reconstruction from JSON file
    with open(architecture_path, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weights_path)
    return model

def url_to_image(url, tmp_img_path):
  # https://blog.goodaudience.com/train-a-keras-neural-network-with-imagenet-synsets-in-google-colaboratory-e68dc4fd759f
  # download the image, convert it to a NumPy array, and then read it into OpenCV format
  resp = urllib.request.urlopen(url) # only works with python 3+
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  I = image
  if len(I.shape) == 3: # check if the image has width, length and channels, as I found some withouth channel
    save_path = tmp_img_path
    cv2.imwrite(save_path, I)

  return image

def predict_image(model, URL):
    tmp_img_path = './tmp-img.jpg'
    actual_image = url_to_image(URL, tmp_img_path) # enter the url of the .jpg image
    img = load_img(tmp_img_path, target_size=(32, 32))
    img_array = img_to_array(img)
    img_array_np_expanded = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array_np_expanded)
    
    print('Probability that the image is a frog:', preds[0,0])
    print('Probability that the image is NOT a frog:', preds[0,1])
    
    os.remove(tmp_img_path)

if __name__ == '__main__':
    model_architecture_path = input("Enter the model architecture file path: ") # './trained-models/frog_identifier_cifar10_model_architecture.json'
    model_weights_path = input("Enter the model weights file path: ") # './trained-models/frog_identifier_cifar10_model_weights.h5'
    model = get_pretrained_model(model_architecture_path, model_weights_path)
    while (True):
        URL = input("Enter the Image URL: ")
        if not URL:
            break
            
        predict_image(model, URL)
