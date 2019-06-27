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

i_model_architecture_path = os.environ['imodel_architecture_path']
i_model_weights_path = os.environ['imodel_wights_path']
i_model = get_pretrained_model(i_model_architecture_path, i_model_weights_path)
tmp = list(map(int, os.environ['itarget_size'].split(",")))
i_target_size = (tmp[0], tmp[1])

c_model_architecture_path = os.environ['cmodel_architecture_path']
c_model_weights_path = os.environ['cmodel_wights_path']
c_model = get_pretrained_model(c_model_architecture_path, c_model_weights_path)
tmp = list(map(int, os.environ['ctarget_size'].split(",")))
c_target_size = (tmp[0], tmp[1])

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

def predict_image(URL):
    tmp_img_path = './tmp-img.jpg'
    actual_image = url_to_image(URL, tmp_img_path) # enter the url of the .jpg image
    img = load_img(tmp_img_path, target_size=i_target_size)
    img_array = img_to_array(img)
    img_array_np_expanded = np.expand_dims(img_array, axis=0)
    preds = i_model.predict(img_array_np_expanded)
    i_result = [
      'Has a frog: ' + str(preds[0,0] * 100), 
      'No frog: ' + str(preds[0,1] * 100)
    ]
    
    img = load_img(tmp_img_path, target_size=c_target_size)
    img_array = img_to_array(img)
    img_array_np_expanded = np.expand_dims(img_array, axis=0)
    preds = c_model.predict(img_array_np_expanded)
    c_result = [
      'bufo: ' + str(preds[0,0] * 100), 'bullfrog: ' + str(preds[0,1] * 100), 
      'chorus: ' + str(preds[0,2] * 100), 'natterjack: ' + str(preds[0,3] * 100), 
      'three: ' + str(preds[0,4] * 100)
    ]

    os.remove(tmp_img_path)
    return str(i_result) + "\n" + str(c_result)

def handle(req):
    try:
        return predict_image(req.decode("utf-8"))
    
    except Exception as e:
        return str(e)
