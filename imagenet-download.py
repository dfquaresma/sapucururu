from bs4 import BeautifulSoup # require pip install BeautifulSoup4
import numpy as np
import requests # require pip install requests
import cv2 # require pip install cv2
import PIL.Image # require pip install Pillow
import urllib # require pip install
import os, sys

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def save_images_on_disk(split_urls, number_of_images, path_to_save, init_range=0):
    for progress in range(init_range, number_of_images): 
        # store all the images on a directory, print out progress whenever progress is 
        # a multiple of 20 so we can follow the (relatively slow) progress
        if (progress % 20 == 0):
            print("downloading img" + str(progress) + ", url: " + str(split_urls[progress]))

        if not split_urls[progress] == None:
            try:
                I = url_to_image(split_urls[progress])
                if (len(I.shape) == 3): # check if the image has width, length and channels
                    image_name = 'img' + str(progress) + '.jpg' # create a name of each image
                    save_path = path_to_save + image_name
                    cv2.imwrite(save_path, I)
            except:
                print("Error downloading img " + str(progress) + ", url: " + str(split_urls[progress]))

def download_images_to_path(url_to_urls, path_to_training, n_of_training_images, path_to_validation,  n_of_validation_images):
    page = requests.get(url_to_urls) # ship synset

    # BeautifulSoup is an HTML parsing library
    soup = BeautifulSoup(page.content, 'html.parser')# puts the content of the website into the soup variable, each url on a different line
    str_soup = str(soup) # convert soup to string so it can be split
    split_urls = str_soup.split('\r\n') # split so each url is a different possition on a list

    img_rows, img_cols = 244, 244 # number of rows and columns to convert the images to
    input_shape = (img_rows, img_cols, 3) # format to store the images (rows, columns, channels) called channels last

    # training data:
    save_images_on_disk(split_urls, n_of_training_images, path_to_training)
    # Validation data:
    save_images_on_disk(split_urls, n_of_validation_images, path_to_validation, init_range=n_of_training_images)

if __name__ == '__main__':
    n_of_training_images, n_of_validation_images = 800, 100
    path_to_training, path_to_validation = "/imagenet/train/isfrog/", "/imagenet/validation/isfrog/"  

    try:
        os.mkdir("./imagenet")
        os.mkdir("./imagenet/train")
        os.mkdir("./imagenet/validation")
        os.mkdir("./imagenet/train/isfrog")
        os.mkdir("./imagenet/validation/isfrog")
    except:
        None

    # downloading frog data 
    download_images_to_path("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644373", 
        path_to_training, n_of_training_images, path_to_validation,  n_of_validation_images) # http://image-net.org/synset?wnid=n01644373

    download_images_to_path("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01641391", 
        path_to_training, n_of_training_images, path_to_validation,  n_of_validation_images) # http://image-net.org/synset?wnid=n01641391

    download_images_to_path("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01641206", 
        path_to_training, n_of_training_images, path_to_validation,  n_of_validation_images) # http://image-net.org/synset?wnid=n01641206

    download_images_to_path("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644900", 
        path_to_training, n_of_training_images, path_to_validation,  n_of_validation_images) # http://image-net.org/synset?wnid=n01644900
