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


def save_images_on_disk(split_urls, path_to_training, path_to_validation, tag):
    count = 0
    for progress in range(len(split_urls)): 
        # store all the images on a directory, print out progress whenever progress is 
        # a multiple of 20 so we can follow the (relatively slow) progress
        if ((progress + 1) % 20 == 0):
            print("downloading img" + str(progress) + ", url: " + str(split_urls[progress]))

        if not split_urls[progress] == None:
            try:
                I = url_to_image(split_urls[progress])
                if (len(I.shape) == 3): # check if the image has width, length and channels
                    image_name = 'img' + str(progress) + '.jpg' # create a name of each image
                    save_path = path_to_training + tag + image_name
                    if count == 0:
                        save_path = path_to_validation + tag + image_name
                    cv2.imwrite(save_path, I)
                    count = (count + 1) % 5 # try to ensure that 20% of data is to validation
            except:
                None # print("Error downloading img " + str(progress) + ", url: " + str(split_urls[progress]))


def download_images_to_path(tag, url_to_urls, path_to_training, path_to_validation):
    page = requests.get(url_to_urls) # ship synset
    soup = BeautifulSoup(page.content, 'html.parser') # puts the content of the website into the soup variable, each url on a different line
    str_soup = str(soup) # convert soup to string so it can be split
    split_urls = str_soup.split('\r\n') # split so each url is a different possition on a list
    save_images_on_disk(split_urls, path_to_training, path_to_validation, tag)


if __name__ == '__main__':
    try:
        os.mkdir("./imagenet")
        os.mkdir("./imagenet/train")
        os.mkdir("./imagenet/validation")
        os.mkdir("./imagenet/train/isfrog")
        os.mkdir("./imagenet/validation/isfrog")
    except:
        None

    path_to_training, path_to_validation = "./imagenet/train/isfrog/", "./imagenet/validation/isfrog/"  

    # downloading frog data 
    download_images_to_path("00-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01640846", path_to_training, path_to_validation)
    download_images_to_path("01-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01643507", path_to_training, path_to_validation)
    download_images_to_path("02-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01643896", path_to_training, path_to_validation)
    download_images_to_path("03-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644373", path_to_training, path_to_validation)
    download_images_to_path("04-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644900", path_to_training, path_to_validation)
    download_images_to_path("05-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01645776", path_to_training, path_to_validation)
    download_images_to_path("06-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01648139", path_to_training, path_to_validation)
    download_images_to_path("07-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01648620", path_to_training, path_to_validation)
    download_images_to_path("08-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01649170", path_to_training, path_to_validation)
    download_images_to_path("09-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01650167", path_to_training, path_to_validation)
    download_images_to_path("10-", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01654637", path_to_training, path_to_validation)
