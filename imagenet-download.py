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

def save_images_on_disk(split_urls, path_to_training, path_to_validation, tag, limit=None):
    count = 0
    for progress in range(len(split_urls)): 
        # store all the images on a directory, print out progress whenever progress is 
        # a multiple of 20 so we can follow the (relatively slow) progress
        if ((progress + 1) % 20 == 0):
            print("downloading img" + str(progress) + ", url: " + str(split_urls[progress]))
        
        if (limit != None and count >= limit): break

        try:
            I = url_to_image(split_urls[progress])
            if (len(I.shape) == 3): # check if the image has width, length and channels
                image_name = 'img' + str(progress) + '.jpg' # create a name of each image
                save_path = path_to_training + tag + image_name
                if (count % 5) == 0: # try to ensure that 20% of data is to validation
                    save_path = path_to_validation + tag + image_name
                cv2.imwrite(save_path, I)
                count += 1
        except:
            None # print("Error downloading img " + str(progress) + ", url: " + str(split_urls[progress]))

def download_images_to_path(url_to_urls, path_to_training, path_to_validation, limit=None):
    page = requests.get(url_to_urls) # ship synset
    soup = BeautifulSoup(page.content, 'html.parser') # puts the content of the website into the soup variable, each url on a different line
    str_soup = str(soup) # convert soup to string so it can be split
    split_urls = str_soup.split('\r\n') # split so each url is a different possition on a list
    tag = url_to_urls.split("=")[-1]
    save_images_on_disk(split_urls, path_to_training, path_to_validation, tag, limit=limit)

def download_frog_images(path_to_train, path_to_validation, limit=None):
    frogs_data_urls = [ 
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01640846", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01643507", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01654637", 
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01643896", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644373", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01645776"#,
        #"http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01650167", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01644900", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01648620"
        #"http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01648139", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01649170"
    ]

    try:
        os.mkdir(path_to_train + "isfrog")
        os.mkdir(path_to_validation + "isfrog")
    except:
        None

    path_to_training, path_to_validation = path_to_train + "isfrog/", path_to_validation + "isfrog/"  
    for url in frogs_data_urls:
        # downloading frog data 
        download_images_to_path(url, path_to_training, path_to_validation, limit=limit)

def download_notfrog_images(path_to_train, path_to_validation, limit=None):
    not_frogs_data_urls = [
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01861778", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01503061", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01661091", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01473806",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07707451", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07557165", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01317541", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13066129",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13024012", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n12997654", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04341686", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03309808",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00441824", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00433661", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00463246", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09416076",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09238926", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09468604", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09366317", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09437454",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n11672400", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13104059", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13100156", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n11722982",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n11773987", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13083023", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02384858", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01321230",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01458842", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01321456", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01767661", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01321579",
        "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01324610", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01324799", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01323781"
        ] 

    try:
        os.mkdir(path_to_train + "notfrog")
        os.mkdir(path_to_validation + "notfrog")
    except:
        None

    path_to_training, path_to_validation = path_to_train + "notfrog/", path_to_validation + "notfrog/" 
    for url in not_frogs_data_urls:
        # downloading not frog data 
        download_images_to_path(url, path_to_training, path_to_validation, limit=limit)

if __name__ == '__main__':
    try:
        os.mkdir("./imagenet")
        os.mkdir("./imagenet/train")
        os.mkdir("./imagenet/validation")
    except:
        None

    path_to_training, path_to_validation = "./imagenet/train/", "./imagenet/validation/" 
    download_frog_images(path_to_training, path_to_validation)
    download_notfrog_images(path_to_training, path_to_validation, 500)
