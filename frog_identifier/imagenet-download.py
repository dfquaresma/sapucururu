from bs4 import BeautifulSoup
import requests, cv2, urllib, os, sys, PIL.Image
import numpy as np

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

def download_frog_images(path_to_train, path_to_validation, specie, url, limit=None):   

    try:
        os.mkdir(path_to_train + specie)
        os.mkdir(path_to_validation + specie)
    except:
        None

    path_to_training, path_to_validation = path_to_train + specie + "/", path_to_validation + specie + "/"  
    
    download_images_to_path(url, path_to_training, path_to_validation, limit=limit)

if __name__ == '__main__':
    try:
        os.mkdir("./imagenet")
        os.mkdir("./imagenet/train")
        os.mkdir("./imagenet/validation")
    except:
        None

    species = ["chorus", "tree", "bufo", "natterjack", "bullfrog"]
    frogs_data_urls = ["http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01652026", 
                      "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01644373", 
                      "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01646292",
                      "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01646648",
                      "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01641577"]
    path_to_training, path_to_validation = "./imagenet/train/", "./imagenet/validation/"

    for i in range(len(species)):
        download_frog_images(path_to_training, path_to_validation, species[i], frogs_data_urls[i])