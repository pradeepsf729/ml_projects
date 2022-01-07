URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

import os
import urllib
import urllib.request
from zipfile import ZipFile
import imageio
import numpy as np

def load_mnist_dataset(path, dataset):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []

    for label in labels:
        img_files_path = os.path.join(path, dataset, label)
        for img_file in os.listdir(img_files_path):
            img_file_path = os.path.join(img_files_path, img_file)
            img_arr = imageio.imread(img_file_path)
            X.append(img_arr)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def create_dataset(path):
    X, y = load_mnist_dataset(path, 'train')
    X_test, y_test = load_mnist_dataset(path, 'test')

    return X, y, X_test, y_test

def download_fasion_mnist_dataset(outout_path):
    if not os.path.isfile(FILE):
        print ('Downloading {0} and saving as {1} ...' % (URL, FILE))
        urllib.request.urlretrieve(URL, FILE)

    if not os.path.isdir(outout_path):
        print ( 'Unzipping images...' )
        with ZipFile(FILE) as zip_images:
            zip_images.extractall(outout_path)
        print ( 'Done!' )
    else:
        print('data already available')

download_fasion_mnist_dataset(FOLDER)

X, y, X_test, y_test = create_dataset(FOLDER)

print(X.shape)
BATCH_SIZE = 10
steps = X.shape[0] // BATCH_SIZE

print(steps)