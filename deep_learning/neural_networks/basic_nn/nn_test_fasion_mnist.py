URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

import os
import urllib
import urllib.request
if not os.path.isfile(FILE):
    print ('Downloading {0} and saving as {1} ...' % (URL, FILE))
    urllib.request.urlretrieve(URL, FILE)

from zipfile import ZipFile

print ( 'Unzipping images...' )
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print ( 'Done!' )