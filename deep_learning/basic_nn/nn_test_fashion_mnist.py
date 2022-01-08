URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

import os
import urllib
import urllib.request
from zipfile import ZipFile
import imageio
import numpy as np
from model import Model
from nn import *
import matplotlib.pyplot as plt
import cv2

# inference
# test some random images

labels = {
    0	: 'T-shirt/top',
    1	: 'Trouser',
    2	: 'Pullover',
    3	: 'Dress',
    4	: 'Coat',
    5	: 'Sandal',
    6	: 'Shirt',
    7	: 'Sneaker',
    8	: 'Bag',
    9	: 'Ankle boot'
}

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
    print('loading images as data arrays')
    X, y = load_mnist_dataset(path, 'train')
    X_test, y_test = load_mnist_dataset(path, 'test')

    return X, y, X_test, y_test

def download_fasion_mnist_dataset(outout_path):
    if not os.path.isfile(FILE):
        print ('Downloading {0} and saving as {1} ...'.format(URL, FILE))
        urllib.request.urlretrieve(URL, FILE)

    if not os.path.isdir(outout_path):
        print ( 'Unzipping images...' )
        with ZipFile(FILE) as zip_images:
            zip_images.extractall(outout_path)
        print ( 'Done!' )
    else:
        print('image data already available - [not downloading!!]')

def prepare_train_test_data():
    download_fasion_mnist_dataset(FOLDER)

    X, y, X_test, y_test = create_dataset(FOLDER)

    #print(X.shape)
    BATCH_SIZE = 10
    steps = X.shape[0] // BATCH_SIZE

    #print(steps)

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # the network accepts numpy 2-D array
    # what we have is 3-D array n_images * x_pixel * y_pixel
    # flatten all the pixels into one dimension
    n_train_img_samples = X.shape[0]
    flattened_X = X.reshape(n_train_img_samples, -1)

    n_test_img_samples = X_test.shape[0]
    flattened_X_test = X_test.reshape(n_test_img_samples, -1)

    # normalizing the pixel values between [-1, +1]
    X = (flattened_X.astype(np.float32) - 127.5) / 127.5
    X_test = (flattened_X_test.astype(np.float32) - 127.5) / 127.5
    return X, y, X_test, y_test

def train_and_test_nn_model(X, y, X_test, y_test, hidden_layer_ip_out):
    # create the network and train
    model = Model()
    model.add(Layer_Dense(X.shape[1], hidden_layer_ip_out))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(hidden_layer_ip_out, hidden_layer_ip_out))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(hidden_layer_ip_out, 10))
    model.add(Activation_Softmax())

    model.set(loss=Loss_CategoricalCrossEntropy(),
                optimizer=Optimizer_Adam(decay = 5e-5),
                accuracy=Accuracy_Categorical())

    model.finalize()

    model.train(X, y, validation_data=(X_test, y_test),
                epochs = 5, batch_size=128, print_per_epoch=1)

    print('Making some predictions from the trained model - \n')
    for i in range(4):
        ind= np.random.randint(0, len(X_test))
        pred = model.forward(X_test[ind], False)
        res = np.argmax(pred)
        print('Predicted - ', labels[res])
        print('Actual - ', labels[y_test[ind]])
        print('*' * 24)
    print('\n')
    return model

def train_model_with_64_64_network(X, y, X_test, y_test):
    return train_and_test_nn_model(X, y, X_test, y_test, 64)

def train_model_with_128_128_network(X, y, X_test, y_test):
    return train_and_test_nn_model(X, y, X_test, y_test, 128)

def evaluate_model_with_given_weights(model_parameters):
    # create new model, but not training
    model = Model()
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # no optimizer needed, as no training of network
    model.set(loss=Loss_CategoricalCrossEntropy(),
                    accuracy=Accuracy_Categorical())

    model.finalize()

    # this sets the weights and baises to the layers (equivalent to training)
    model.set_parameters(model_parameters)

    # check the accuracy.
    model.evaluate(X_test, y_test)

    # do some predictions
    for i in range(3):
        ind= np.random.randint(0, len(X_test))
        pred = model.forward(X_test[ind], False)
        res = np.argmax(pred)
        print('Predicted - ', labels[res])
        print('Actual - ', labels[y_test[ind]])
        print('*' * 24)

if __name__ == '__main__':
    ## Get the parameters of previous model and load and try new model without training.

    X, y, X_test, y_test = prepare_train_test_data()

    model = train_model_with_64_64_network(X, y, X_test, y_test)
    model = train_model_with_128_128_network(X, y, X_test, y_test)

    print('Evaluating model with loading pre-trained parameters')
    evaluate_model_with_given_weights(model.get_parameters())

    # loading params into file (persisting parameters)
    model.save_parameters('fashion_mnist.params')

    #model.load_parameters('fashion_mnist.params')
    loaded_params = model.get_parameters()

    print('Evaluating model with params loaded from file')
    evaluate_model_with_given_weights(loaded_params)

    print('Evaluating model with total model loaded from file')
    # saving the entire model to a file (instead of just params)
    model.save('fashion_mnist.model')

    model_new = Model.load('fashion_mnist.model')
    # check the accuracy.
    model_new.evaluate(X_test, y_test)

    # testing the model with realtime image.
    image_files = ['tshirt.png', 'pants.png']

    for img_file in image_files:
        # load the image, convert to grey scale
        image_data = cv2.imread( img_file , cv2.IMREAD_GRAYSCALE)

        # convert to data array and compress to pixel size of that network can take (28 * 28)
        image_data = cv2.resize(image_data, ( 28 , 28 ))

        # the images thatwere trained were with black background,
        # we need to invert so make the test image as same.
        image_data = 255 - image_data

        # reshape to n_sample * 784 (28 * 28 pixels flattened)
        image_data = image_data.reshape( 1 , - 1 ).astype(np.float32)

        # scaling to [-1 , 1]
        image_data = (image_data - 127.5) / 127.5

        # predict the outcome
        model_preds = model_new.predict(image_data)

        # for a categorical classifier, the output will be set of probabilities (softmax)
        # selecting max probability for each sample outcome
        predicted_class = np.argmax(model_preds, axis=1)

        for res in predicted_class:
            print('image - ', img_file, ' item predicted - ', labels[res])

