'''
Trains a mnist data set using the developed convolution layer.
'''
import numpy as np
import mnist
from maxpool import MaxPool2
from conv import Conv
from softmax import SoftMax
import sys 

train_images = mnist.train_images()
train_labels = mnist.train_labels()


test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(len(train_images), len(test_images))


conv = Conv(8)      # 28x28x1 -> 26x26x8
pool = MaxPool2()   # 26x26x8 -> 13x13x8
softmax = SoftMax(13 * 13 * 8, 10) # 13x13x8 -> 10

only_once = False

def train(image, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward pass
    out, loss, accuracy = forward(image, label)

    # initial gradient on loss
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # back propagation
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    conv.backprop(gradient, lr)

    return loss, accuracy

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    image = (image / 255) - 0.5
    conv_out = conv.forward(image)
    pool_out = pool.forward(conv_out)
    final_out = softmax.forward(pool_out)

    # Calculate cross-entropy loss and accuracy.
    loss = -np.log(final_out[label])
    acc = 1 if np.argmax(final_out) == label else 0

    return final_out, loss, acc

def validate(test_images, test_labels):
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)

print('MNIST CNN initialized!')

for i in range(100):
    print('--------- epoch %d ---------' % i)
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images[:1000], train_labels[:1000])):
        # Print stats every 100 steps.
        if i % 100 == 99:
            print( 
                '[Step  %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        # Do a forward pass.
        l, acc = train(im, label)
        loss += l
        num_correct += acc

validate(test_images[:1000], test_labels[:1000])