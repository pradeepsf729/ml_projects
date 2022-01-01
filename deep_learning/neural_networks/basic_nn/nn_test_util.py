import mnist
import pandas as pd
import time
from nn import *

class Model:
    '''
    Just placholder to store layer object
    '''
    pass

def train(X, y, num_epochs=100):

    regularization_params = {'weight_regularizer_l2' : 5e-4, 
                             'bias_regularizer_l2' : 5e-4 }

    # each image in mnist is 28 * 28 pixels
    # thus the total features will be 784
    dense1 = Layer_Dense(784, 1024, **regularization_params)

    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()

    # Create dropout layer
    dropout1 = Layer_Dropout( 0.1 )

    # This is the output layer where the output will
    # have ten neurons corresponding to each digit.
    dense2 = Layer_Dense(1024, 10)

    # Create Softmax classifier's combined loss and activation
    softmax_and_loss_function = \
        Activation_Softmax_Loss_CategoricalCrossEntropy()

    epochs = num_epochs
    print_loss_accuracy = True
    epoch_print_step = 100 # loss and accuracy printed for every.

    l = []
    a = []
    l_r = []
    o1 = Optimizer_SGD(learning_rate=1.0,  decay=1e-3, momentum=0.9)
    o2 = Optimizer_RmsProp(learning_rate = 0.02 , decay = 1e-5 ,rho = 0.999)
    o3 = Optimizer_Adam(decay = 5e-7)
    o4 = Optimizer_Adagrad(decay = 1e-4 )
    #optimizers = [o1, o2, o3 , o4]
    # using only one optimizer
    optimizers = [o3]
    for o in optimizers:
        optimizer = o
        for epoch in range(1, epochs):
            start = time.time()
            # do the forward pass first
            dense1.forward(X.reshape(X.shape[0], 784))
            activation1.forward(dense1.outputs)
            dense2.forward(activation1.outputs)
            loss = softmax_and_loss_function.forward(dense2.outputs, y)

            reg_loss = softmax_and_loss_function.regularization_loss(dense1)
            reg_loss += softmax_and_loss_function.regularization_loss(dense2)

            loss += reg_loss

            l.append(loss)
            # calculate accuracy
            predictions = np.argmax(softmax_and_loss_function.output, axis=1)

            if len (y.shape) == 2 : # incase one-hot encoded.
                y = np.argmax(y, axis = 1 )
            accuracy = np.mean(predictions == y)
            a.append(accuracy * 100)

            if print_loss_accuracy:
                if not epoch % epoch_print_step :
                    print (f'epoch: {epoch} , ' + 
                    f'acc: {accuracy * 100} %, ' +
                    f'loss: {loss} , ' +
                    f'lr: {optimizer.current_learning_rate} ' )

            l_r.append(optimizer.current_learning_rate)

            # backpropagation
            softmax_and_loss_function.backward(softmax_and_loss_function.output, y)
            dense2.backward(softmax_and_loss_function.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # update the weights
            optimizer.preupdate_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.postupdate_params()
            elpsed = round((time.time() - start), 2)
            print('epoch %d done, time taken : %.2f seconds' % (epoch, elpsed))
        print('optimizer    : ', o.__class__.__name__)
        print('total epochs : ', epochs)
        print('max accuracy : ', max(a), '%')
        print('min losss    : ', min(l))
        print('---- ************************************* ----\n')

    model = Model()
    model.dense1 = dense1
    model.dense2 = dense2
    model.activation1 = activation1
    model.loss_activation = softmax_and_loss_function
    return model

def validate(model, X_test, y_test):
    model.dense1.forward(X_test.reshape(X_test.shape[0], 784))
    model.activation1.forward(model.dense1.outputs)
    model.dense2.forward(model.activation1.outputs)
    loss = model.loss_activation.forward(model.dense2.outputs, y_test)

    # get index of highest value for each predicted output 
    # that index will be the class predicted by the model
    predictions = np.argmax(model.loss_activation.output, axis = 1 )
    if len (y_test.shape) == 2 : # incase one-hot encoded.
        y_test = np.argmax(y_test, axis = 1 )
    acc = np.mean(predictions == y_test)
    print ( f'validation, acc: {acc * 100} %, loss: {loss} ' )

if __name__ == '__main__':
    # Create dataset
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    # considering only 1000 images out of 60000 images
    start = time.time()
    X = train_images
    y = train_labels

    model = train(X, y, num_epochs=100)

    end = time.time()
    print('Training time : ', round((end - start), 2), 'seconds')
    test_images = mnist.test_images()   
    test_labels = mnist.test_labels()
    X_test = test_images
    y_test = test_labels

    validate(model, X_test, y_test)

