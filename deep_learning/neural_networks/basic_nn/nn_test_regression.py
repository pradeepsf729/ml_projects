import numpy as np
from nnfs.datasets import sine_data
import nnfs
from nn import *
import matplotlib.pyplot as plt

nnfs.init()

class Model:
    pass

def train(X, y, num_epochs):
    dense1 = Layer_Dense(1, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 64)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(64, 1)
    activation3 = Activation_Linear()
    loss = Loss_MeanSquaredError()

    optimizer = Optimizer_Adam(learning_rate = 0.005 , decay = 1e-3)

    print_loss_accuracy = True
    epoch_print_step = 1000
    for epoch in range(1, num_epochs):
        # forward pass
        dense1.forward(X)
        activation1.forward(dense1.outputs)
        dense2.forward(activation1.outputs)
        activation2.forward(dense2.outputs)
        dense3.forward(activation2.outputs)
        activation3.forward(dense3.outputs)

        # calculate loss
        loss_output = loss.calculate(activation3.outputs, y)

        # calculate accuracy
        precision = np.std(y)/250
        accuracy = np.mean(np.abs(activation3.outputs - y) < precision)

        # print the loss, accuracy as we progress training
        if print_loss_accuracy:
            if not epoch % epoch_print_step :
                print (f'epoch: {epoch} , ' + 
                        f'acc: {accuracy * 100} %, ' +
                        f'loss: {loss_output} , ' +
                        f'lr: {optimizer.current_learning_rate} ' )
        # backward pass
        loss.backward(activation3.outputs, y)
        
        activation3.backward(loss.dinputs)
        dense3.backward(activation3.dinputs)
        
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)

        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # updates params with optimizer.
        optimizer.preupdate_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.postupdate_params()

    model = Model()
    model.dense1 = dense1
    model.dense2 = dense2
    model.dense3 = dense3
    model.activation1 = activation1
    model.activation2 = activation2
    model.activation3 = activation3
    model.loss_activation = loss
    return model

def validate(model, X_test, y_test):
    model.dense1.forward(X_test)
    model.activation1.forward(model.dense1.outputs)
    model.dense2.forward(model.activation1.outputs)
    model.activation2.forward(model.dense2.outputs)
    model.dense3.forward(model.activation2.outputs)
    model.activation3.forward(model.dense3.outputs)
    loss = model.loss_activation.calculate(model.activation3.outputs, y)

    precision = np.std(y_test)/250
    accuracy = np.mean(np.abs(model.activation3.outputs - y_test) < precision)

    print ( f'validation, acc: {accuracy * 100} %, loss: {loss} ' )
    plt.plot(X_test, y_test)
    plt.plot(X_test, model.activation3.outputs)
    plt.show()

X, y = sine_data()
model = train(X, y, num_epochs=10001)

X_test, y_test = sine_data()
validate(model, X_test, y_test)