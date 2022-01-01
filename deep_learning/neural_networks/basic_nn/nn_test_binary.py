from nn import *
import numpy as np
from nnfs.datasets import spiral_data



class Model:
    '''
    Just placholder to store layer object
    '''
    pass

def train(X, y):
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 1)
    activation2 = Activation_Sigmoid()
    loss = Loss_BinaryCrossEntropy()

    optimizer = Optimizer_Adam(decay = 5e-7)

    epochs = 10000
    print_loss_accuracy = True
    epoch_print_step = 1000
    for epoch in range(1, epochs):
        # forward pass
        dense1.forward(X)
        activation1.forward(dense1.outputs)
        dense2.forward(activation1.outputs)
        activation2.forward(dense2.outputs)

        # calculate loss
        loss_output = loss.calculate(activation2.outputs, y)

        # calculate accuracy
        predictions = (activation2.outputs > 0.5) * 1
        accuracy = np.mean(predictions == y)

        # print the loss, accuracy as we progress training
        if print_loss_accuracy:
            if not epoch % epoch_print_step :
                print (f'epoch: {epoch} , ' + 
                        f'acc: {accuracy * 100} %, ' +
                        f'loss: {loss_output} , ' +
                        f'lr: {optimizer.current_learning_rate} ' )
        # backward pass
        loss.backward(activation2.outputs, y)
        activation2.backward(loss.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # updates params with optimizer.
        optimizer.preupdate_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.postupdate_params()

    model = Model()
    model.dense1 = dense1
    model.dense2 = dense2
    model.activation1 = activation1
    model.activation2 = activation2
    model.loss_activation = loss
    return model

def validate(model, X_test, y_test):
    model.dense1.forward(X_test)
    model.activation1.forward(model.dense1.outputs)
    model.dense2.forward(model.activation1.outputs)
    model.activation2.forward(model.dense2.outputs)
    loss = model.loss_activation.calculate(model.activation2.outputs, y)

    # get index of highest value for each predicted output 
    # that index will be the class predicted by the model
    predictions = (model.activation2.outputs > 0.5) * 1
    acc = np.mean(predictions == y_test)
    print ( f'validation, acc: {acc * 100} %, loss: {loss} ' )

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1)

model = train(X, y)

X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

validate(model, X_test, y_test)