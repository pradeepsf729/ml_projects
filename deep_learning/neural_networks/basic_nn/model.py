import numpy as np
from nn import *


class Model:
    '''
    Indicates a neural network trained model.
    '''

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        '''
        Addas a given layer to the end of the layers stack.
        '''
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        '''
        used to provide the loss function and the optimizer
        to be used for training.
        ex: model.set(loss=CrossEntroyLoss, optimizer=SGD)
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_per_epoch=1, validation_data=None):
        '''
        Method to train the model.
        X, y - training data and labels mandatory params
        epochs - Number of epochs to be used.
        print_per_epoch - Loss and acuracy will be
                          printed for every 'print_per_epoch' times.
        '''

        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):

            # perform forward pass
            output = self.forward(X, training=True)

            # calculate loss
            data_loss, reg_loss = self.loss.calculate(output, y, include_regularization=True)
            total_loss = data_loss + reg_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # perform backpropagation
            self.backward(output, y)

            # update the params for layers with weights and biases
            self.optimizer.preupdate_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.postupdate_params()

            # Temporary
            # Print a summary
            if not epoch % print_per_epoch:
                print('epoch ', epoch)
                print('Loss - ', data_loss, reg_loss)
                print('Accuracy - ', accuracy)

        if validation_data:
            X_test, y_test = validation_data
            output = self.forward(X_test, training=False)

            # calculate loss
            data_loss = self.loss.calculate(output, y_test, include_regularization=False)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_test)

            print('Validation - result')
            print('Loss - ', data_loss, reg_loss)
            print('Accuracy - ', accuracy)

    def finalize(self):
        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        # set of layer which can be trained (that contain weights)
        self.trainable_layers = []

        for i, layer in enumerate(self.layers):

            # for first layer prev layer in input layer.
            if i == 0:
                layer.prev = self.input_layer
                layer.next = self.layers[i + 1]

            # middle hidden layers (initialize with prev and next layers)
            elif i < layer_count - 1:
                layer.prev = self.layers[i - 1]
                layer.next = self.layers[i + 1]
            
            else: # the last layer (output layer)
                layer.prev = self.layers[i - 1]
                layer.next = self.loss
                self.output_layer_activation = layer

            if hasattr(layer, 'weights') or hasattr(layer, 'biases'):
                self.trainable_layers.append(layer)

        # Update loss object with trainable layers
        # this helps in performing regularization only for applicable layers.
        self.loss.remember_trainable_layers(self.trainable_layers)

        # if the output layer if softmax and loss if cross entropy
        # we can use more efficient Activation_Softmax_Loss_CategoricalCrossentropy class
        if (isinstance(self.layers[-1], Activation_Softmax)) and \
            isinstance(self.loss, Loss_CategoricalCrossEntropy):
                self.softmax_classifier_output = \
                    Activation_Softmax_Loss_CategoricalCrossEntropy()

    def forward(self, X, training):
        '''
        Completes the forward pass of all the layers
        in this model.
        '''
        self.input_layer.forward(X, training)

        # all layers forward, with taking previous layers inputs.
        for layer in self.layers:
            layer.forward(layer.prev.outputs, training)

        return layer.outputs # last layer output

    def backward(self, outputs, y):
        '''
        Perform backprop of the model.
        - outputs - outputs from final layer of forward.
        - y - Actual y label values
        '''

        # handle differently when using Softmax and cross entroy
        # as combined.
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(outputs, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]): # skip last one
                layer.backward(layer.next.dinputs)
        else:
            self.loss.backward(outputs, y)
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)

