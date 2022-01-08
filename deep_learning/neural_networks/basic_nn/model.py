import numpy as np
from nn import *
import pickle
import copy

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

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        '''
        used to provide the loss function and the optimizer
        to be used for training.
        ex: model.set(loss=CrossEntroyLoss, optimizer=SGD)
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *,
                epochs=1, batch_size=None,
                print_per_epoch=1, validation_data=None):
        '''
        Method to train the model.
        X, y - training data and labels mandatory params
        epochs - Number of epochs to be used.
        print_per_epoch - Loss and acuracy will be
                          printed for every 'print_per_epoch' times.
        '''

        self.accuracy.init(y)

        # Default value if batch size is not set
        train_steps = 1

        if batch_size is not None:
            n_samples = len(X)
            train_steps = n_samples // batch_size

            # add one more step for the leftovers
            # incase the samples are not multiple of batch_size
            if train_steps * batch_size < n_samples:
                train_steps += 1

        for epoch in range(1, epochs + 1):

            # reset for every epoch
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1 ) * batch_size]
                    batch_y = y[step * batch_size:(step + 1 ) * batch_size]

                # perform forward pass
                output = self.forward(batch_X, training=True)

                # calculate loss
                data_loss, reg_loss = \
                        self.loss.calculate(output, batch_y, include_regularization=True)
                total_loss = data_loss + reg_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # perform backpropagation
                self.backward(output, batch_y)

                # update the params for layers with weights and biases
                self.optimizer.preupdate_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.postupdate_params()

            # Temporary
            # Print a summary
            epoch_data_loss, epoch_reg_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if not epoch % print_per_epoch:
                print('epoch ', epoch, end=' : ')
                print('Loss - ', epoch_loss, end=' ') 
                print('Accuracy - ', epoch_accuracy)

        if validation_data:
            self.evaluate(*validation_data, batch_size=batch_size)

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
        if self.loss is not None:
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

    def evaluate(self, X, y, *, batch_size=None):

        if batch_size is not None:
            n_samples = len(X)
            validation_steps = n_samples // batch_size

            # add one more step for the leftovers
            # incase the samples are not multiple of batch_size
            if validation_steps * batch_size < n_samples:
                validation_steps += 1
        else:
            validation_steps = 1 # only in single step

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is not None:
                batch_X = X[step * batch_size : (step+1) * batch_size]
                batch_y = y[step * batch_size : (step+1) * batch_size]
            else:
                batch_X = X
                batch_y = y

            output = self.forward(batch_X, training=False)

            # accumulate the loss for this batch
            self.loss.calculate(output, batch_y, include_regularization=False)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # get the accumulated loss and accuracy of all batches.
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print('Validation - result')
        print('Loss - ', validation_loss)
        print('Accuracy - ', validation_accuracy)

    def get_parameters(self):
        # get the the weights and biases of all 
        # the trainable layers.(which has weights and biases)

        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):
        # sets the parameters for all the layers
        # in this model

        for param_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*param_set)

    def save_parameters(self, file_path):
        '''
        Loads the trained parameters (weights, biases)
        into a file (persisting for loading in future).
        The parameters will be serialized using pickle.
        '''
        with open(file_path, 'wb') as fd:
            pickle.dump(self.get_parameters(), fd)

    def load_parameters(self, file_path):
        '''
        Loads the network parameters from a given file
        '''
        with open(file_path, 'rb') as fd:
            self.set_parameters(pickle.load(fd))

    def save(self, path):
        '''
        Saves the whole network model, itself instead of just
        weights and parameters.
        
        The loss, accuracy and gradients will be removed and the
        network layer and parameters (like weights, biases) will 
        be preserved.

        this way, the model can be loaded and also further trained.
        '''
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # reset the loss and accuracy.
        self.loss.new_pass()
        self.accuracy.new_pass()

        # remove the properties from input and loss layer.
        self.input_layer.__dict__.pop('output', None) # ignore if not present
        self.loss.__dict__.pop('dinputs', None)

        # for each of the hidden, activation layers remove the inputs, gradients etc...
        # except weights and biases and optimizer params
        for layer in model.layers:
            for property in [ 'inputs' , 'output' , 'dinputs' , 'dweights' , 'dbiases' ]:
                layer.__dict__.pop(property, None)

        # persist the model.
        with open(path, 'wb') as fd:
            pickle.dump(model, fd)

    @staticmethod
    def load(model_file):
        '''
        Loads a total model object from given file.
        '''
        with open(model_file,'rb') as fd:
            model = pickle.load(fd)
        
        return model

