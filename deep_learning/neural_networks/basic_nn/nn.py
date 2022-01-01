import numpy as np

class Layer_Dense:
    '''
    Indicates  one layer in NN
    '''
    def __init__(self,
                n_inputs,
                n_neurons,
                weight_regularizer_l1 = 0,
                weight_regularizer_l2 = 0,
                bias_regularizer_l1 = 0,
                bias_regularizer_l2 = 0):
        '''
        Initiliazes a hindden dense layer in a network
        - n_inputs - Number of features in one input sample.
        - n_neurons - Number of neurons in the layer.
        '''

        # the shape of the weights needs to be x,y
        # where x - number of neurons, y - number of features in one sample.
        # but here we initiliaze as (input_features, neurons) to avoid doing the transpose on forward pass.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # lambda parameter for L1 and L2 regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        '''
        Calculates one forward pass with this network weights
        and biases with given inputs.
        - inputs - inputs to be used.
        '''
        # input values will be used in reverse pass on backprop.
        self.inputs = inputs

        # Calculate output values from inputs, weights and biases
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        '''
        Calculates the gradients for weights and biases,
        The parameters of this layer will be updated with these by
        the optimizer.
        - dinputs - the gradient values of the next layer acquired from chain rule.
        '''
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # update with regularization derivative also.
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            dL2 = 2 * self.weight_regularizer_l2 * self.weights
            self.dweights += dL2

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            dL2 = 2 * self.bias_regularizer_l2 * self.biases
            self.dbiases += dL2

        # Gradients on values.
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    '''
    Performs ReLU activation on any given layer.
    '''
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        # based on the property of ReLU function,
        # ReLU(x < 0) = 0
        # ReLU(x > 0) = x
        self.dinputs[self.inputs < 0] = 0

class Activation_Softmax:
    '''
    Class to perform Softmax on given inputs
    '''
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        each_row_sum = np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = exp_values / each_row_sum

    def backward(self, dvalues):
        '''
        For calculating the gradients for this layer,
        called during backpropagation.
        '''
        # creates empty array.
        self.dinputs = np.empty_like(dvalues)

        temp = zip(self.outputs, dvalues)
        for index, (single_sample_ouput, single_dvalues) in enumerate(temp):
            # Flatten output array
            single_sample_ouput = single_sample_ouput.reshape( - 1 , 1 )

            # derivative of softmax ouput for this sample
            jacobian_matrix = \
                    np.diagflat(single_sample_ouput) - np.dot(single_sample_ouput, single_sample_ouput.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues *  (1.0 - self.outputs) * self.outputs

class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses) # average loss of all samples.

    def regularization_loss ( self , layer ):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss):
    '''
    Class to implement CrossEntropy Loss.
    '''
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)

        # clipping to avoid divide-by-zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # if already one-hot encoded, handle it
        if len(y_true.shape) == 1:
            # here mean, not one hot encoded, we just needs to pick the index of class label
            sample_range = range(n_samples)
            correct_confidences \
                = y_pred_clipped[sample_range, y_true]
        elif len(y_true.shape) == 2:
            # y_true is one hot encoded, multiply so that
            # other class labels gets nullified by mulitpliying with zero
            correct_confidences \
                = np.sum(y_pred_clipped * y_true , axis=1)
        
        neg_log_likelyhood = -np.log(correct_confidences)
        return neg_log_likelyhood

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len (dvalues)

        # Number of labels in every sample
        labels = len (dvalues[ 0 ])

        # If labels are sparse, turn them into one-hot vector
        if len (y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossEntropy(Loss):
    def forward(self,  y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )

        sample_losses = -((y_true * np.log(y_pred_clipped)) + ((1 - y_true) * np.log(1 - y_pred_clipped)))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        n_samples = len (dvalues)

        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len (dvalues[ 0 ])

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7 , 1 - 1e-7 )

        # Calculate gradient
        self.dinputs = - ((y_true / clipped_dvalues) - (( 1 - y_true) / ( 1 - clipped_dvalues))) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / n_samples

    def regularization_loss (self , layer):
        return super().regularization_loss(layer)

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    '''
    Instead of using separate classes for Softmax and Entropy loss
    Combining both of these operations for formward and backward pass.
    '''
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):

        # calculate the softmax.
        self.activation.forward(inputs)
        self.output = self.activation.outputs

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # total sample inputs
        n_samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1 )
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # actual gradient calculation.
        self.dinputs[range (n_samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / n_samples

    def regularization_loss (self , layer):
        return self.loss.regularization_loss(layer)

class Optimizer_SGD:
    '''
    Describes the StochasticgradientDescent optimizer and its operations
    to updates the weights as part of neural network training.

    For SGD - learning rate default value is 1.0
    '''
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def preupdate_params(self):
        '''
        Updates the learning to be used for the params update.
        Gets invoked before updating the weights.
        '''
        if self.decay:
            new_lr = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))
            self.current_learning_rate = new_lr

    def update_params(self, layer):
        '''
        Invoked to updates the weights and biases of the given layer.
        '''
        # currently used learning rate
        cur_lr = self.current_learning_rate
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                # for first update init with zeros
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            # take previous updates multiplied by 
            # retain factor and update with
            # current gradients
            weight_updates = \
                (self.momentum * layer.weight_momentums) - (cur_lr * layer.dweights)
            
            # store them to use for next iter
            layer.weight_momentums = weight_updates
    
            bias_updates = \
                (self.momentum * layer.bias_momentums) - (cur_lr * layer.dbiases)

            layer.bias_momentums = bias_updates
        else:
            # without using momentum, simple regular update with learning rate
            weight_updates = -(cur_lr * layer.dweights)
            bias_updates = -(cur_lr * layer.dbiases)

        layer.weights = layer.weights + weight_updates
        layer.baises = layer.biases + bias_updates

    def postupdate_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def preupdate_params(self):
        '''
        Updates the learning to be used for the params update.
        Gets invoked before updating the weights.
        '''
        if self.decay:
            new_lr = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))
            self.current_learning_rate = new_lr

    def update_params(self, layer):
        '''
        Invoked to updates the weights and biases of the given layer.
        '''
        # currently used learning rate
        cur_lr = self.current_learning_rate

        if not hasattr(layer, 'weight_cache'):
            # for first update init with zeros
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # take previous updates multiplied by 
        # retain factor and update with
        # current gradients
        weight_updates = \
            (cur_lr * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)

        bias_updates = \
            (cur_lr * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights = layer.weights - weight_updates
        layer.baises = layer.biases - bias_updates

    def postupdate_params(self):
        self.iterations += 1

class Optimizer_RmsProp:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def preupdate_params(self):
        if self.decay:  # normal SGD with learning rate decay update.
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr (layer, 'weight_cache'):
            # first epoch
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (self.rho * layer.weight_cache) + ((1 - self.rho) * layer.dweights ** 2)
        layer.bias_cache = (self.rho * layer.bias_cache) + ((1 - self.rho) * layer.dbiases ** 2)

        weight_updates = \
            (self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        bias_updates = \
            (self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights = layer.weights - weight_updates
        layer.biases  = layer.biases - bias_updates

    def postupdate_params(self):
        self.iterations += 1

class Optimizer_Adam:
    '''
    Class for ADamOptimizer - Adaptive Momentum optimizer.
    '''
    def __init__(
                self,
                learning_rate=0.001,
                decay=0.0,
                epsilon=1e-7,
                beta_1 = 0.9,
                beta_2 = 0.999):
        '''
        Initlizae the hyper parameters with default values.
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def preupdate_params(self):
        if self.decay:  # normal SGD with learning rate decay update.
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr (layer, 'weight_cache'):
            # first epoch
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums) + (( 1 - self.beta_1) * layer.dweights)
        layer.bias_momentums = (self.beta_1 * layer.bias_momentums) + (( 1 - self.beta_1) * layer.dbiases)

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))
        bias_momentums_corrected = layer.bias_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))

        layer.weight_cache = (self.beta_2 * layer.weight_cache) + ((1 - self.beta_2) * layer.dweights ** 2)
        layer.bias_cache = (self.beta_2 * layer.bias_cache) + ((1 - self.beta_2) * layer.dbiases ** 2)

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        bias_cache_corrected = layer.bias_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))

        weight_updates = \
            (self.current_learning_rate * weight_momentums_corrected) / (np.sqrt(weight_cache_corrected) + self.epsilon)
        bias_updates = \
            (self.current_learning_rate * bias_momentums_corrected) / (np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights = layer.weights - weight_updates
        layer.biases  = layer.biases - bias_updates

    def postupdate_params(self):
        self.iterations += 1

class Layer_Dropout:
    '''
    Class to implement the dropout for any layer
    '''
    def __init__(self, rate):
        self.rate = 1 - rate
    
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial( 1 , self.rate, size = inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
