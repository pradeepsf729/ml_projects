import numpy as np

from nn import *
from model import Model

# for data generation for testing.
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(samples=1000, classes=2)
y = y.reshape(-1, 1)

X_test, y_test = create_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

# lambda values for regularization
regularization_params = {'weight_regularizer_l2' : 5e-4, 
                        'bias_regularizer_l2' : 5e-4 }

# initialize the model
model = Model()

# add layers
model.add(Layer_Dense(2, 64, **regularization_params))
model.add(Activation_ReLU())
model.add(Layer_Dropout())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())

loss = Loss_BinaryCrossEntropy()
optimizer = Optimizer_Adam(decay = 5e-7)
accuracy = Accuracy_Categorical()

model.set(loss=loss, optimizer=optimizer, accuracy=accuracy)

model.finalize()

model.train(X, y, 
            validation_data=(X_test, y_test),
            epochs=10000, print_per_epoch=1000)
