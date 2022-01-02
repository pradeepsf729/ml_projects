import numpy as np
from model import Model
from nn import *
import matplotlib.pyplot as plt

# Sine sample dataset
def create_data(samples=1000):

    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)

    return X, y

X, y = create_data(samples=10000)
X_test, y_test = create_data(samples=100)

model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError(),
          optimizer=Optimizer_Adam(),
          accuracy=Accuracy_Regression())

model.finalize()

model.train(X, y,
            validation_data=(X_test, y_test),
            epochs=10000, print_per_epoch=1000)
