'''
Linear regression modelling using neural networks with pytorch.

Details:
Given different combination of wine contents and the given quality metric,
predict the quality of the wine for any newly given combination
'''

from scipy.sparse.construct import random
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('winequality-white.csv', sep=';')

output_col = 'quality'
print('Total data - ', df.shape)
print('Output column - ', output_col)

X = df.drop(output_col, axis=1).copy()
y = df[output_col].copy()

# prep train, test data
result = train_test_split(X, y)
X_train, X_test = result[0], result[1]
y_train, y_test = result[2], result[3]

# load into tensors
tr_ip = torch.tensor(X_train.values)
tr_ip = tr_ip.to(torch.float32)

tr_op =  torch.tensor(y_train.values)
tr_op = tr_op.to(torch.float32)
tr_op = tr_op.unsqueeze(1)

test_ip = torch.tensor(X_test.values)
test_ip = test_ip.to(torch.float32)

test_op = torch.tensor(y_test.values)
test_op = test_op.to(torch.float32)
test_op = test_op.unsqueeze(1)

# setup the model
model = nn.Sequential(
    nn.Linear(11, 128),
    nn.Tanh(),
    nn.Linear(128, 1))

print('NN Model - ', model)

# setup loss, optimizer and softmax
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)
optimizer = optim.Adam(model.parameters())

# train
def train(tr_ip, tr_op, model, loss, optimizer, epochs):
    for epoch in range(1, epochs + 1):
        tr_predicts = model(tr_ip)
        tr_predicts = tr_predicts.to(torch.float32)

        # covert to grad tensor
        tr_predicts = tr_predicts.clone().detach().requires_grad_(True)
        tr_op = tr_op.clone().detach().requires_grad_(True)

        # calculate the loss and backward prop
        optimizer.zero_grad()
        loss_val = loss(tr_predicts, tr_op)
        loss_val.backward()

        # update the parameters for next step
        optimizer.step()

        if epoch % 5000 == 0:
            print("epoch ", epoch , ' Loss - ', loss_val)

train(tr_ip, tr_op, model, loss, optimizer, 30000)

# check accuracy with test data
test_pred = model(test_ip)
test_pred = test_pred.to(torch.float32)
test_loss = loss(test_pred, test_op)

print('Loss on test data - ', test_loss)
