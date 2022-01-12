import torch
import torch.optim as optim

'''
NN model to fit the relation between the actual readings
and measured readings.
'''

# actual readings
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]

# measured readings
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w1, w2 ,b):
    '''
    line-fit model
    '''
    #return w * t_u + b
    return w2 * t_u ** 2 + w1 * t_u + b

def loss_fn(t_p, t_c):
    '''
    mean squared loss
    '''
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1

def grad_fn(t_u, t_p, w, b):
    '''
    total gradient per one backward pass
    '''
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b) # chain rule
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b) # chain rule
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

# initialize the params on start
params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)

def train(n_epochs, optimizer, params, t_u, t_c):
    '''
    training the model for given epochs.
    '''
    for epoch in range(1, n_epochs + 1):
        # with autograd on every backward pass, the gradient on tensors
        # gets accumulated (summed up), reset to zero before next backward pass
        if params.grad is not None:
            params.grad.zero_()

        w1, w2, b = params
        t_p = model(t_u, w1, w2, b)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        # with torch.no_grad:
        #     grad = params.grad
        #     params = params - learning_rate * grad

        optimizer.step()

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

t_un = 0.1 * t_u

learning_rate = 1e-2
#optimizer = optim.SGD([params], lr=learning_rate)
optimizer = optim.Adam([params], lr=learning_rate)
train(5000, optimizer, params, t_un, t_c)

