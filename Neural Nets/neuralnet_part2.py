# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 0
class2 = 19

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(6, 6, 3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Flatten(),
            # nn.MaxPool3d(2),
            nn.Linear(6144, 16),
            nn.LeakyReLU(),
            nn.Linear(16, out_size),
        )
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.optim_ = optim.Adam(self.get_parameters(), lrate)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return (self.model.parameters())
        # return []


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        normalize(x)
        x = x.view(-1, 3,32,32)
        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        normalize(x)
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        self.optim_.zero_grad()
        loss.backward()
        self.optim_.step()
        return loss.item()

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss_set = []
    yhats = []
    i=0
    train_set = normalize(train_set)
    dev_set = normalize(dev_set)
    loss_fn = nn.CrossEntropyLoss()
    setlen = len(train_set)
    net = NeuralNet(0.0004, loss_fn, train_set.size()[1], 2)
    curr_batch = train_set.size()[0] // batch_size
    first = np.random.randint(0, setlen)

    while i in range(n_iter):
        # first = (i % curr_batch) 
        first *= batch_size
        last = first+batch_size
        # first = np.random.randint(0, setlen)
        # last = np.random.randint(0, setlen)
        loss_set.append(net.step(train_set[first: last+first], train_labels[first :last+first]))
        i+=1

    yhats = np.argmax(net.forward(dev_set).detach().numpy(), axis = 1).tolist()
    return loss_set ,yhats, net

def normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x-mean)/std
    return x
