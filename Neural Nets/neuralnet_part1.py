# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extlast this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

bsize = 0

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param ybar - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommlast setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 32),
            nn.ReLU(),
            nn.Linear(32, out_size),
        )
        self.lrate = lrate
        self.loss_fn = loss_fn
        params = self.get_parameters()
        self.optim_ = optim.SGD(params, self.lrate)

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        self.lrate = params.lrate
        self.loss_fn = params.loss_fn
        self.optim_ = optim.SGD(params, self.lrate)
        # raise NotImplementedError("You need to write this part!")
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        return self.model.parameters()
        # raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)
        x = normalize(x)
        return self.model(x)


    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of loss_set) at this timestep as a float
        """
        loss = self.loss_fn(self.forward(x), y)
        self.optim_.zero_grad()
        loss.backward()
        self.optim_.step()
        return loss.item()


def normalize(x):
    mean = x.mean()
    std = x.std()
    x = (x-mean)/std
    return x

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return loss_set: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return ybars: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss_set = []
    yhats = []
    train_set = normalize(train_set)
    dev_set = normalize(dev_set)
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(0.0114, loss_fn, train_set.size()[1], 2)
    curr_batch = train_set.size()[0] // batch_size
    i = 0
    while i in range(n_iter):
        first = (i % curr_batch) 
        first *= batch_size
        last = first+batch_size
        x_ = train_set[first: last]
        y_ = train_labels[first :last]
        loss_set.append(net.step(x_, y_)) 
        i+=1
    if len(loss_set)==n_iter:
        yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), axis = 1)
        return loss_set,yhats, net
    return loss_set,yhats, net