import numpy as np
from numpy import exp, log, sqrt
from numpy.linalg import inv
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import norm
from selectinf.distributions.discrete_family import discrete_family


# build/train neural network, compute selection probability, compute p-value
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 200)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(200, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        a4 = self.out(h3)
        y = self.out_act(a4)
        return y


def train_epoch(Z, W, net, opt, criterion, batch_size=50):
    """
    :param Z: predictors
    :param W: labels
    :param net: nn
    :param opt: torch.optim
    :param criterion: BCEloss
    :param batch_size: batch size
    :return: losses
    """
    net.train()
    losses = []
    for beg_i in range(0, Z.size(0), batch_size):
        x_batch = Z[beg_i:beg_i + batch_size, :]
        y_batch = W[beg_i:beg_i + batch_size, None]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        y_hat = net(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        opt.step()
        losses.append(loss.data.numpy())
    return losses[-1]
