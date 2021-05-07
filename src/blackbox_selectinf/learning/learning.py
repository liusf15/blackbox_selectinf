import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import norm
from selectinf.distributions.discrete_family import discrete_family


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


def train_epoch(Z, W, net, opt, criterion, batch_size=500):
    """
    :param Z: predictors
    :param W: labels
    :param net: nn
    :param opt: torch.optim
    :param criterion: BCEloss
    :param batch_size: batch size
    :return: losses
    """
    Z = torch.tensor(Z, dtype=torch.float)
    W = torch.tensor(W, dtype=torch.float)
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


def learn_select_prob(Z_train, W_train, lr=1e-3, num_epochs=1000, batch_size=500, savemodel=False, modelname="model.pt"):
    d = Z_train.shape[1]
    net = Net(d)
    opt = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    e_losses = []
    for e in range(num_epochs):
        e_losses.append(train_epoch(Z_train, W_train, net, opt, criterion, batch_size))
    if savemodel:
        torch.save(net.state_dict(), modelname)
    return net


def get_weight(net, target_theta, N_0, gamma):
    """

    :param net:
    :param target_theta: shape (ll, num_select)
    :param N_0:
    :param gamma:
    :return:
    """
    if target_theta.shape[1] == gamma.shape[1]:
        pass
    elif target_theta.shape[0] == gamma.shape[1]:
        target_theta = target_theta.T
    else:
        raise AssertionError("invalid shape of target_theta")
    if N_0.shape[0] != gamma.shape[0]:
        raise AssertionError("invalid shape of N_0 or gamma")
    ll = target_theta.shape[0]
    print(target_theta.shape)
    tmp = np.zeros(ll)
    for j in range(ll):
        tilde_d = N_0 + gamma @ target_theta[j, :]
        tilde_d = torch.tensor(tilde_d, dtype=torch.float)
        tmp[j] = net(tilde_d)
    return tmp


def get_CI(target_val, weight_val, target_var, observed_target):
    target_sd = np.sqrt(target_var)
    weight_val_2 = weight_val * norm.pdf((target_val - observed_target) / target_sd)
    exp_family = discrete_family(target_val.reshape(-1), weight_val_2.reshape(-1))
    interval = exp_family.equal_tailed_interval(observed_target, alpha=0.05)
    rescaled_interval = (interval[0] * target_var + observed_target,
                         interval[1] * target_var + observed_target)
    return rescaled_interval
