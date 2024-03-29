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


def learn_select_prob(Z_train, W_train, lr=1e-3, Z_data=None, net=None, thre=.99, consec_epochs=2, num_epochs=1000, batch_size=500, savemodel=False, modelname="model.pt", verbose=False, print_every=1000):
    d = Z_train.shape[1]
    if net is None:
        net = Net(d)
    opt = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    e_losses = []
    count = 0
    flag = 0  # indicate success of training
    pr_data = None
    for e in range(num_epochs):
        e_losses.append(train_epoch(Z_train, W_train, net, opt, criterion, batch_size))
        if Z_data is not None:
            pr_data = net(Z_data)
            if pr_data >= thre:
                count += 1
            else:
                count = 0
            if count == consec_epochs:  # exceeds the threshold in 5 consecutive epochs
                flag = 1
                print("pr_data", pr_data, "stop training at epoch", e)
                break
        if verbose and e % print_every == 0:
            print("Epochs {} out of {}, loss={}, pr_data={}".format(e, num_epochs, e_losses[-1], pr_data))
        if e_losses[-1] == 0.:
            print("zero loss, break")
            break
    if Z_data is not None:
        pr_data = net(Z_data)
        print("pr_data", pr_data.item())
    if savemodel:
        torch.save(net.state_dict(), modelname)
    if Z_data is None:
        return net
    else:
        return net, flag, pr_data


def check_learning(net, lassoClass, N_0, gamma, nb, maxiter):
    X = lassoClass.X
    Y = lassoClass.Y
    n = lassoClass.n
    num_select = lassoClass.num_select
    target_var = np.diag(lassoClass.Sigma1)
    target_sd = np.sqrt(target_var)
    theta_data = lassoClass.test_statistic(X, Y)
    gamma_list = np.linspace(-3 * target_sd, 3 * target_sd, 101)
    n_b = n
    count = 0
    pval = [[] for x in range(num_select)]
    for ell in range(maxiter):
        idx_b = np.random.choice(n, n_b, replace=True)
        X_b = X[idx_b, :]
        Y_b = Y[idx_b]
        if not np.all(lassoClass.select(X_b, Y_b) == lassoClass.sign):
            continue
        else:
            count += 1
            d_M = lassoClass.test_statistic(X_b, Y_b)
            observed_target = d_M
            for k in range(num_select):
                target_theta_k = d_M[k] + gamma_list[:, k]
                target_theta_0 = np.tile(d_M, [101, 1])
                target_theta_0[:, k] = target_theta_k
                weight_val_0 = get_weight(net, target_theta_0, N_0, gamma)
                weight_val_2 = weight_val_0 * norm.pdf((target_theta_0[:, k] - observed_target[k]) / target_sd[k])
                exp_family = discrete_family(target_theta_0.reshape(-1), weight_val_2.reshape(-1))
                hypothesis = theta_data[k]
                pivot = exp_family.cdf((hypothesis - observed_target[k]) / target_var[k], x=observed_target[k])
                pivot = 2 * min(pivot, 1 - pivot)
                pval[k].append(pivot)
            if count == nb:
                break
        return pval


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
    tmp = np.zeros(ll)
    for j in range(ll):
        tilde_d = N_0 + gamma @ target_theta[j, :]
        tilde_d = torch.tensor(tilde_d, dtype=torch.float)
        tmp[j] = net(tilde_d)
    return tmp


def get_CI(target_val, weight_val, target_var, observed_target, return_pvalue=False):
    target_sd = np.sqrt(target_var)
    weight_val_2 = weight_val * norm.pdf((target_val - observed_target) / target_sd)
    exp_family = discrete_family(target_val.reshape(-1), weight_val_2.reshape(-1))
    try:
        interval = exp_family.equal_tailed_interval(observed_target, alpha=0.05)
    except:
        interval = [np.nan, np.nan]
    rescaled_interval = (interval[0] * target_var + observed_target,
                         interval[1] * target_var + observed_target)
    if not return_pvalue:
        return rescaled_interval
    else:
        pivot = exp_family.cdf((0 - observed_target) / target_var, x=observed_target)
        pval = 2 * min(pivot, 1 - pivot)
        return rescaled_interval, pval
