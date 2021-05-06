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


def basis_lr(X, Y):
    """
    :param X: design matrix
    :param Y: response
    :return: basis
    """
    XY = X.T @ Y / len(Y)
    XX = np.mean(X, 0)
    return np.concatenate([XY, XX])


def estimate_proj_Z_D0_lr(X, Y, E, n_b=500):
    n, p = X.shape
    Z_list = []
    D0_list = []
    m = n_b
    S = 1000
    for i in range(S):
        idx_b = np.random.choice(n, m, replace=True)
        X_b = X[idx_b, :]
        Y_b = Y[idx_b]
        # X_b = np.random.randn(m, p)
        # Y_b = np.random.binomial(1, 1 / (1 + exp(- X_b[:, E] @ beta_ls)), m)
        Z_b = basis_lr(X_b, Y_b)
        Z_list.append(Z_b)

        logreg = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
        logreg.fit(X_b[:, E], Y_b)
        beta_ls_b = logreg.coef_.squeeze()
        if beta_ls_b.size == 1:
            beta_ls_b = beta_ls_b.reshape(1, )
        pr_b = 1 / (1 + exp(-X_b[:, E] @ beta_ls_b)).reshape(len(Y_b))
        D0_list.append(X_b[:, ~E].T @ (Y_b - pr_b) / sqrt(m))

    Z_list = np.array(Z_list)
    Z_list_centered = Z_list - np.mean(Z_list, 0)[None, :]
    D0_list = np.array(D0_list)
    D0_list_centered = D0_list - np.mean(D0_list, 0)[None, :]
    cov_Z_D_0 = Z_list_centered.T @ D0_list_centered / S  # scale as 1/sqrt(m)
    var_D_0 = np.cov(D0_list.T)
    if np.sum(~E) == 1:
        var_D_0 = var_D_0.reshape(1, 1)
    proj_Z_D0 = cov_Z_D_0 @ np.linalg.inv(var_D_0)
    return proj_Z_D0


def basis_lr_indep(X, Y, E, proj):
    Z = basis_lr(X, Y)
    logreg = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
    logreg.fit(X[:, E], Y)
    beta_ls_new = logreg.coef_.squeeze()
    if beta_ls_new.size == 1:
        beta_ls_new = beta_ls_new.reshape(1, )
    pr_b = 1 / (1 + exp(-X[:, E] @ beta_ls_new)).reshape(len(Y))
    D_0_new = X[:, ~E].T @ (Y - pr_b) / sqrt(len(Y))
    return Z - proj @ D_0_new  # np.concatenate([Z - proj_Z_D0 @ D_0_new, D_0_new])


def basis_dtl(X, win_idx, X_win_2):
    """
    :param X: Stage 1 data
    :param win_idx: winner index
    :param X_win_2: Stage 2 data
    :return: basis
    """
    d_M = (np.sum(X[win_idx, :]) + np.sum(X_win_2)) / (len(X[win_idx, :]) + len(X_win_2))
    return np.concatenate([np.mean(X, 1), np.mean(X ** 2, 1), [d_M]])


def basis_dtl_naive(X, win_idx, X_win_2):
    d_M = (np.sum(X[win_idx, :]) + np.sum(X_win_2)) / (len(X[win_idx, :]) + len(X_win_2))
    tmp = np.mean(X, 1)
    tmp[win_idx] = d_M
    return np.concatenate([tmp, np.array([1])])


def estimate_proj_Z_D0_dtl(X, win_idx, X_win_2):
    n_k = X.shape[1]
    m = len(X_win_2)
    mu_est = (np.sum(X[win_idx, :]) + np.sum(X_win_2)) / (len(X[win_idx, :]) + len(X_win_2))
    var_0 = 1 / n_k - 1 / (n_k + m)
    proj = m / (n_k + m) / n_k * (mu_est * 2) / var_0
    return proj


def basis_dtl_indep(X, win_idx, X_win_2, proj):
    K = X.shape[0]
    # proj = estimate_proj_Z_D0_dtl(X, win_idx, X_win_2)
    Z = np.concatenate([np.mean(X, 1), np.mean(X ** 2, 1)])
    d_M = (np.sum(X[win_idx, :]) + np.sum(X_win_2)) / (len(X[win_idx, :]) + len(X_win_2))
    d_0 = np.mean(X[win_idx, :]) - d_M
    Z[win_idx] = d_M
    Z[win_idx + K] = np.mean(X[win_idx, :] ** 2) - proj * d_0
    return Z


def gen_train_data_lr(X, Y, s, ntrain, n_b, logistic=False, seed=1, alpha=0.1, indep=False, proj=None):
    """
    :param X: original design matrix, (n, p)
    :param Y: original response, (n, )
    :param s: sign of lasso estimator
    :param ntrain: total number of training data
    :param n_b: number of datapoints to bootstrap
    :param seed: random seed
    :param alpha: parameter for LogisticRegression l1 penality
    :return: Z_train, W_train
    """
    n = X.shape[0]
    Z_train = []
    W_train = []
    np.random.seed(seed)
    for i in range(ntrain):
        idx_b = np.random.choice(n, n_b, replace=True)
        X_b = X[idx_b, :]
        Y_b = Y[idx_b]
        if logistic:
            lasso = LogisticRegression(penalty='l1', C=alpha, solver='liblinear', fit_intercept=False, tol=1e-7)
        else:
            lasso = Lasso(alpha=alpha, fit_intercept=False)
        lasso.fit(X_b, Y_b)
        E = s != 0
        # W_train.append(np.all((lasso.coef_ != 0) == E))
        W_train.append(np.all(np.sign(lasso.coef_) == s))
        if not indep:
            Z_train.append(basis_lr(X_b, Y_b))
        else:
            Z_train.append(basis_lr_indep(X_b, Y_b, E, proj))
    Z_train = torch.tensor(np.array(Z_train), dtype=torch.float)
    W_train = torch.tensor(np.array(W_train), dtype=torch.float)
    return Z_train, W_train


def gen_train_data_dtl(X, win_idx, X_win_2, ntrain, n_b, m_b, proj=None, seed=1, indep=False, UC=False, naive=False,
                       mix=False, winner_only=False):
    """
    :param X: original Stage 1 data
    :param win_idx: winner index
    :param X_win_2: original Stage 2 data
    :param ntrain: total number of training data
    :param n_b: number of datapoints to bootstrap in Stage 1
    :param m_b: number of datapoints to bootstrap in Stage 2
    :param proj: Cov(bar x^2) / Var(D_0)
    :param seed: random seed
    :return: Z_train, W_train
    """
    [K, n_k] = X.shape
    m = len(X_win_2)
    Z_train = []
    W_train = []
    X_cat = np.concatenate([X[win_idx], X_win_2])
    np.random.seed(seed)
    if not winner_only:
        for i in range(ntrain):
            X_b = np.zeros([K, n_b])
            for k in range(K):
                if k != win_idx:
                    X_b[k, :] = X[k, np.random.choice(n_k, n_b, replace=True)]
                if k == win_idx:
                    if mix:
                        X_b[k, :] = X_cat[np.random.choice(n_k + m, n_b, replace=True)]
                    else:
                        X_b[k, :] = X[k, np.random.choice(n_k, n_b, replace=True)]
            # selection
            if not UC:
                idx = np.argmax(np.mean(X_b, 1))
            else:
                idx = np.argmax(np.mean(X_b, 1) + 2 * np.std(X_b, 1, ddof=1))
            if idx == win_idx:
                W_train.append(1)
            else:
                W_train.append(0)
            # basis
            m = len(X_win_2)
            if not mix:
                x_win_2_new = X_win_2[np.random.choice(m, m_b, replace=True)]
            else:
                x_win_2_new = X_cat[np.random.choice(m + n_k, m_b, replace=True)]
            D_M_b = (np.sum(X_b[win_idx, :]) + np.sum(x_win_2_new)) / (len(X_b[win_idx, :]) + len(x_win_2_new))
            if not indep and not naive:
                Z_train.append(np.concatenate([np.mean(X_b, 1), np.mean(X_b ** 2, 1), [D_M_b]]))
            if indep:
                Z_train.append(basis_dtl_indep(X_b, win_idx, x_win_2_new, proj))
            if naive:
                Z_train.append(basis_dtl_naive(X_b, win_idx, x_win_2_new))
    else:  # only bootstrap the winner data
        X_bar = np.mean(X, 1)
        max_rest = np.sort(X_bar)[-2]
        for i in range(ntrain):
            if mix:
                X_b_1 = X_cat[np.random.choice(n_k + m, n_b, replace=True)]
                X_b_2 = X_cat[np.random.choice(n_k + m, m_b, replace=True)]
            else:
                X_b_1 = X[win_idx, np.random.choice(n_k, n_b, replace=True)]
                X_b_2 = X_win_2[np.random.choice(m, m_b, replace=True)]
            D_M_b = (np.sum(X_b_1) + np.sum(X_b_2)) / (n_b + m_b)
            if np.mean(X_b_1) >= max_rest:
                W_train.append(1)
            else:
                W_train.append(0)
            basis_new = X_bar
            basis_new[win_idx] = D_M_b
            basis_new = np.concatenate([basis_new, np.array([1])])
            Z_train.append(basis_new)

    Z_train = torch.tensor(np.array(Z_train), dtype=torch.float)
    W_train = torch.tensor(np.array(W_train), dtype=torch.float)
    return Z_train, W_train


def block_bootstrap(X, win_idx, X_win_2, ntrain, blocklength=10, seed=1, naive=True, mix=False, winner_only=False):
    [K, n_k] = X.shape
    m = len(X_win_2)
    Z_train = []
    W_train = []
    num_blocks = n_k
    num_blocks_resample = num_blocks // blocklength
    np.random.seed(seed)
    for i in range(ntrain):
        X_b = np.zeros([K, num_blocks_resample * blocklength])
        for k in range(K):
            indices = np.random.choice(n_k - blocklength, num_blocks_resample, replace=True)
            for s in range(num_blocks_resample):
                idx = indices[s]
                data_block = X[k, idx: idx + blocklength]
                X_b[k, s * blocklength: (s + 1) * blocklength] = data_block
        if np.max(np.mean(X_b, 1)) == np.mean(X_b[win_idx, :]):
            W_train.append(1)
        else:
            W_train.append(0)

        # stage 2
        num_blocks = m
        num_blocks_resample = num_blocks // blocklength
        indices = np.random.choice(m - blocklength, num_blocks_resample, replace=True)
        x_win_2_new = []
        for s in range(num_blocks_resample):
            idx = indices[s]
            x_win_2_new.extend(X_win_2[idx: idx + blocklength])
        x_win_2_new = np.array(x_win_2_new)
        if naive:
            Z_train.append(basis_dtl_naive(X_b, win_idx, x_win_2_new))
        else:
            Z_train.append(basis_dtl(X_b, win_idx, x_win_2_new))

    Z_train = torch.tensor(np.array(Z_train), dtype=torch.float)
    W_train = torch.tensor(np.array(W_train), dtype=torch.float)
    return Z_train, W_train


def sele_prob_tildeD(D, net):
    pr = net.relu1(D)
    pr = net.fc2(pr)
    pr = net.relu2(pr)
    pr = net.out(pr)
    pr = net.out_act(pr)
    return pr


def estimate_proj_tildeD_DM(X, Y, E, n_b, net, logistic=False, indep=False, proj_Z_D0=None):
    n = len(Y)
    m = n_b
    tilde_D_list = []
    D_M_list = []
    for i in range(1000):
        idx_b = np.random.choice(n, m, replace=True)
        X_b = X[idx_b, :]
        Y_b = Y[idx_b]
        if indep:
            Z_b = basis_lr_indep(X_b, Y_b, E, proj_Z_D0)
        else:
            Z_b = basis_lr(X_b, Y_b)
        Z_b = torch.tensor(Z_b, dtype=torch.float)
        tilde_D_list.append(net.fc1(Z_b).detach().numpy())
        if logistic:
            logreg = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
        else:
            logreg = LinearRegression(fit_intercept=False)
        logreg.fit(X_b[:, E], Y_b)
        D_M_list.append(logreg.coef_.squeeze())

    tilde_D_list = np.array(tilde_D_list)
    tilde_D_centered = tilde_D_list - np.mean(tilde_D_list, 0)[None, :]

    D_M_list = np.array(D_M_list)
    D_M_centered = D_M_list - np.mean(D_M_list, 0)

    cov_tilde_D_D_M = tilde_D_centered.T @ D_M_centered / 1000
    var_D_M = D_M_centered.T @ D_M_centered / 1000
    # var_D_M = Sigma11
    if np.sum(E) == 1:
        var_D_M = var_D_M.reshape(1, 1)
        cov_tilde_D_D_M = cov_tilde_D_D_M[:, None]
    proj = cov_tilde_D_D_M @ np.linalg.inv(var_D_M)
    return proj


def estimate_proj_Z_DM(X, Y, E, n_b, logistic=False):
    n = len(Y)
    Z_list = []
    D_M_list = []
    for i in range(1000):
        idx_b = np.random.choice(n, n_b, replace=True)
        X_b = X[idx_b, :]
        Y_b = Y[idx_b]
        Z_b = basis_lr(X_b, Y_b)
        Z_list.append(Z_b)
        if logistic:
            lr = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
        else:
            lr = LinearRegression(fit_intercept=False)
        lr.fit(X_b[:, E], Y_b)
        D_M_list.append(lr.coef_.squeeze())
    Z_list = np.array(Z_list)
    Z_centered = Z_list - np.mean(Z_list, 0)[None, :]
    D_M_list = np.array(D_M_list)
    D_M_centered = D_M_list - np.mean(D_M_list, 0)
    cov_Z_DM = Z_centered.T @ D_M_centered / 1000
    var_D_M = D_M_centered.T @ D_M_centered / 1000
    if np.sum(E) == 1:
        var_D_M = var_D_M.reshape(1, 1)
        cov_Z_DM = cov_Z_DM[:, None]
    proj = cov_Z_DM @ np.linalg.inv(var_D_M)
    return proj


def weight(target_val, net):
    tmp = []
    for j in range(target_val.shape[1]):
        tilde_d = target_val[:, j]
        tilde_d = torch.tensor(tilde_d, dtype=torch.float)
        tmp.append(net(tilde_d))
        # pr = net.relu1(tilde_d)
        # pr = net.fc2(pr)
        # pr = net.relu2(pr)
        # pr = net.out(pr)
        # pr = net.out_act(pr)
        # tmp.append(pr.detach().numpy())
    return tmp


# def weight_fn(Z_data_np, target_val, net, tar_idx=None):
#     Z_new = np.copy(Z_data_np)
#     Z_new = torch.tensor(Z_new, dtype=torch.float)
#     if tar_idx is None:
#         tar_idx = -1
#     weight_val = []
#     for tar in target_val:
#         Z_new[tar_idx] = tar
#         weight_val.append(net(Z_new))
#     weight_val = np.squeeze(weight_val)
#     return weight_val


def CI(target_val, weight_val, target_var, observed_target):
    target_sd = np.sqrt(target_var)
    weight_val_2 = weight_val * norm.pdf((target_val - observed_target) / target_sd)
    exp_family = discrete_family(target_val, weight_val_2)
    interval = exp_family.equal_tailed_interval(observed_target, alpha=0.05)
    rescaled_interval = (interval[0] * target_var + observed_target,
                         interval[1] * target_var + observed_target)
    return rescaled_interval

