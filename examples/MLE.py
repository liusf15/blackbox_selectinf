from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt


# generate original data
# K = 30
K = 50
n = 1000
m = 500
X_dim = n * K + m
np.random.seed(13)
mu_list = np.random.rand(K) * 2 - 1  # mu: in the range [-1, 1]
mu_argsort = np.argsort(mu_list)
mu_list[mu_argsort[-2]] = mu_list[mu_argsort[-1]]
mu_list[mu_argsort[-3]] = mu_list[mu_argsort[-1]]
seed = 0
np.random.seed(seed)
X = np.zeros([K, n])
for k in range(K):
    X[k, :] = np.random.randn(n) + mu_list[k]
X_bar = np.mean(X, 1)
max_rest = np.sort(X_bar)[-2]
win_idx = np.argmax(X_bar)
X_2 = np.random.randn(m) + mu_list[win_idx]
Z = np.mean(X, 1)
Z = np.concatenate([Z, np.mean(X_2).reshape(1, )])


def get_flat(X, X_2):
    X_flat = X.reshape(-1)
    X_flat = np.concatenate([X_flat, X_2.reshape(-1)])
    X_flat = torch.tensor(X_flat, dtype=torch.float)
    return X_flat


X_flat = get_flat(X, X_2)
ntrain = 100000
n_b = n
m_b = m
X_train_tmp = []
Z_train_tmp = []
X_pooled = np.concatenate([X[win_idx], X_2])
for i in range(ntrain):
    X_b = np.zeros([K, n_b])
    for k in range(K):
        if k != win_idx:
            X_b[k, :] = X[k, np.random.choice(n, n_b, replace=True)]
        if k == win_idx:
            X_b[k, :] = X_pooled[np.random.choice(n + m, n_b, replace=True)]
    X_2_b = X_pooled[np.random.choice(n + m, m_b, replace=True)]
    idx = np.argmax(np.mean(X_b, 1))
    X_train_tmp.append(get_flat(X_b, X_2_b))
    Z_b = np.mean(X_b, 1)
    Z_b = np.concatenate([Z_b, np.mean(X_2_b).reshape(1, )])
    Z_train_tmp.append(Z_b)

X_train = torch.cat(X_train_tmp).reshape(ntrain, X_dim)  # ntrain * input_dim
Z_train = torch.tensor(np.concatenate(Z_train_tmp).reshape(ntrain, len(Z)), dtype=torch.float)


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 200)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(200, X_dim)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        a4 = self.out(h3)
        # y = self.out_act(a4)
        y = a4
        return y


net = Net(input_dim=len(Z))
opt = optim.Adam(net.parameters(), lr=1e-3)
batch_size = 500


def train(X_train, model, opt, batch_size):
    model.train()
    losses = []
    ntrain = X_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        Z_batch = Z_train[beg_i:beg_i + batch_size, :]
        Z_batch = Variable(Z_batch)
        X_batch = X_train[beg_i:beg_i + batch_size, :]
        X_batch = Variable(X_batch)
        opt.zero_grad()
        recon_batch = model(Z_batch)
        loss = F.mse_loss(recon_batch, X_batch, reduction='sum')
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


Z_t = torch.tensor(Z, dtype=torch.float)
losses = []
num_epochs = 100
for epoch in range(num_epochs):
    ll = train(X_train, net, opt, batch_size)
    losses.append(ll)
    if epoch % 1 == 0:
        print(epoch, ll)
        print(torch.linalg.norm(net(Z_t) - X_flat))
        torch.save(net.state_dict(), "MLE_model_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))

plt.plot(losses)
plt.yscale('log')
plt.savefig("losses.png")

net.load_state_dict(torch.load("MLE_model_seed_0_n_1000_K_50_m_500.pt"))
# sample data




