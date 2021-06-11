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
from blackbox_selectinf.usecase.DTL import DropTheLoser
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)


# generate original data
K = 50
n = 1000
m = 500
# K = 5
# n = 100
# m = 50
np.random.seed(1)
mu_list = np.random.rand(K) * 2 - 1  # mu: in the range [-1, 1]
# mu_list = np.array([1, 1, .9, .8, .7])
mu_argsort = np.argsort(mu_list)
mu_list[mu_argsort[-2]] = mu_list[mu_argsort[-1]]
# mu_list[mu_argsort[-3]] = mu_list[mu_argsort[-1]]
seed = 1
np.random.seed(seed)
X = np.zeros([K, n])
for k in range(K):
    X[k, :] = np.random.randn(n) + mu_list[k]
X_bar = np.mean(X, 1)
max_rest = np.sort(X_bar)[-2]
win_idx = np.argmax(X_bar)
X_2 = np.random.randn(m) + mu_list[win_idx]

DTL_class = DropTheLoser(X, X_2)
Z_data = DTL_class.basis(X, X_2)

ntrain = 10000
n_b = n
m_b = m
basis_type = 'withD0'
training_data = DTL_class.gen_train_data(ntrain=ntrain, n_b=n_b, m_b=m_b, basis_type=basis_type)
Z_train = training_data['Z_train']
W_train = training_data['W_train']
gamma = training_data['gamma']
print(np.mean(W_train))

pos_ind = W_train == 1
Z_pos = torch.tensor(Z_train[pos_ind, :], dtype=torch.float)
W_pos = W_train[pos_ind]
input_dim = Z_pos.shape[1]
bottleneck_dim = 10


# train VAE
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 30)
        self.fc21 = nn.Linear(30, bottleneck_dim)
        self.fc22 = nn.Linear(30, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, 30)
        self.fc4 = nn.Linear(30, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(input_dim=input_dim)
opt = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 128


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    loglik = F.mse_loss(recon_x, x, reduction='sum')
    # loglik = nn.MSELoss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loglik + KLD


def train(X_train, model, opt, batch_size):
    model.train()
    losses = []
    ntrain = X_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        opt.zero_grad()
        recon_batch, mu, logvar = model(x_batch)
        loss = loss_function(recon_batch, x_batch, mu, logvar)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    ll = train(Z_pos, model, opt, batch_size)
    losses.append(ll)
    if epoch % 100 == 0:
        print(epoch, ll)
        z = torch.randn(200, bottleneck_dim)
        z_dec = model.decode(z)
        print(torch.all(torch.argmax(z_dec[:, :50], 1) == win_idx))
        # print(max(abs(torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar)))
        torch.save(model.state_dict(), "VAE_model_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))

torch.save(model.state_dict(), "VAE_basis_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))
plt.plot(np.array(losses))
plt.yscale('log')

model.load_state_dict(torch.load("VAE_model_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m)))

# plt.figure()
# plt.hist(Z_pos[:, 0].numpy(), density=True)
# plt.hist(z_dec[:, 0].detach().numpy(), density=True)
#
# torch.std(Z_pos[:, 0])
# torch.std(z_dec[:, 0])

output_dim = n * K + m


# train a decoder
class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 5000)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(5000, output_dim)
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


def recon_loss(Z_gen, X_dec):
    X_dec_1 = X_dec[:, :n * K].reshape(-1, K, n)
    X_dec_2 = X_dec[:, n * K:].reshape(-1, m)
    Z_dec = torch.mean(X_dec_1, 2)
    Z_dec = torch.cat([Z_dec, torch.mean(X_dec_2, 1).view(-1, 1)], 1)
    # Z_dec[:, win_idx] = torch.cat([X_dec_1[:, win_idx, :], X_dec_2], 1).mean()
    return torch.sum((Z_gen - Z_dec)**2) / Z_gen.shape[0]


def train_decoder(Z_train, model, opt, batch_size):
    model.train()
    losses = []
    ntrain = Z_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        x_batch = Z_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        opt.zero_grad()
        X_recon = model(x_batch)
        loss = recon_loss(x_batch, X_recon)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


decoder = Decoder(input_dim)
opt_dec = optim.Adam(decoder.parameters(), lr=1e-3)
batch_size = 128

# Z_gen = torch.tensor(Z_pos, dtype=torch.float)
Z_gen = model.decode(torch.randn([5000, bottleneck_dim]))
print(torch.all(torch.argmax(Z_gen, 1) == win_idx))
# X_dec = decoder(Z_gen)
losses = []
for epoch in range(20):
    ll = train_decoder(Z_gen, decoder, opt_dec, batch_size)
    losses.append(ll)
    if epoch % 1 == 0:
        print(epoch, ll)
        torch.save(decoder.state_dict(), "decoder_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))

plt.plot(losses)
decoder.load_state_dict(torch.load('decoder_seed_8_n_1000_K_50_m_500.pt'))

z_tmp = model.decode(torch.randn(200, bottleneck_dim))
print(torch.all(torch.argmax(z_tmp, 1) == win_idx))
# z_tmp = torch.tensor(Z_data, dtype=torch.float)
x_tmp = decoder(z_tmp).detach().numpy()
x_tmp_1 = x_tmp[:, :n * K].reshape(-1, K, n)
print(np.all(np.argmax(np.mean(x_tmp_1, 2), 1) == win_idx))
print(np.mean(x_tmp_1, (0, 2))-X_bar)


