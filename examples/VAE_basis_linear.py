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
from blackbox_selectinf.usecase.Lasso import LassoClass
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import pickle


n = 300
n_b = n
p = 10
input_dim = n * (p + 1)
# beta = np.array([0, 0, 5, 5]) / np.sqrt(n)
beta = np.zeros(p)
beta[:2] = 5 / np.sqrt(n)
seed = 1
np.random.seed(seed)
X = np.random.randn(n, p)
Y = X @ beta + np.random.randn(n)
lbd = 20
data_type = 'linear'
lassoClass = LassoClass(X, Y, lbd, data_type)
print(lassoClass.sign)

ntrain = 50000
lassoClass = LassoClass(X, Y, lbd, data_type)
training_data = lassoClass.gen_train_data(ntrain=ntrain, n_b=n_b)
Z_train = training_data['Z_train']
W_train = training_data['W_train']
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
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc21 = nn.Linear(20, bottleneck_dim)
        self.fc22 = nn.Linear(20, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, 20)
        self.fc4 = nn.Linear(20, input_dim)

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


vae_model = VAE(input_dim=input_dim)
opt = optim.Adam(vae_model.parameters(), lr=1e-3)
batch_size = 128

losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    ll = train(Z_pos, vae_model, opt, batch_size)
    losses.append(ll)
    if epoch % 100 == 0:
        print(epoch, ll)
        z = torch.randn(200, bottleneck_dim)
        z_dec = vae_model.decode(z)
        # print(max(abs(torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar)))
        torch.save(vae_model.state_dict(), "VAElinear_model_seed_{}_n_{}_p_{}.pt".format(seed, n, p))


output_dim = n * (p + 1)
plt.plot(losses)


# train a decoder
class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 1000)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(1000, output_dim)
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


def recon_loss(x_batch, X_recon):
    X_b = X_recon[:, :n * p].reshape(-1, n, p)
    Y_b = X_recon[:, n * p:].reshape(-1, n)
    Z1 = torch.mean(X_b, 1)
    Z2 = torch.einsum('ijk, ij -> ik', X_b, Y_b) / len(Y_b[0])
    Z_b = torch.cat([Z2, Z1], 1)
    return torch.sum((x_batch - Z_b)**2) / x_batch.shape[0]


def train_decoder(Z_train, decoder, opt, batch_size):
    decoder.train()
    losses = []
    ntrain = Z_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        x_batch = Z_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        opt.zero_grad()
        X_recon = decoder(x_batch)
        loss = recon_loss(x_batch, X_recon)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


decoder = Decoder(input_dim)
opt_dec = optim.Adam(decoder.parameters(), lr=1e-3)
batch_size = 128

# Z_gen = torch.tensor(Z_pos, dtype=torch.float)
Z_gen = vae_model.decode(torch.randn([1000, bottleneck_dim]))
Z_gen = torch.tensor(Z_pos, dtype=torch.float)
# X_dec = decoder(Z_gen)
losses = []
for epoch in range(1000):
    ll = train_decoder(Z_gen, decoder, opt_dec, batch_size)
    losses.append(ll)
    if epoch % 100 == 0:
        print(epoch, ll)
        torch.save(decoder.state_dict(), "linear_decoder_seed_{}_n_{}_p_{}.pt".format(seed, n, p))

        z_tmp = vae_model.decode(torch.randn(bottleneck_dim))
        x_tmp = decoder(z_tmp).detach().numpy()
        X_b = x_tmp[:n * p].reshape(n, p)
        Y_b = x_tmp[n * p:]
        print(lassoClass.select(X_b, Y_b))
        print(np.all(lassoClass.select(X_b, Y_b) == lassoClass.sign))

#######################################
# POSI
num_select = lassoClass.num_select
E = lassoClass.E
beta_E = beta[E]
ntrain = 1000
indep = False
training_data = lassoClass.gen_train_data(ntrain=ntrain, n_b=n_b, remove_D0=indep)
Z_train = training_data['Z_train']
W_train = training_data['W_train']
gamma = training_data['gamma']
Z_data = lassoClass.basis(X, Y)

# generate positive data using VAE and decoder
n_vae = 1000
z_tmp = vae_model.decode(torch.randn(n_vae, bottleneck_dim))
x_tmp = decoder(z_tmp).detach().numpy()
Z_train_vae = []
W_train_vae = []
for i in range(n_vae):
    X_b = x_tmp[i, :n * p].reshape(n, p)
    Y_b = x_tmp[i, n * p:].reshape(n, )
    if np.all(lassoClass.select(X_b, Y_b) == lassoClass.sign):
        W_train_vae.append(1)
    else:
        W_train_vae.append(1)
    Z_train_vae.append(lassoClass.basis(X_b, Y_b))

Z_train_vae = np.array(Z_train_vae)
W_train_vae = np.array(W_train_vae)
print(np.mean(W_train), np.mean(W_train_vae))
Z_train_tot = np.concatenate([Z_train, Z_train_vae])
W_train_tot = np.concatenate([W_train, W_train_vae])

batch_size = 128
net = learn_select_prob(Z_train, W_train, num_epochs=1000, batch_size=batch_size, verbose=True, print_every=100)
pr_data = net(torch.tensor(Z_data, dtype=torch.float))
print(pr_data)

theta_data = lassoClass.test_statistic(X, Y)
N_0 = Z_data - gamma @ theta_data
target_var = np.diag(lassoClass.Sigma1)
target_sd = np.sqrt(target_var)
gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 101)
interval_nn = np.zeros([num_select, 2])
weight_val = np.zeros([num_select, 101])
covered_nn = []
for k in range(num_select):
    target_theta_k = theta_data[k] + gamma_list[:, k]
    target_theta = np.tile(theta_data, [101, 1])
    target_theta[:, k] = target_theta_k
    weight_val[k, :] = get_weight(net, target_theta, N_0, gamma)
    interval = get_CI(target_theta_k, weight_val[k, :], target_var[k], theta_data[k])
    interval_nn[k, :] = interval
    if interval_nn[k, 0] <= beta_E[k] <= interval_nn[k, 1]:
        covered_nn.append(1)
    else:
        covered_nn.append(0)


# use VAE data
net_vae = learn_select_prob(Z_train_tot, W_train_tot, num_epochs=1000, batch_size=batch_size, verbose=True, print_every=100)

pr_data_vae = net_vae(torch.tensor(Z_data, dtype=torch.float))
print(pr_data_vae)
interval_nn_vae = np.zeros([num_select, 2])
weight_val_vae = np.zeros([num_select, 101])
covered_nn_vae = []
for k in range(num_select):
    target_theta_k = theta_data[k] + gamma_list[:, k]
    target_theta = np.tile(theta_data, [101, 1])
    target_theta[:, k] = target_theta_k
    weight_val_vae[k, :] = get_weight(net_vae, target_theta, N_0, gamma)
    interval = get_CI(target_theta_k, weight_val_vae[k, :], target_var[k], theta_data[k])
    interval_nn_vae[k, :] = interval
    if interval_nn_vae[k, 0] <= beta_E[k] <= interval_nn_vae[k, 1]:
        covered_nn_vae.append(1)
    else:
        covered_nn_vae.append(0)

# true interval
D_0 = lassoClass.D_0
prob_gamma = np.zeros([num_select, 101])
interval_true = np.zeros([num_select, 2])
fig, ax = plt.subplots(ncols=num_select, figsize=(4 * num_select, 5))
covered_true = []
for i in range(num_select):
    e_i = np.zeros(num_select)
    e_i[i] = 1
    for jj in range(101):
        if data_type == 'linear':
            prob_gamma[i, jj] = lassoClass.linear_KKT(theta_data + gamma_list[jj, i] * e_i, D_0 * np.sqrt(n))
        else:
            prob_gamma[i, jj] = lassoClass.logistic_KKT(theta_data + gamma_list[jj, i] * e_i, D_0 * np.sqrt(n))
    D_M_gamma = theta_data + np.outer(gamma_list[:, i], e_i)
    interval_true[i, :] = get_CI(D_M_gamma[:, i], prob_gamma[i, :], target_var[i], theta_data[i])
    if interval_true[i, 0] <= beta_E[i] <= interval_true[i, 1]:
        covered_true.append(1)
    else:
        covered_true.append(0)
    # plot
    if num_select == 1:
        plt.plot(D_M_gamma[:, i], weight_val[i, :], label='nn')
        plt.plot(D_M_gamma[:, i], weight_val_vae[i, :], label='nn + vae', ls=':')
        plt.plot(D_M_gamma[:, i], prob_gamma[i, :], label='truth', ls='--')
        plt.legend()
    else:
        ax[i].plot(D_M_gamma[:, i], weight_val[i, :], label='nn')
        ax[i].plot(D_M_gamma[:, i], weight_val_vae[i, :], label='nn + vae', ls=':')
        ax[i].plot(D_M_gamma[:, i], prob_gamma[i, :], label='truth', ls='--')
        ax[i].legend()
plt.savefig("VAE_linear_seed_{}.png".format(seed))

print(interval_nn[:, 1] - interval_nn[:, 0], '\n', covered_nn)
print(interval_nn_vae[:, 1] - interval_nn_vae[:, 0], '\n', covered_nn_vae)
print(interval_true[:, 1] - interval_true[:, 0], '\n', covered_true)
