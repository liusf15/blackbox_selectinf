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
import pickle


n = 1000
n_b = n
p = 4
input_dim = n * (p + 1)
beta = np.array([0, 0, 5, 5]) / np.sqrt(n)
seed = 1
np.random.seed(seed)
X = np.random.randn(n, p)
Y = X @ beta + np.random.randn(n)
lbd = 30
data_type = 'linear'
lassoClass = LassoClass(X, Y, lbd, data_type)
print(lassoClass.sign)


def get_flat(X, Y):
    X_flat = X.reshape(-1)
    X_flat = np.concatenate([X_flat, Y.reshape(-1)])
    X_flat = torch.tensor(X_flat, dtype=torch.float)
    return X_flat


X_flat = get_flat(X, Y)
bottleneck_dim = 60


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 200)
        self.fc31 = nn.Linear(200, bottleneck_dim)
        self.fc22 = nn.Linear(400, 200)
        self.fc32 = nn.Linear(200, bottleneck_dim)
        self.fc4 = nn.Linear(bottleneck_dim, 200)
        self.fc5 = nn.Linear(200, 400)
        self.fc6 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h21 = F.relu(self.fc21(h1))
        h22 = F.relu(self.fc22(h1))
        return self.fc31(h21), self.fc32(h22)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)
        # return torch.sigmoid(self.fc6(h5))

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


ntrain = 1000
n_b = n
X_train_tmp = []
count = 0
for i in range(1000000):
    idx_b = np.random.choice(n, n_b, replace=True)
    X_b = X[idx_b, :]
    Y_b = Y[idx_b]
    if np.all(lassoClass.select(X_b, Y_b) == lassoClass.sign):
        count += 1
        X_train_tmp.append(get_flat(X_b, Y_b))
        if count % 100 == 0:
            print(count)
    if count == ntrain:
        break
print("ratio of selection", count / i)

X_train = torch.cat(X_train_tmp).reshape(ntrain, input_dim)  # ntrain * input_dim
losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    ll = train(X_train, model, opt, batch_size)
    losses.append(ll)
    if epoch % 10 == 0:
        print(epoch, ll)
        z = torch.randn(bottleneck_dim)
        X_gen = model.decode(z).detach().numpy()
        X_gen_X = X_gen[:n * p].reshape(n, p)
        X_gen_Y = X_gen[n * p:].reshape(n)
        print(lassoClass.select(X_gen_X, X_gen_Y))

        # torch.save(model.state_dict(), "VAElinear_model_seed_{}_n_{}.pt".format(seed, n))
        # path = open("vae_losses.pickle", 'wb')
        # pickle.dump(losses, path)
        # path.close()

plt.plot(losses)
torch.save(model.state_dict(), "VAElinear_model_seed_{}_n_{}.pt".format(seed, n))
# plt.plot(np.array(losses))
# plt.yscale('log')
#
#
# sample
count = 0
# model.load_state_dict(torch.load("VAElinear_model_beta_5_seed_1_n_1000.pt"))
for i in range(100):
    z = torch.randn(1, bottleneck_dim)
    X_gen = model.decode(z)
    X_gen_X = X_gen[:, :n * p].reshape(n, p)
    X_gen_Y = X_gen[:, n * p:].reshape(n)
    X_new = X_gen_X.detach().numpy()
    Y_new = X_gen_Y.detach().numpy()
    sign_new = lassoClass.select(X_new, Y_new)
    print(sign_new)
    if np.all(sign_new == lassoClass.sign):
        count += 1
        print(count)

print(count / 100)
