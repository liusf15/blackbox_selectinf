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
input_dim = K * n + m
np.random.seed(13)
mu_list = np.random.rand(K) * 2 - 1  # mu: in the range [-1, 1]
mu_argsort = np.argsort(mu_list)
mu_list[mu_argsort[-2]] = mu_list[mu_argsort[-1]]
mu_list[mu_argsort[-3]] = mu_list[mu_argsort[-1]]
seed = 8
np.random.seed(seed)
X = np.zeros([K, n])
for k in range(K):
    X[k, :] = np.random.randn(n) + mu_list[k]
X_bar = np.mean(X, 1)
max_rest = np.sort(X_bar)[-2]
win_idx = np.argmax(X_bar)
X_2 = np.random.randn(m) + mu_list[win_idx]


def get_flat(X, X_2):
    X_flat = X.reshape(-1)
    X_flat = np.concatenate([X_flat, X_2.reshape(-1)])
    X_flat = torch.tensor(X_flat, dtype=torch.float)
    return X_flat


X_flat = get_flat(X, X_2)


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 55)
        self.fc22 = nn.Linear(400, 55)
        self.fc3 = nn.Linear(55, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

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


# generate training data by bootstrap
ntrain = 2**14
n_b = n
m_b = m
X_train_tmp = []
X_pooled = np.concatenate([X[win_idx], X_2])
count = 0
for i in range(100000):
    X_b = np.zeros([K, n_b])
    for k in range(K):
        if k != win_idx:
            X_b[k, :] = X[k, np.random.choice(n, n_b, replace=True)]
        if k == win_idx:
            X_b[k, :] = X_pooled[np.random.choice(n + m, n_b, replace=True)]
    X_2_b = X_pooled[np.random.choice(n + m, m_b, replace=True)]
    idx = np.argmax(np.mean(X_b, 1))
    # X_train_tmp.append(get_flat(X_b, X_2_b))
    if idx == win_idx:
        count = count + 1
        X_train_tmp.append(get_flat(X_b, X_2_b))
        if count % 100 == 0:
            print(count)
    if count == ntrain:
        break
print("ratio of selection", count / i)  # 0.6856664574178698

X_train = torch.cat(X_train_tmp).reshape(ntrain, input_dim)  # ntrain * input_dim
torch.mean(X_train[:, :n * K].reshape(ntrain, K, n), [0, 2])
torch.tensor(X_bar, dtype=torch.float)


losses = []
num_epochs = 20
for epoch in range(num_epochs):
    ll = train(X_train, model, opt, batch_size)
    losses.append(ll)
    if epoch % 10 == 0:
        print(epoch, ll)
        z = torch.randn(200, 55)
        X_gen = model.decode(z)
        X_gen_1 = X_gen[:, :n * K].reshape(200, K, n)
        print(torch.argmax(torch.mean(X_gen_1, 2), 1))
        print(max(abs(torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar)))
        torch.save(model.state_dict(), "VAE_model_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))

plt.plot(np.array(losses) / ntrain)
plt.yscale('log')

torch.save(model.state_dict(), "VAE_model_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m))

# sample
model.load_state_dict(torch.load("VAE_model_n_{}_K_{}_m_{}.pt".format(n, K, m)))
z = torch.randn(200, 55)
X_gen = model.decode(z)
X_gen_1 = X_gen[:, :n * K].reshape(200, K, n)
torch.argmax(torch.mean(X_gen_1, 2), 1)
win_idx
torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar
max(abs(torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar))
plt.hist(torch.mean(X_gen_1, [0, 2]).detach().numpy() - X_bar, 50)



#
# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')
