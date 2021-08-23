import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    loglik = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loglik + KLD


def train_vae(X_train, model, opt, batch_size):
    model.train()
    losses = []
    ntrain = X_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        opt.zero_grad()
        recon_batch, mu, logvar = model(x_batch)
        loss = vae_loss(recon_batch, x_batch, mu, logvar)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


class Decoder_logistic(nn.Module):
    def __init__(self, input_dim, n, p):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 1000)
        self.relu3 = nn.ReLU()
        self.out_x = nn.Linear(1000, n * p)
        self.out_y = nn.Linear(1000, n)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        x = self.out_x(h3)
        y = self.out_y(h3)
        y = self.out_act(y)
        return torch.cat((x, y), 1)


class Decoder_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 1000)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(1000, output_dim)

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        a4 = self.out(h3)
        y = a4
        return y


def decoder_loss(x_batch, X_recon, n, p):
    X_b = X_recon[:, :n * p].reshape(-1, n, p)
    Y_b = X_recon[:, n * p:].reshape(-1, n)
    Z1 = torch.mean(X_b, 1)
    Z2 = torch.einsum('ijk, ij -> ik', X_b, Y_b) / len(Y_b[0])
    Z_b = torch.cat([Z2, Z1], 1)
    return torch.sum((x_batch - Z_b)**2) / x_batch.shape[0]


def train_decoder(Z_train, decoder, opt, batch_size, n, p):
    decoder.train()
    losses = []
    ntrain = Z_train.size(0)
    for beg_i in range(0, ntrain, batch_size):
        x_batch = Z_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        opt.zero_grad()
        X_recon = decoder(x_batch)
        loss = decoder_loss(x_batch, X_recon, n, p)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    return losses[-1]


def train_networks(n, p, bottleneck_dim, Z_pos, vae_path, decoder_path, output_dim, data_type='linear', print_every=100, vae_epochs=1000, dec_epochs=10):
    input_dim = Z_pos.shape[1]
    vae_model = VAE(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    vae_opt = optim.Adam(vae_model.parameters(), lr=1e-3)
    batch_size = 100
    vae_losses = []
    print("Start training VAE")
    for epoch in range(vae_epochs):
        ll = train_vae(Z_pos, vae_model, vae_opt, batch_size)
        vae_losses.append(ll)
        if epoch % print_every == 0:
            print("VAE epoch: ", epoch, "loss: ", ll)
            torch.save(vae_model.state_dict(), vae_path)
    print("Saved VAE model to ", vae_path)
    torch.save(vae_model.state_dict(), vae_path)

    decoder_losses = []
    if data_type == 'linear':
        decoder = Decoder_linear(input_dim, output_dim)
    elif data_type == 'binary':
        decoder = Decoder_logistic(input_dim, n, p)
    decoder_opt = optim.Adam(decoder.parameters(), lr=1e-3)
    Z_gen = vae_model.decode(torch.randn([1000, bottleneck_dim]))
    print("Start training decoder")
    for epoch in range(dec_epochs):
        ll = train_decoder(Z_gen, decoder, decoder_opt, batch_size=100, n=n, p=p)
        decoder_losses.append(ll)
        if epoch % 1 == 0:
            print("Decoder epoch: ", epoch, "loss: ", ll)
            torch.save(decoder.state_dict(), decoder_path)
    print("Saved decoder to ", decoder_path)



