from importlib import reload
import blackbox_selectinf
import blackbox_selectinf.learning.learning
reload(blackbox_selectinf)
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
reload(blackbox_selectinf.usecase.DTL)
from blackbox_selectinf.usecase.DTL import DropTheLoser
import numpy as np
import matplotlib.pyplot as plt


# Stage 1
K = 50
n_k = 1000  # equal sample size
sigma = 1  # equal variance
np.random.seed(1)
mu_list = np.random.rand(K) * 2 - 1  # mu: in the range [-1, 1]
X = np.zeros([K, n_k])
np.random.seed(2)
for k in range(K):
    X[k, :] = np.random.randn(n_k) + mu_list[k]

X_bar = np.mean(X, 1)
win_idx = np.argmax(X_bar)

# Stage 2
m = 500  # sample size in stage 2
np.random.seed(2)
X_win_2 = np.random.randn(m) + mu_list[win_idx]
win_2_bar = np.mean(X_win_2)

DTL_class = DropTheLoser(X, X_win_2)
Z_data = DTL_class.basis(X, X_win_2)
theta_data = DTL_class.theta_hat
n_b = 700
m_b = 300
Z_train, W_train, gamma = DTL_class.gen_train_data(1000, n_b, m_b)

# train
net = learn_select_prob(Z_train, W_train, num_epochs=1000, batch_size=100)

N_0 = Z_data - gamma @ theta_data.reshape(1, )

target_var = 1 / np.sqrt(n_b + m_b)
gamma_list = np.linspace(-20 / np.sqrt(n_b + m_b), 20 / np.sqrt(n_b + m_b), 101)
target_theta = theta_data + gamma_list
target_theta = target_theta.reshape(1, 101)
weight_val = get_weight(net, target_theta, N_0, gamma)
interval = get_CI(target_theta, weight_val, target_var, theta_data)
plt.plot(target_theta[0, :], weight_val)
