import numpy as np
from scipy.stats import f
from statsmodels.stats.stattools import durbin_watson


class AR_model(object):
    def __init__(self, X, Y, Q_L=1.6, Q_U=2.3, upper=True, basis_type='residual'):
        self.Y = Y
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.Q_L = Q_L
        self.Q_U = Q_U
        self.upper = upper
        self.hat = X @ np.linalg.inv(X.T @ X) @ X.T
        self.basis_type = basis_type

    def cov_mat(self, rho, n=None, sigma=1):
        if n is None:
            n = self.n
        C = np.tile(np.arange(1, self.n + 1), (n, 1))
        C_cov = np.power(rho, abs(C - C.T)) / (1 - rho ** 2) * sigma**2
        return C_cov

    def basis(self, res):
        tmp = np.cov(res[:-1], res[1:])
        Z_b = np.array([np.mean(res[:-1]), np.mean(res[1:]), tmp[0, 0], tmp[1, 1], tmp[0, 1]])
        return Z_b

    def basis_linear(self, X_b, Y_b):
        XY = X_b.T @ Y_b / len(Y_b)
        XX = np.mean(X_b, 0)
        return np.concatenate([XY, XX])
    
    def test_statistic(self, resids):
        rho_hat = (np.mean(resids[1:] * resids[:-1]) - np.mean(resids[1:]) * np.mean(resids[:-1])) / \
                  (np.mean(resids[:-1] ** 2) - np.mean(resids[:-1]) ** 2)
        return rho_hat, durbin_watson(resids)
    
    def gen_train_data(self, ntrain, n_b, beta_hat, rho_hat):
        C_cov = self.cov_mat(rho_hat, n_b)
        C_inv = np.linalg.inv(C_cov)
        Z_train = []
        W_train = np.zeros(ntrain)
        theta_hat_train = []
        X = np.copy(self.X)
        for i in range(ntrain):
            e_b = np.random.multivariate_normal(np.zeros(n_b), C_cov)
            Y_b = X @ beta_hat + e_b
            res = Y_b - self.hat @ Y_b
            if self.basis_type == 'residual':
                Z_b = self.basis(res)  # residual
                theta_hat, dw = self.test_statistic(res)
            else:
                Z_b = self.basis_linear(X, Y_b)
                theta_hat = np.linalg.inv(X.T @ C_inv @ X) @ X.T @ C_inv @ Y_b
                dw = durbin_watson(res)
            Z_train.append(Z_b)

            if self.upper and dw >= self.Q_U:
                W_train[i] = 1
            elif not self.upper and dw <= self.Q_L:
                W_train[i] = 1
            theta_hat_train.append(theta_hat)

        Z_train = np.array(Z_train)
        theta_hat_train = np.array(theta_hat_train)
        cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train)) / ntrain
        if self.basis_type == 'residual':
            var_theta = np.var(theta_hat_train)
            Gamma = cov_Z_theta / var_theta
        else:
            var_theta = np.cov(theta_hat_train.T)
            print(cov_Z_theta.shape, var_theta.shape)
            Gamma = cov_Z_theta @ np.linalg.inv(var_theta)
        return Z_train, W_train, Gamma, var_theta
