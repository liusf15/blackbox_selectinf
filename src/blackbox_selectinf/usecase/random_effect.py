import numpy as np
from scipy.stats import f


class random_effect(object):
    def __init__(self, X, level, basis_type='mean_var'):
        self.X = X
        self.I = X.shape[0]
        self.n = X.shape[1]
        self.N = self.I * self.n
        self.cutoff = f.ppf(level, self.I - 1, self.N - self.I)
        self.basis_type = basis_type

    def basis(self, X_b, basis_type='mean_var'):
        n_b = X_b.shape[1]
        # if basis_type == 'mean_var':
        #     # return np.concatenate([np.mean(X_b, 1), np.var(X_b, ddof=1, axis=1) / n_b])  # new
        #     # multiply the first half by sqrt n, the second half by n
        #     return np.concatenate([np.mean(X_b, 1), np.var(X_b, ddof=1, axis=1) / n_b])

        if basis_type == 'mean_var':
            return np.concatenate([np.mean(X_b, 1), np.mean(X_b**2, 1)])  # old

        if basis_type == 'SS_nor':
            X_bar_i_b = np.mean(X_b, 1)
            X_bar_b = np.mean(X_bar_i_b)
            SSA_b = n_b * np.sum((X_bar_i_b - X_bar_b) ** 2)
            SSE_b = np.sum((X_b - X_bar_i_b[:, None]) ** 2)
            I = self.I
            return np.stack([SSA_b / n_b / (I - 1), SSE_b / n_b / (n_b - 1) / I])

    def test_statistic(self, X_b):
        """
        :param X_b: dataset
        :return: \hat \sigma_A, F statistics
        """
        n_b = X_b.shape[1]
        N_b = self.I * n_b
        X_bar_i_b = np.mean(X_b, 1)
        X_bar_b = np.mean(X_bar_i_b)
        SSA_b = n_b * np.sum((X_bar_i_b - X_bar_b) ** 2)
        SSE_b = np.sum((X_b - X_bar_i_b[:, None]) ** 2)
        # multiply theta_hat by n
        return (SSA_b / (self.I - 1) - SSE_b / (N_b - self.I)) / n_b, SSA_b / SSE_b * (N_b - self.I) / (self.I - 1)
        # SSA_b = n_b * np.sum((X_bar_i_b - X_bar_b) ** 2)
        # SSE_b = np.sum((X_b - X_bar_i_b[:, None]) ** 2)
        # return (SSA_b / (self.I - 1) - SSE_b / (self.N - self.I)) / self.n, \
        #        SSA_b / SSE_b * (self.N - self.I) / (self.I - 1)

    def gen_train_data(self, ntrain, n_b):
        Z_train = []
        W_train = np.zeros(ntrain)
        theta_hat_train = []
        I = self.I
        n = self.n
        N = n * I
        N_b = n_b * I
        for t in range(ntrain):
            X_b = np.zeros([I, n_b])
            for i in range(I):
                X_b[i, :] = self.X[i, np.random.choice(n, n_b, replace=True)]
            Z_b = self.basis(X_b, self.basis_type)  # new
            # Z_b = self.basis(X_b) * n_b / n  # old
            Z_train.append(Z_b)
            # old
            # SSA_b = n * (np.sum(Z_b[:I] ** 2) - I * np.mean(Z_b[:I]) ** 2)
            # SSE_b = n * np.sum(Z_b[I:]) - n * np.sum(Z_b[:I] ** 2)
            # F_stat_hat = SSA_b / SSE_b * (N - I) / (I - 1)
            # theta_hat_b = (SSA_b / (I - 1) - SSE_b / (N - I)) / n
            # new
            # theta_hat_b, F_stat_hat = self.test_statistic(X_b)
            if self.basis_type == 'mean_var':
                # SSA_b = n_b * (np.sum(Z_b[:I] ** 2) - I * np.mean(Z_b[:I]) ** 2)
                # SSE_b = np.mean(Z_b[I:]) * (I * n_b * (n_b - 1))
                # theta_hat_b = (SSA_b / (I - 1) - SSE_b / (N_b - I)) / n_b
                # F_stat_hat = SSA_b / SSE_b * (N_b - I) / (I - 1)
                tmp1 = np.sum((Z_b[:I] - np.mean(Z_b[:I]))**2) / (I - 1)
                tmp2 = np.mean(Z_b[I:])
                theta_hat_b = tmp1 - tmp2
                F_stat_hat = tmp1 / tmp2
            elif self.basis_type == 'SS_nor':
                theta_hat_b = Z_b[0] - Z_b[1]
                F_stat_hat = Z_b[0] / Z_b[1]
            theta_hat_train.append(theta_hat_b)
            # print(F_stat_hat, self.cutoff)
            if F_stat_hat >= self.cutoff:
                W_train[t] = 1
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train)) / ntrain
        var_theta = np.var(theta_hat_train)
        Gamma = cov_Z_theta / var_theta
        return Z_train, W_train, Gamma, var_theta, theta_hat_train

    def gen_parametric(self, ntrain, n_b, mu=1, sigma=1, sigma_a=0):
        Z_train = []
        W_train = np.zeros(ntrain)
        theta_hat_train = []
        I = self.I
        n = self.n
        N = n * I
        N_b = n_b * I
        for t in range(ntrain):
            a = np.random.randn(I) * sigma_a
            X_b = mu + np.tile(a[:, None], (1, n_b)) + np.random.randn(I, n_b) * sigma
            Z_b = self.basis(X_b, self.basis_type)  # new
            # Z_b = self.basis(X_b) * n_b / n  # old
            Z_train.append(Z_b)  # multiplied by n
            # old
            if self.basis_type == 'mean_var':
                SSA_b = n * (np.sum(Z_b[:I] ** 2) - I * np.mean(Z_b[:I]) ** 2)
                SSE_b = n * np.sum(Z_b[I:]) - n * np.sum(Z_b[:I] ** 2)
                F_stat_hat = SSA_b / SSE_b * (N - I) / (I - 1)
                theta_hat_b = (SSA_b / (I - 1) - SSE_b / (N - I)) / n
            elif self.basis_type == 'SS_nor':
                F_stat_hat = Z_b[0] / Z_b[1]
                theta_hat_b = Z_b[0] - Z_b[1]
            # new
            # theta_hat_b, F_stat_hat = self.test_statistic(X_b)
            # SSA_b = n * (np.sum(Z_b[:I] ** 2) - I * np.mean(Z_b[:I]) ** 2)
            # SSE_b = np.mean(Z_b[I:]) * (I * n_b * (n_b - 1))
            # theta_hat_b = (SSA_b / (I - 1) - SSE_b / (N_b - I)) / n_b
            # F_stat_hat = SSA_b / SSE_b * (N_b - I) / (I - 1)
            theta_hat_train.append(theta_hat_b)  # multiplied by n
            # print(F_stat_hat, self.cutoff)
            if F_stat_hat >= self.cutoff:
                W_train[t] = 1
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train)) / ntrain
        var_theta = np.var(theta_hat_train)
        Gamma = cov_Z_theta / var_theta
        return Z_train, W_train, Gamma, theta_hat_train
