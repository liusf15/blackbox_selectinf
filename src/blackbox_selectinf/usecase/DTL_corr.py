import numpy as np


class DTL_corr(object):
    def __init__(self, X, X_2, selection='mean', uc=2):
        self.X = X
        self.X_2 = X_2
        self.K = X.shape[0]
        self.n = X.shape[1]
        self.m = len(X_2)
        self.X_bar = np.mean(X, 1)
        self.selection = selection
        self.uc = uc
        self.UC = self.X_bar + uc * np.std(X, axis=1, ddof=1)
        if selection == 'mean':
            self.win_idx = np.argmax(self.X_bar)
        else:
            self.win_idx = np.argmax(self.UC)
        stat = self.test_statistic(self.X, self.X_2, return_D0=True)
        self.theta_hat = stat[0]
        self.D_0 = stat[1]

    def basis(self, X_b, X_2_b, basis_type="naive"):
        """
        Compute the basis given any generated data
        :param X_b:
        :param X_2_b:
        :param basis_type:
        :return:
        """
        d_M = (np.sum(X_b[self.win_idx, :]) + np.sum(X_2_b)) / (len(X_b[self.win_idx, :]) + len(X_2_b))
        if basis_type == "complete":
            Z_b = np.concatenate([np.mean(X_b, 1), np.mean(X_b ** 2, 1), [np.mean(X_2_b)], [np.mean(X_2_b ** 2)]])
        elif basis_type == "naive":
            Z_b = np.mean(X_b, 1)
            Z_b[self.win_idx] = d_M
        elif basis_type == "withD0":
            Z_b = np.mean(X_b, 1)
            Z_b = np.concatenate([Z_b, np.array(d_M).reshape(1, )])
        else:
            raise AssertionError("invalid basis_type")
        return Z_b

    def test_statistic(self, X_b, X_2_b, return_D0=False):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        d_M = (np.sum(X_b[self.win_idx, :]) + np.sum(X_2_b)) / (len(X_b[self.win_idx, :]) + len(X_2_b))
        if not return_D0:
            return d_M
        d_0 = np.mean(X_b[self.win_idx, :]) - np.mean(X_2_b)
        return d_M, d_0

    def gen_train_data(self, ntrain, n_b, m_b, basis_type="naive", return_gamma=True, remove_D0=False, blocklength=10):
        """
        bootstrap training data
        :param ntrain: number of training data
        :param return_gamma: whether return gamma = Cov(B, theta_hat) Var(theta_hat)^{-1}
        :return:
        """
        Z_train = []
        W_train = []
        theta_hat_train = []
        D0_train = []
        X_pooled = np.concatenate([self.X[self.win_idx], self.X_2])
        num_blocks_1 = n_b // blocklength
        num_blocks_2 = m_b // blocklength
        for i in range(ntrain):
            X_b = np.zeros([self.K, num_blocks_1 * blocklength])
            for k in range(self.K):
                if k != self.win_idx:
                    indices = np.random.choice(self.n - blocklength, num_blocks_1, replace=True)
                    for s in range(num_blocks_1):
                        idx = indices[s]
                        data_block = self.X[k, idx: idx + blocklength]
                        X_b[k, s * blocklength: (s + 1) * blocklength] = data_block
                if k == self.win_idx:
                    indices = np.random.choice(self.n + self.m - blocklength, num_blocks_1, replace=True)
                    for s in range(num_blocks_1):
                        idx = indices[s]
                        data_block = X_pooled[idx: idx + blocklength]
                        X_b[k, s * blocklength: (s + 1) * blocklength] = data_block
            if self.selection == 'mean':
                idx = np.argmax(np.mean(X_b, 1))
            else:
                idx = np.argmax(np.mean(X_b, 1) + self.uc * np.std(X_b, axis=1, ddof=1))
            if idx == self.win_idx:
                W_train.append(1)
            else:
                W_train.append(0)
            indices = np.random.choice(self.n + self.m - blocklength, num_blocks_2, replace=True)
            X_2_b = []
            for s in range(num_blocks_2):
                idx = indices[s]
                X_2_b.extend(X_pooled[idx: idx + blocklength])
            X_2_b = np.array(X_2_b)
            Z_train.append(self.basis(X_b, X_2_b, basis_type))
            stat = self.test_statistic(X_b, X_2_b, return_D0=True)
            theta_hat_train.append(stat[0])
            D0_train.append(stat[1])

        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        result = {'Z_train': Z_train, 'W_train': W_train}
        theta_hat_train = np.array(theta_hat_train)
        D0_train = np.array(D0_train).reshape(len(D0_train), 1)
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result['gamma'] = gamma
        if remove_D0:
            D0_train = np.array(D0_train)
            cov_Z_D0 = (Z_train - np.mean(Z_train, 0)).T @ (D0_train - np.mean(D0_train, 0)) / ntrain
            var_D0 = np.cov(D0_train.T)
            if var_D0.size == 1:
                var_D0 = var_D0.reshape(1, 1)
                cov_Z_D0 = cov_Z_D0.reshape(len(cov_Z_D0), 1)
            gamma_D0 = cov_Z_D0 @ np.linalg.inv(var_D0)
            Z_train_indep = Z_train - D0_train @ gamma_D0.T
            result['Z_train'] = Z_train_indep
            result['gamma_D0'] = gamma_D0
            result['D0_train'] = D0_train
        return result



