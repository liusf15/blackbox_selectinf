import numpy as np


class DropTheLoser(object):
    def __init__(self, X, X_2):
        self.X = X
        self.X_2 = X_2
        self.K = X.shape[0]
        self.n = X.shape[1]
        self.m = len(X_2)
        self.X_bar = np.mean(X, 1)
        self.win_idx = np.argmax(self.X_bar)
        self.theta_hat = self.test_statistic(self.X, self.X_2)

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
            Z_b = np.concatenate([np.mean(X_b, 1), np.mean(X_b ** 2, 1), [d_M]])
        elif basis_type == "naive":
            Z_b = np.mean(X_b, 1)
            Z_b[self.win_idx] = d_M
        elif basis_type == "withD0":
            Z_b = np.mean(X_b, 1)
            Z_b = np.concatenate([Z_b, d_M])
        else:
            raise AssertionError("invalid basis_type")
        return Z_b

    def test_statistic(self, X_b, X_2_b):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        d_M = (np.sum(X_b[self.win_idx, :]) + np.sum(X_2_b)) / (len(X_b[self.win_idx, :]) + len(X_2_b))
        return d_M

    def D_0(self, X_b, X_2_b):
        """
        D_0 is assumed to have mean zero
        :param X_b:
        :param X_2_b:
        :return:
        """
        pass

    def gen_train_data(self, ntrain, n_b, m_b, basis_type="naive", return_gamma=True):
        """
        bootstrap training data
        :param ntrain: number of training data
        :param return_gamma: whether return gamma = Cov(B, theta_hat) Var(theta_hat)^{-1}
        :return:
        """
        Z_train = []
        W_train = []
        theta_hat_train = []
        X_pooled = np.concatenate([self.X[self.win_idx], self.X_2])
        for i in range(ntrain):
            X_b = np.zeros([self.K, n_b])
            for k in range(self.K):
                if k != self.win_idx:
                    X_b[k, :] = self.X[k, np.random.choice(self.n, n_b, replace=True)]
                if k == self.win_idx:
                    X_b[k, :] = X_pooled[np.random.choice(self.n + self.m, n_b, replace=True)]
            idx = np.argmax(np.mean(X_b, 1))
            if idx == self.win_idx:
                W_train.append(1)
            else:
                W_train.append(0)
            X_2_b = X_pooled[np.random.choice(self.n + self.m, m_b, replace=True)]
            Z_train.append(self.basis(X_b, X_2_b, basis_type))
            theta_hat_train.append(self.test_statistic(X_b, X_2_b))

        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            return Z_train, W_train, gamma
        return Z_train, W_train



