import numpy as np


class DropTheLoser3(object):
    def __init__(self, X, n_1, n_2, n_3, win_idx1, win_idx):
        self.X = X
        # self.X_2 = X_2
        # self.X_3 = X_3
        self.K = len(X)
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        # self.X_bar_1 = np.mean(X_1, 1)
        # self.X_bar_2 = np.mean(X_2, 1)
        # self.selection = selection
        # self.uc = uc
        # self.UC = self.X_bar + uc * np.std(X, axis=1, ddof=1)
        # if selection == 'mean':
        self.win_idx1 = win_idx1 #np.argsort(self.X_bar_1)[::-1][-10:]
        self.win_idx_in_2 = list(self.win_idx1).index(win_idx)
        self.win_idx = win_idx  #self.win_idx1[self.win_idx_in_2]
        # else:
        #     self.win_idx = np.argmax(self.UC)
        self.theta_hat = X[win_idx].mean()

    def basis(self, X_b):
        """
        Compute the basis given any generated data
        :param X_b:
        :param X_2_b:
        :param basis_type:
        :return:
        """
        Z_b = np.zeros(self.K)
        for k in range(self.K):
            Z_b[k] = np.mean(X_b[k])
        # if basis_type == "complete":
        #     Z_b = np.concatenate([np.mean(X_b, 1), np.mean(X_b ** 2, 1), [np.mean(X_2_b)], [np.mean(X_2_b ** 2)]])
        # elif basis_type == "naive":
        #     Z_b = np.mean(X_b, 1)
        #     Z_b[self.win_idx] = d_M
        # elif basis_type == "withD0":
        #     Z_b = np.mean(X_b, 1)
        #     Z_b = np.concatenate([Z_b, np.array(d_M).reshape(1, )])
        # else:
        #     raise AssertionError("invalid basis_type")
        return Z_b

    def test_statistic(self, X_win_tot):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        # tmp = np.hstack([X_1_b[self.win_idx], X_2_b[self.win_idx_in_2], X_3_b])
        d_M = np.mean(X_win_tot)
        # d_M = (np.sum(X_1_b[self.win_idx, :]) + np.sum(X_2_b) + np.sum(X_3_b)) / (len(X_b[self.win_idx, :]) + len(X_2_b))
        # if not return_D0:
        #     return d_M
        # d_0 = np.mean(X_b[self.win_idx, :]) - np.mean(X_2_b)
        return d_M

    def gen_train_data(self, ntrain, return_gamma=True):
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
        for i in range(ntrain):
            X_b_1 = np.zeros([self.K, self.n_1])
            for k in range(self.K):
                tmp = np.random.choice(self.X[k], size=self.n_1, replace=True)
                X_b_1[k, :] = tmp
            idx_1 = np.argsort(np.mean(X_b_1, 1))[::-1][-10:]
            # Stage 2
            X_b_2 = np.zeros([10, self.n_2])
            s = 0
            for k in self.win_idx1:
                X_b_2[s, :] = np.random.choice(self.X[k], size=self.n_2, replace=True)
                s += 1
            idx = self.win_idx1[np.argmax(np.mean(X_b_2, 1))]
            if idx == self.win_idx:
                W_train.append(1)
            else:
                W_train.append(0)
            # Stage 3
            X_b_3 = np.random.choice(self.X[idx], size=self.n_3, replace=True)

            X_b = []
            for k in range(self.K):
                if k not in self.win_idx1:
                    X_b.append(X_b_1[k, :])
                elif k == self.win_idx:
                    X_b.append(np.hstack([X_b_1[self.win_idx], X_b_2[self.win_idx_in_2], X_b_3]))
                else:
                    X_b.append(np.hstack([X_b_1[k, :], X_b_2[list(self.win_idx1).index(k), :]]))
            Z_train.append(self.basis(X_b))
            theta_hat_train.append(self.test_statistic(X_b[self.win_idx]))

        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        result = {'Z_train': Z_train, 'W_train': W_train}
        theta_hat_train = np.array(theta_hat_train)
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result['gamma'] = gamma
        return result


