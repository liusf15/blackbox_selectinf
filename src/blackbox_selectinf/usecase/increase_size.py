import numpy as np
from scipy.stats import t


class two_sample(object):
    def __init__(self, X_1, X_2, k, mu_1, mu_2, sigma, n_1, n_2, m_1, m_2, alpha_0):
        self.X_1 = X_1
        self.X_2 = X_2
        self.k = k
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.sigma = sigma
        self.n_1 = n_1
        self.n_2 = n_2
        self.m_1 = m_1
        self.m_2 = m_2
        self.N_1 = n_1 + k * m_1
        self.N_2 = n_2 + k * m_2
        self.alpha_0 = alpha_0

    def basis(self, X_1b, X_2b):
        n_1 = self.n_1
        n_2 = self.n_2
        m_1 = self.m_1
        m_2 = self.m_2
        N_1 = self.N_1
        N_2 = self.N_2
        assert len(X_1b) == N_1, "invalid size of X_1b"
        assert len(X_2b) == N_2, "invalid size of X_1b"
        mean_1 = np.concatenate(
            [np.mean(X_1b[:n_1]).reshape(1, ), np.array([np.mean(X_1b[: (i + m_1)]) for i in range(n_1, N_1, m_1)])])
        mean_2 = np.concatenate(
            [np.mean(X_2b[:n_2]).reshape(1, ), np.array([np.mean(X_2b[: (i + m_2)]) for i in range(n_2, N_2, m_2)])])
        sd_1 = np.concatenate([np.std(X_1b[:n_1], ddof=1).reshape(1, ),
                               np.array([np.std(X_1b[: (i + m_1)], ddof=1) for i in range(n_1, N_1, m_1)])])
        sd_2 = np.concatenate([np.std(X_2b[:n_2], ddof=1).reshape(1, ),
                               np.array([np.std(X_2b[: (i + m_2)], ddof=1) for i in range(n_2, N_2, m_2)])])

        Z = np.concatenate([mean_1, mean_2, sd_1, sd_2])
        return Z

    def test_statistic(self, X_1b, X_2b):
        N_1 = len(X_1b)
        N_2 = len(X_2b)
        s_all = np.sqrt(
            ((N_1 - 1) * np.std(X_1b, ddof=1) + (N_2 - 1) * np.std(X_2b, ddof=1)) / (N_1 + N_2 - 2)) * np.sqrt(
            1 / N_1 + 1 / N_2)
        t_stat = (np.mean(X_1b) - np.mean(X_2b)) / s_all
        return t_stat

    def gen_train_data(self, ntrain, cond_exact=True):
        Z_train = []
        W_train = np.zeros(ntrain)
        theta_hat_train = []
        k = self.k
        n_1 = self.n_1
        n_2 = self.n_2
        m_1 = self.m_1
        m_2 = self.m_2
        N_1 = self.N_1
        N_2 = self.N_2
        X_1 = self.X_1
        X_2 = self.X_2
        for i in range(ntrain):
            history = np.zeros(k + 1)
            X_1b = self.X_1 * 0
            X_2b = self.X_2 * 0
            X_1b[:n_1] = X_1[np.random.choice(N_1, n_1, replace=True)]
            X_2b[:n_2] = X_2[np.random.choice(N_2, n_2, replace=True)]
            t_stat = self.test_statistic(X_1b[:n_1], X_2b[:n_2])
            quant = t.ppf(1 - self.alpha_0 / 2, n_1 + n_2 - 2)
            if abs(t_stat) > quant:
                history[0] = 1
            for j in range(k):
                X_1b[n_1 + j * m_1: n_1 + (j + 1) * m_1] = X_1[np.random.choice(N_1, m_1, replace=True)]
                X_2b[n_2 + j * m_2: n_2 + (j + 1) * m_2] = X_2[np.random.choice(N_2, m_2, replace=True)]
                t_stat = self.test_statistic(X_1b[:n_1 + (j + 1) * m_1], X_2b[: n_2 + (j + 1) * m_2])
                quant = t.ppf(1 - self.alpha_0 / 2, n_1 + (j + 1) * m_1 + n_2 + (j + 1) * m_2 - 2)
                if abs(t_stat) > quant:
                    history[j + 1] = 1
            if cond_exact and np.prod(1 - history[:k]) * history[k] == 1:
                W_train[i] = 1
            if not cond_exact and np.any(history == 1):
                W_train[i] = 1
            Z_b = self.basis(X_1b, X_2b)
            Z_train.append(Z_b)
            # theta_hat_train.append(t_stat)
            theta_hat_train.append(np.mean(X_1b) - np.mean(X_2b))
        Z_train = np.array(Z_train)
        theta_hat_train = np.array(theta_hat_train)
        cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train)) / ntrain
        var_theta = np.var(theta_hat_train)
        Gamma = cov_Z_theta / var_theta
        return Z_train, W_train, Gamma
