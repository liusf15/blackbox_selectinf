import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


class BH(object):
    def __init__(self, X, alpha=.05, sigma=1, one_side=False):
        self.X = X
        self.alpha = alpha
        self.sigma = sigma
        self.I = X.shape[0]
        self.n = X.shape[1]
        self.one_side = one_side

    def basis(self, X_b):
        return np.mean(X_b, 1)

    def test_statistic(self, X_b):
        return np.mean(X_b, 1)

    def gen_train_data(self, ntrain, n_b):
        I = self.I
        X = self.X
        n = self.n
        Z_train = []
        W_train = np.zeros([ntrain, I])
        for r in range(ntrain):
            X_b = np.zeros([I, n_b])
            for i in range(I):
                X_b[i, :] = np.random.choice(X[i, :], size=n_b, replace=True)
            X_bar_b = np.mean(X_b, 1)
            Z_train.append(X_bar_b)
            if self.one_side:
                pvals_b = norm.cdf(X_bar_b / self.sigma * np.sqrt(n))
            else:
                pvals_b = 2 * (1 - norm.cdf(abs(X_bar_b) / self.sigma * np.sqrt(n)))
            bh = multipletests(pvals_b, self.alpha, method='fdr_bh')
            W_train[r, :] = bh[0] * 1
        Z_train = np.array(Z_train)
        return Z_train, W_train


class group_BH(object):
    def __init__(self, X, alpha=.05, sigma=1, one_side=False):
        self.X = X
        self.alpha = alpha
        self.sigma = sigma
        self.m = X.shape[0]
        self.I = X.shape[1]
        self.n = X.shape[2]
        self.one_side = one_side

    def basis(self, X_b):
        X_bar_b = np.mean(X_b, 2)
        return X_bar_b.reshape(-1)

    def test_statistic(self, X_b):
        X_bar_b = np.mean(X_b, 2)
        return X_bar_b.reshape(-1)

    def gen_train_data(self, ntrain, n_b, selected_family, mean=False, fix_level=False):
        I = self.I
        X = self.X
        n = self.n
        m = self.m
        Z_train = []
        W_train = np.zeros([ntrain, len(selected_family), I])
        for r in range(ntrain):
            X_b = np.zeros([m, I, n_b])
            for t in range(m):
                for i in range(I):
                    X_b[t, i, :] = np.random.choice(X[t, i, :], size=n_b, replace=True)
            X_bar_b = np.mean(X_b, 2)
            Z_train.append(X_bar_b.reshape(-1))
            if self.one_side:
                pvals_b = norm.cdf(X_bar_b / self.sigma * np.sqrt(n))
            else:
                pvals_b = 2 * (1 - norm.cdf(abs(X_bar_b) / self.sigma * np.sqrt(n)))
            if mean:
                pvals_b_group_min = np.mean(pvals_b, 1)
                selected_b = np.where(pvals_b_group_min <= 0.5)[0]
            else:
                pvals_b_group_min = np.min(pvals_b, 1)
                selected_b = np.where(pvals_b_group_min <= 0.05)[0]
            s = 0
            if fix_level:
                alpha_corrected = self.alpha * len(selected_family) / m
            else:
                alpha_corrected = self.alpha * len(selected_b) / m
            for t in selected_family:
                if t in selected_b:
                    bh = multipletests(pvals_b[t, :], alpha_corrected, method='fdr_bh')
                    W_train[r, s, :] = bh[0] * 1
                s += 1
        Z_train = np.array(Z_train)
        return Z_train, W_train
