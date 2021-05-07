import sys
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression


class LassoClass(object):
    def __init__(self, X, Y, lbd, data_type):
        self.X = X
        self.Y = Y
        self.lbd = lbd
        self.n = X.shape[0]
        self.p = X.shape[1]
        if data_type == "linear":
            self.alpha = lbd / self.n
        elif data_type == "binary":
            self.alpha = 1 / self.lbd
        else:
            raise AssertionError("invalid data_type")
        if data_type == "linear":
            self.lasso_l1 = Lasso(alpha=self.alpha)
        else:
            self.lasso_l1 = LogisticRegression(penalty='l1', C=self.alpha, solver='liblinear', fit_intercept=False, tol=1e-7)
        self.lasso_l1.fit(X, Y)
        self.beta_hat = self.lasso_l1.coef_.reshape(self.p)
        self.sign = np.sign(self.beta_hat)
        self.E = self.beta_hat != 0
        self.X_E = self.X[:, self.E]
        self.num_select = np.sum(self.E)
        if self.num_select == 0:
            print("select none")
            sys.exit(1)
        if data_type == "linear":
            self.lr = LinearRegression(fit_intercept=False)
        else:
            self.lr = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
        self.lr.fit(self.X_E, self.Y)
        self.beta_ls = self.lr.coef_.squeeze()
        if self.beta_ls.size == 1:
            self.beta_ls = self.beta_ls.reshape(1, )
        if data_type == "linear":
            self.D_0 = self.X[:, ~self.E].T @ (self.Y - self.X_E @ self.beta_ls) / np.sqrt(self.n)
        else:
            pr = 1 / (1 + np.exp(-self.X_E @ self.beta_ls)).reshape(self.n)
            self.D_0 = self.X[:, ~self.E].T @ (self.Y - pr) / np.sqrt(self.n)

        self.Sigma1 = np.linalg.inv(self.X_E.T @ self.X_E)
        self.G = self.X_E.T @ self.X_E
        self.G_inv = np.linalg.inv(self.G)
        self.Sigma2 = self.X[:, ~self.E].T @ self.X[:, ~self.E] - (self.X[:, ~self.E].T @ self.X_E) @ self.G_inv @ (self.X_E.T @ self.X[:, ~self.E])
        self.Sigma2 = self.Sigma2 / self.n

    def select(self, X_b, Y_b,):
        self.lasso_l1.fit(X_b, Y_b)
        beta_tmp = self.lasso_l1.coef_.reshape(self.p)
        return np.sign(beta_tmp)

    def basis(self, X_b, Y_b, basis_type="naive"):
        """
        Compute the basis given any generated data
        :param X_b:
        :param X_2_b:
        :param basis_type:
        :return:
        """
        if basis_type == "naive":
            XY = X_b.T @ Y_b / len(Y_b)
            XX = np.mean(X_b, 0)
            return np.concatenate([XY, XX])
        elif basis_type == "indep":
            pass  # TODO
        else:
            raise AssertionError("invalid basis_type")

    def remove_D0(self, Z_train, D0_train, gamma_D0):
        return Z_train - D0_train @ gamma_D0.T

    def test_statistic(self, X_b, Y_b, return_D0=False):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_b[:, self.E], Y_b)
        beta_ls_b = lr.coef_.squeeze()
        if not return_D0:
            return beta_ls_b
        else:
            D0_b = X_b[:, ~self.E].T @ (Y_b - X_b[:, self.E] @ beta_ls_b) / np.sqrt(len(Y_b))
            return beta_ls_b, D0_b

    def gen_train_data(self, ntrain, n_b, basis_type="naive", return_gamma=True, return_gamma_D0=False):
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
            idx_b = np.random.choice(self.n, n_b, replace=True)
            X_b = self.X[idx_b, :]
            Y_b = self.Y[idx_b]
            Z_train.append(self.basis(X_b, Y_b, basis_type))
            if np.all(self.select(X_b, Y_b) == self.sign):
                W_train.append(1)
            else:
                W_train.append(0)
            if not return_gamma_D0:
                theta_hat_train.append(self.test_statistic(X_b, Y_b))
            else:
                stat = self.test_statistic(X_b, Y_b, return_D0=True)
                theta_hat_train.append(stat[0])
                D0_train.append(stat[1])
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        result = [Z_train, W_train]
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result = [Z_train, W_train, gamma]
        if return_gamma_D0:
            D0_train = np.array(D0_train)
            cov_Z_D0 = (Z_train - np.mean(Z_train, 0)).T @ (D0_train - np.mean(D0_train, 0)) / ntrain
            var_D0 = np.cov(D0_train.T)
            if var_D0.size == 1:
                var_D0 = var_D0.reshape(1, 1)
                cov_Z_D0 = cov_Z_D0.reshape(len(cov_Z_D0), 1)
            gamma_D0 = cov_Z_D0 @ np.linalg.inv(var_D0)
            result = [Z_train, W_train, gamma, gamma_D0]
        return result


class TwoStageLasso(object):
    pass
    # TODO
