import sys
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso


class LassoClass(object):
    def __init__(self, X, Y, lbd, data_type, basis_type="simple"):
        self.X = X
        self.Y = Y
        self.lbd = lbd
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.data_type = data_type
        self.basis_type = basis_type
        if data_type == "linear":
            self.alpha = lbd / self.n
        elif data_type == "binary":
            self.alpha = 1 / self.lbd
        else:
            raise AssertionError("invalid data_type")
        if data_type == "linear":
            self.lasso_l1 = Lasso(alpha=self.alpha, fit_intercept=False)
        else:
            self.lasso_l1 = LogisticRegression(penalty='l1', C=self.alpha, solver='liblinear', fit_intercept=False, tol=1e-7)
        self.lasso_l1.fit(X, Y)
        self.beta_hat = self.lasso_l1.coef_.reshape(self.p)
        self.sign = np.sign(self.beta_hat)
        self.E = self.beta_hat != 0
        self.s_E = self.sign[self.E]
        self.X_E = self.X[:, self.E]
        self.num_select = int(np.sum(self.E))
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
            self.W = np.identity(self.n)
        else:
            pr = 1 / (1 + np.exp(-self.X_E @ self.beta_ls)).reshape(self.n)
            self.D_0 = self.X[:, ~self.E].T @ (self.Y - pr) / np.sqrt(self.n)
            self.W = np.diag(pr * (1 - pr))

        self.Sigma1 = np.linalg.inv(self.X_E.T @ self.W @ self.X_E)
        self.G = self.X_E.T @ self.X_E
        self.G_inv = np.linalg.inv(self.G)
        self.Sigma2 = self.X[:, ~self.E].T @ self.W @ self.X[:, ~self.E] - (self.X[:, ~self.E].T @ self.W @ self.X_E) @ np.linalg.inv(self.X_E.T @ self.W @ self.X_E) @ (self.X_E.T @ self.W @ self.X[:, ~self.E])
        self.Sigma2 = self.Sigma2 / self.n

    def interval_lee(self):
        # interval by Lee et al
        if self.data_type == 'linear':
            g = glm.gaussian(self.X, self.Y)
        else:
            g = glm.logistic(self.X, self.Y)
        model = lasso.lasso(g, self.lbd)
        model.fit()
        summ = model.summary(compute_intervals=True)
        interval_lee = np.zeros([self.num_select, 2])
        interval_lee[:, 0] = summ.lower_confidence
        interval_lee[:, 1] = summ.upper_confidence
        return interval_lee

    def select(self, X_b, Y_b,):
        self.lasso_l1.fit(X_b, Y_b)
        beta_tmp = self.lasso_l1.coef_.reshape(self.p)
        return np.sign(beta_tmp)

    def basis(self, X_b, Y_b, basis_type="simple"):
        """
        Compute the basis given any generated data
        :param X_b:
        :param X_2_b:
        :param basis_type:
        :return:
        """
        XY = X_b.T @ Y_b / len(Y_b)
        XX = np.mean(X_b, 0)
        if basis_type == "simple":
            return np.concatenate([XY, XX])
        elif basis_type == "quad":
            tmp = (X_b.T @ X_b).reshape(-1)
            return np.concatenate([XY, XX, tmp])
        elif basis_type == "original":
            return np.concatenate([X_b.reshape(-1), Y_b])
        else:
            raise AssertionError("invalid basis type")

    def linear_KKT(self, D_M, D_0):
        tmp = D_M - self.G_inv @ self.s_E * self.lbd
        t1 = np.all(np.sign(tmp) == self.s_E)
        t2 = np.max(abs(self.X[:, ~self.E].T @ self.X_E @ self.G_inv @ self.s_E + 1 / self.lbd * D_0)) <= 1
        return t1 & t2

    def logistic_KKT(self, D_M, D_0):
        s_E = self.sign[self.E]
        tmp = 1 / (1 + np.exp(-self.X_E @ D_M)) - self.lbd * self.X_E @ self.G_inv @ s_E
        eta_hat = np.log(tmp / (1 - tmp))
        beta_hat = self.G_inv @ self.X_E.T @ eta_hat
        t1 = np.all(np.sign(beta_hat) == s_E)
        if D_0.size == 0:
            t2 = True
        else:
            t2 = np.max(abs(self.X[:, ~self.E].T @ self.X_E @ self.G_inv @ s_E + 1 / self.lbd * D_0)) <= 1
        return t1 & t2

    def test_statistic(self, X_b, Y_b, return_D0=False):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        n_b = len(Y_b)
        self.lr.fit(X_b[:, self.E], Y_b)
        beta_ls_b = self.lr.coef_.squeeze()
        if self.num_select == 1:
            beta_ls_b = beta_ls_b.reshape(1, )
        if not return_D0:
            return beta_ls_b
        else:
            if self.data_type == 'linear':
                D0_b = X_b[:, ~self.E].T @ (Y_b - X_b[:, self.E] @ beta_ls_b) / np.sqrt(len(Y_b))
            else:
                pr = 1 / (1 + np.exp(-X_b[:, self.E] @ beta_ls_b)).reshape(n_b)
                D0_b = X_b[:, ~self.E].T @ (Y_b - pr) / np.sqrt(n_b)
            return beta_ls_b, D0_b

    def gen_train_data(self, ntrain, n_b, return_gamma=True, remove_D0=False, target_pos=-1, perturb=False):
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
        count_pos = 0
        for i in range(ntrain):
            if not perturb:
                idx_b = np.random.choice(self.n, n_b, replace=True)
                X_b = self.X[idx_b, :]
                Y_b = self.Y[idx_b]
            else:
                X_b = self.X + np.random.randn(self.n, self.p) * .5
                Y_b = np.copy(self.Y)
            if np.sum(Y_b) == 0 or np.sum(1 - Y_b) == 0:
                continue
            Z_train.append(self.basis(X_b, Y_b, basis_type=self.basis_type))

            if np.all(abs(self.select(X_b, Y_b)) == abs(self.sign)):
                W_train.append(1)
                count_pos += 1
            else:
                W_train.append(0)
            if not remove_D0:
                theta_hat_train.append(self.test_statistic(X_b, Y_b))
            else:
                stat = self.test_statistic(X_b, Y_b, return_D0=True)
                theta_hat_train.append(stat[0])
                D0_train.append(stat[1])
            if count_pos == target_pos:
                break
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        result = {"Z_train": Z_train, "W_train": W_train}
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            # var_theta = (theta_hat_train - np.mean(theta_hat_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result["gamma"] = gamma
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
            result["gamma_D0"] = gamma_D0
            result['D0_train'] = D0_train
        return result

    def gen_train_data_parametric(self, ntrain, n_b, return_gamma=True, remove_D0=False, target_pos=-1):
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
        count_pos = 0
        for i in range(ntrain):
            idx_b = np.random.choice(self.n, n_b, replace=True)
            X_b = self.X[idx_b, :]
            if self.data_type == 'linear':
                Y_b = X_b[:, self.E] @ self.beta_ls + np.random.randn(n_b)
            else:
                prob = 1 / (1 + np.exp(-X_b[:, self.E] @ self.beta_ls))
                Y_b = np.random.binomial(1, prob, n_b)
                if np.sum(Y_b) == 0 or np.sum(1 - Y_b) == 0:
                    continue
            Z_train.append(self.basis(X_b, Y_b, basis_type=self.basis_type))

            if np.all(abs(self.select(X_b, Y_b)) == abs(self.sign)):
                W_train.append(1)
                count_pos += 1
            else:
                W_train.append(0)
            if not remove_D0:
                theta_hat_train.append(self.test_statistic(X_b, Y_b))
            else:
                stat = self.test_statistic(X_b, Y_b, return_D0=True)
                theta_hat_train.append(stat[0])
                D0_train.append(stat[1])
            if count_pos == target_pos:
                break
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        result = {"Z_train": Z_train, "W_train": W_train}
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            # var_theta = (theta_hat_train - np.mean(theta_hat_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result["gamma"] = gamma
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
            result["gamma_D0"] = gamma_D0
            result['D0_train'] = D0_train
        return result

    def gen_train_data_select_single(self, ntrain, n_b):
        Z_train = []
        W_train = np.zeros([ntrain, self.p])
        theta_hat_train = np.zeros([ntrain, self.p])
        for i in range(ntrain):
            idx_b = np.random.choice(self.n, n_b, replace=True)
            X_b = self.X[idx_b, :]
            Y_b = self.Y[idx_b]
            Z_train.append(self.basis(X_b, Y_b, basis_type=self.basis_type))
            s_b = self.select(X_b, Y_b)
            W_train[i, s_b != 0] = 1
            theta_hat_train[i, :] = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_b
        Z_train = np.array(Z_train)
        # find covariance between Z and theta_hat
        Gamma = np.zeros([Z_train.shape[1], self.p])
        for i in range(self.p):
            C_12 = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train[:, i] - np.mean(theta_hat_train[:, i])) / ntrain
            C_22 = np.cov(theta_hat_train[i, :])
            C_12 = C_12.reshape(-1, 1)
            C_22 = C_22.reshape(1, 1)
            Gamma[:, i] = C_12 @ np.linalg.inv(C_22).reshape(1, )
        return Z_train, W_train, Gamma


class TwoStageLasso(object):
    def __init__(self, X_1, Y_1, X_2, Y_2, lbd, data_type):
        self.X_1 = X_1
        self.X_2 = X_2
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.n = X_1.shape[0]
        self.p = X_1.shape[1]
        self.m = X_2.shape[0]
        self.lbd = lbd
        self.data_type = data_type
        if data_type == "linear":
            self.alpha = lbd / self.n
        elif data_type == "binary":
            self.alpha = 1 / self.lbd
        else:
            raise AssertionError("invalid data_type")
        if data_type == "linear":
            self.lasso_l1 = Lasso(alpha=self.alpha)
        else:
            self.lasso_l1 = LogisticRegression(penalty='l1', C=self.alpha, solver='liblinear', fit_intercept=False,
                                               tol=1e-7)
        self.lasso_l1.fit(self.X_1, self.Y_1)
        self.beta_hat_1 = self.lasso_l1.coef_.reshape(self.p)
        self.sign = np.sign(self.beta_hat_1)
        self.E = self.beta_hat_1 != 0
        self.s_E = self.sign[self.E]
        self.X_E_1 = self.X_1[:, self.E]
        self.num_select = int(np.sum(self.E))
        if self.num_select == 0:
            print("select none")
            sys.exit(1)
        if data_type == "linear":
            self.lr = LinearRegression(fit_intercept=False)
        else:
            self.lr = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
        self.lr.fit(self.X_E_1, self.Y_1)
        self.beta_ls_1 = self.lr.coef_.squeeze()
        if self.beta_ls_1.size == 1:
            self.beta_ls_1 = self.beta_ls_1.reshape(1, )

        self.X = np.concatenate([self.X_1, self.X_2])
        self.Y = np.concatenate([self.Y_1, self.Y_2])
        self.X_E = self.X[:, self.E]
        self.lr.fit(self.X[:, self.E], self.Y)
        self.beta_ls = self.lr.coef_.squeeze().reshape(-1)
        if data_type == "linear":
            self.W1 = np.identity(self.n)
            self.W = np.identity(self.n + self.m)
        else:
            pr_hat = 1 / (1 + np.exp(-self.X_E @ self.beta_ls)).reshape(self.n + self.m)
            self.W = np.diag(pr_hat * (1 - pr_hat))
            pr_hat_1 = 1 / (1 + np.exp(-self.X_E_1 @ self.beta_ls_1)).reshape(self.n)
            self.W1 = np.diag(pr_hat_1 * (1 - pr_hat_1))
        self.G1 = self.X_E_1.T @ self.X_E_1
        self.G1_inv = np.linalg.inv(self.G1)
        self.Sigma1 = np.linalg.inv(self.X_E.T @ self.W @ self.X_E)
        self.G = self.X_E.T @ self.X_E
        self.G_inv = np.linalg.inv(self.G)
        self.Sigma2 = self.X[:, ~self.E].T @ self.W @ self.X[:, ~self.E] - (
                    self.X[:, ~self.E].T @ self.W @ self.X_E) @ np.linalg.inv(self.X_E.T @ self.W @ self.X_E) @ (self.X_E.T @ self.W @ self.X[:, ~self.E])
        self.Sigma2 = self.Sigma2 / self.n

        if data_type == "linear":
            self.D_0 = self.X[:, ~self.E].T @ (self.Y - self.X_E @ self.beta_ls) / np.sqrt(self.n + self.m)
        else:
            pr = 1 / (1 + np.exp(-self.X_E @ self.beta_ls)).reshape(self.n + self.m)
            self.D_0 = self.X[:, ~self.E].T @ (self.Y - pr) / np.sqrt(self.n + self.m)

        self.lr.fit(self.X_2[:, self.E], self.Y_2)
        self.beta_ls_2 = self.lr.coef_.squeeze().reshape(-1)

    def select(self, X_b, Y_b,):
        self.lasso_l1.fit(X_b, Y_b)
        beta_tmp = self.lasso_l1.coef_.reshape(self.p)
        return np.sign(beta_tmp)

    def linear_KKT(self, D_M, D_0):
        tmp = D_M - self.G1_inv @ self.s_E * self.lbd
        t1 = np.all(np.sign(tmp) == self.s_E)
        t2 = np.max(abs(self.X_1[:, ~self.E].T @ self.X_E_1 @ self.G1_inv @ self.s_E + 1 / self.lbd * D_0)) < 1
        return t1 & t2

    def logistic_KKT(self, D_M, D_0):
        tmp = 1 / (1 + np.exp(-self.X_E_1 @ D_M)) - self.lbd * self.X_E_1 @ self.G1_inv @ self.s_E
        eta_hat = np.log(tmp / (1 - tmp))
        beta_hat = self.G_inv @ self.X_E.T @ eta_hat
        t1 = np.all(np.sign(beta_hat) == self.s_E)
        t2 = np.max(abs(self.X_1[:, ~self.E].T @ self.X_E_1 @ self.G1_inv @ self.s_E + 1 / self.lbd * D_0)) < 1
        return t1 & t2

    def basis(self, X_b, Y_b):
        return np.concatenate([np.mean(X_b, 0), X_b.T @ Y_b / len(Y_b)])

    def remove_D0(self, Z_train, D0_train, gamma_D0):
        return Z_train - D0_train @ gamma_D0.T

    def test_statistic(self, X_b, Y_b, return_D0=False):
        """
        Compute the test statistic \hat{\theta}
        :param X_b:
        :param X_2_b:
        :return:
        """
        n_b = len(Y_b)
        self.lr.fit(X_b[:, self.E], Y_b)
        beta_ls_b = self.lr.coef_.squeeze()
        if self.num_select == 1:
            beta_ls_b = beta_ls_b.reshape(1, )
        if not return_D0:
            return beta_ls_b
        else:
            if self.data_type == 'linear':
                D0_b = X_b[:, ~self.E].T @ (Y_b - X_b[:, self.E] @ beta_ls_b) / np.sqrt(len(Y_b))
            else:
                pr = 1 / (1 + np.exp(-X_b[:, self.E] @ beta_ls_b)).reshape(n_b)
                D0_b = X_b[:, ~self.E].T @ (Y_b - pr) / np.sqrt(n_b)
            return beta_ls_b, D0_b

    def gen_train_data(self, ntrain, n_b, m_b, return_gamma=True, remove_D0=False, target_pos=-1):
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
        # target_pos = 50
        count_pos = 0
        for i in range(ntrain):
            # idx_1 = np.random.choice(self.n + self.m, n_b, replace=True)
            # X_1_b = self.X[idx_1, :]
            # Y_1_b = self.Y[idx_1]
            # idx_2 = np.random.choice(self.n + self.m, m_b, replace=True)
            # X_2_b = self.X[idx_2, :]
            # Y_2_b = self.Y[idx_2]
            idx_1 = np.random.choice(self.n, n_b, replace=True)
            X_1_b = self.X_1[idx_1, :]
            Y_1_b = self.Y_1[idx_1]
            idx_2 = np.random.choice(self.m, m_b, replace=True)
            X_2_b = self.X_2[idx_2, :]
            Y_2_b = self.Y_2[idx_2]
            X_b = np.concatenate([X_1_b, X_2_b])
            Y_b = np.concatenate([Y_1_b, Y_2_b])

            if np.sum(Y_1_b) == 0 or np.sum(1 - Y_1_b) == 0:
                continue

            Z_train.append(self.basis(X_b, Y_b))

            # self.lr.fit(X_1_b[:, self.E], Y_1_b)
            # beta_ls_b = self.lr.coef_.squeeze().reshape(-1)
            # D0_b = X_1_b[:, ~self.E].T @ (Y_1_b - X_1_b[:, self.E] @ beta_ls_b)
            # W_train.append(self.linear_KKT(beta_ls_b, D0_b))

            if np.all(abs(self.select(X_1_b, Y_1_b)) == abs(self.sign)):
                W_train.append(1)
                count_pos += 1
                print("get {} positive data".format(count_pos))
            else:
                W_train.append(0)
            if not remove_D0:
                theta_hat_train.append(self.test_statistic(X_b, Y_b))
            else:
                stat = self.test_statistic(X_b, Y_b, return_D0=True)
                theta_hat_train.append(stat[0])
                D0_train.append(stat[1])
            if count_pos == target_pos:
                break
        Z_train = np.array(Z_train)
        W_train = np.array(W_train)
        theta_hat_train = np.array(theta_hat_train)
        result = {"Z_train": Z_train, "W_train": W_train}
        if return_gamma:
            cov_Z_theta = (Z_train - np.mean(Z_train, 0)).T @ (theta_hat_train - np.mean(theta_hat_train, 0)) / ntrain
            var_theta = np.cov(theta_hat_train.T)
            if var_theta.size == 1:
                var_theta = var_theta.reshape(1, 1)
                cov_Z_theta = cov_Z_theta.reshape(len(cov_Z_theta), 1)
            gamma = cov_Z_theta @ np.linalg.inv(var_theta)
            result["gamma"] = gamma
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
