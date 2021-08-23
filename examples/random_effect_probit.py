"""
random effect model
"""
import numpy as np
from scipy.stats import norm
from blackbox_selectinf.usecase.random_effect import random_effect
from importlib import reload
import blackbox_selectinf.usecase.random_effect
reload(blackbox_selectinf.usecase.random_effect)
from blackbox_selectinf.usecase.random_effect import random_effect
from blackbox_selectinf.learning.learning import get_CI
import argparse
import pickle
from sklearn import svm


parser = argparse.ArgumentParser(description='random effect')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n', type=int, default=300)
parser.add_argument('--n_b', type=int, default=30)
parser.add_argument('--m', type=int, default=500)
parser.add_argument('--m_b', type=int, default=500)
parser.add_argument('--I', type=int, default=500)
parser.add_argument('--sigma_a', type=float, default=0.0)
parser.add_argument('--level', type=float, default=0.8)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--max_it', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=2000)
parser.add_argument('--logname', type=str, default='log')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=5)
parser.add_argument('--basis_type', type=str, default='SS_nor')
parser.add_argument('--useplugin', action='store_true', default=False)
parser.add_argument('--var1', action='store_true', default=False)
args = parser.parse_args()


def main():
    I = args.I
    n = args.n
    n_b = args.n_b
    m = args.m
    ntrain = args.ntrain
    N = n * I
    N_2 = m * I
    sigma = 1
    mu = 1
    sigma_a = args.sigma_a  # null
    level = args.level
    basis_type = args.basis_type
    for j in range(args.idx, args.idx + args.nrep):
        print("simulation", j)
        np.random.seed(j)
        a = np.random.randn(I) * sigma_a  # random effects
        X = mu + np.tile(a[:, None], (1, n)) + np.random.randn(I, n) * sigma
        a_2 = np.random.randn(I) * sigma_a
        X_2 = mu + np.tile(a_2[:, None], (1, m)) + np.random.randn(I, m) * sigma
        X_bar_i = np.mean(X, 1)
        X_bar = np.mean(X_bar_i)
        SSA_1 = n * np.sum((X_bar_i - X_bar) ** 2)
        SSE_1 = np.sum((X - X_bar_i[:, None]) ** 2)
        F_stat = SSA_1 / SSE_1 * (N - I) / (I - 1)
        sigmasq_hat1 = SSE_1 / (N - I)
        sigmasq_a_hat1 = (SSA_1 / (I - 1) - SSE_1 / (N - I)) / n

        X_bar_i_2 = np.mean(X_2, 1)
        X_bar_2 = np.mean(X_bar_i_2)
        # estimate target_var using X_2 only
        SSA_2 = m * np.sum((X_bar_i_2 - X_bar_2) ** 2)
        SSE_2 = np.sum((X_2 - X_bar_i_2[:, None]) ** 2)
        # F_stat = SSA / SSE * (N - I) / (I - 1)
        sigmasq_hat2 = SSE_2 / (N_2 - I)
        sigmasq_a_hat2 = (SSA_2 / (I - 1) - SSE_2 / (N_2 - I)) / m
        # sigmasq_a_hat2 = np.maximum(0, sigmasq_a_hat2)
        print("sigmasq_a", sigma_a**2, "sigmasq_a_hat", sigmasq_a_hat1, sigmasq_a_hat2)
        print("sigmasq", sigma**2, "sigmasq_hat", sigmasq_hat1, sigmasq_hat2)

        target_var2 = 2 / (m**2) * ((sigmasq_hat2 + m * sigmasq_a_hat2)**2 / (I - 1) + sigmasq_hat2**2 / (N_2 - I))
        target_sd2 = np.sqrt(target_var2)

        target_var1 = 2 / (n ** 2) * (
                    (sigmasq_hat1 + n * sigmasq_a_hat1) ** 2 / (I - 1) + sigmasq_hat1 ** 2 / (N - I))
        target_sd1 = np.sqrt(target_var1)

        if args.var1:
            target_var = target_var1
            target_sd = target_sd1
        else:
            target_var = target_var2
            target_sd = target_sd2

        var_cheat = 2 / (n ** 2) * ((sigma**2 + n * sigma_a**2) ** 2 / (I - 1) + sigma ** 4 / (N - I))
        sd_cheat = np.sqrt(var_cheat)
        print("target_sd2", target_sd2, "sd_cheat", sd_cheat)

        re_class = random_effect(X, level, basis_type=basis_type)
        cutoff = re_class.cutoff
        theta_data, F_stat = re_class.test_statistic(X)
        reject = F_stat >= cutoff
        print(reject)
        if not reject:
            print("no report")
            continue
        logs = {}
        logs['seed'] = j
        logs['sigmasq_a'] = sigma_a ** 2
        Z_data = re_class.basis(X, basis_type=basis_type)

        mu_hat1 = np.mean(X)
        training_data = re_class.gen_parametric(ntrain=ntrain, n_b=n, mu=mu_hat1, sigma=np.sqrt(sigmasq_hat1),
                                                sigma_a=np.sqrt(max(sigmasq_a_hat1, 0)))
        Z_train1 = training_data[0]
        W_train1 = training_data[1]
        Gamma = training_data[2]
        theta_hat_train1 = training_data[3]

        Z_train = Z_train1
        W_train = W_train1
        N_0 = Z_data - Gamma * theta_data
        gamma_list = np.linspace(-5 * target_sd, 5 * target_sd, 201)
        target_theta = theta_data + gamma_list
        target_theta = target_theta.reshape(1, len(gamma_list))

        # SVR
        clf = svm.SVC(C=1)
        clf.fit(Z_train, W_train)
        pr_data = clf.predict(Z_data.reshape(1, -1))
        print("pr_data", pr_data)
        logs['pr_data'] = pr_data
        weight_val = np.zeros(len(gamma_list))
        for i in range(len(gamma_list)):
            Z_b = N_0 + Gamma * target_theta[0, i]
            weight_val[i] = clf.predict(Z_b.reshape(1, -1))

        interval_nn, pvalue_nn = get_CI(target_theta, weight_val, target_var, theta_data, return_pvalue=True)
        interval_nn = np.maximum(interval_nn, 0)
        print("interval_nn", interval_nn)
        logs['covered_nn'] = 0
        if interval_nn[0] <= sigma_a**2 <= interval_nn[1]:
            logs['covered_nn'] = 1
        print("covered_nn", logs['covered_nn'])
        logs['width_nn'] = interval_nn[1] - interval_nn[0]

        # true interval
        U = SSA_1 / n / (I - 1)  # normal, mean
        V = SSE_1 / n / (n - 1) / I
        mu_1 = sigmasq_a_hat2 + (1 - re_class.cutoff) * sigma ** 2 / n
        mu_2 = sigmasq_a_hat2
        nu_1 = 2 * (sigma ** 2 + n * sigmasq_a_hat1) ** 2 / (I - 1) / n ** 2
        nu_2 = 2 * sigma ** 4 / n ** 2 / (n - 1) / I
        prob_true = np.zeros(len(gamma_list))
        for i in range(len(gamma_list)):
            t = target_theta[0, i]
            cond_mean = mu_1 + (t - sigmasq_a_hat2) * (nu_1 + re_class.cutoff * nu_2) / (nu_1 + nu_2)
            cond_var = nu_1 + re_class.cutoff ** 2 * nu_2 - (nu_1 + re_class.cutoff * nu_2) ** 2 / (nu_1 + nu_2)
            prob_true[i] = norm.cdf(cond_mean / np.sqrt(cond_var))
        interval_true, pvalue_true = get_CI(target_theta[0], prob_true, target_var, theta_data, return_pvalue=True)

        interval_true = np.maximum(interval_true, 0)
        print("interval_true", interval_true)
        logs['covered_true'] = 0
        if interval_true[0] <= sigma_a ** 2 <= interval_true[1]:
            logs['covered_true'] = 1
        print("covered_true", logs['covered_true'])
        logs['width_true'] = interval_true[1] - interval_true[0]

        # naive interval
        interval_naive = sigmasq_a_hat1 + (norm.ppf(0.025) * target_sd, -norm.ppf(0.025) * target_sd)
        interval_naive = np.maximum(interval_naive, 0)
        print("interval_naive", interval_naive)
        logs['covered_naive'] = 0
        if interval_naive[0] <= sigma_a ** 2 <= interval_naive[1]:
            logs['covered_naive'] = 1
        print("covered_naive", logs['covered_naive'])
        logs['width_naive'] = interval_naive[1] - interval_naive[0]

        # stage 2 interval
        interval_2 = sigmasq_a_hat2 + (norm.ppf(0.025) * target_sd2, -norm.ppf(0.025) * target_sd2)
        interval_2 = np.maximum(interval_2, 0)
        print("interval_2", interval_2)
        logs['covered_2'] = 0
        if interval_2[0] <= sigma_a ** 2 <= interval_2[1]:
            logs['covered_2'] = 1
        print("covered_2", logs['covered_2'])
        logs['width_2'] = interval_2[1] - interval_2[0]

        path = open('{}_n_{}_I_{}_nb_{}_m_{}_sigmaa_{}_{}.pickle'.format(args.logname, n, I, n_b, m, sigma_a, j), 'wb')
        pickle.dump(logs, path)
        path.close()


if __name__ == "__main__":
    main()

