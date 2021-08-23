import torch
import numpy as np
from scipy.stats import norm
from blackbox_selectinf.usecase.AR_model import AR_model
from importlib import reload
import blackbox_selectinf.usecase.AR_model
reload(blackbox_selectinf.usecase.AR_model)
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import argparse
import pickle
from statsmodels.stats.stattools import durbin_watson


parser = argparse.ArgumentParser(description='AR model inference for beta')
parser.add_argument('--basis_type', type=str, default='linear')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--p', type=int, default=10)
parser.add_argument('--n_b', type=int, default=100)
parser.add_argument('--rho', type=float, default=0.0)
parser.add_argument('--Q_L', type=float, default=1.9)
parser.add_argument('--Q_U', type=float, default=2.2)
parser.add_argument('--upper', action='store_false', default=True)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--max_it', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--logname', type=str, default='log')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=5)
args = parser.parse_args()


def main():
    Q_L = args.Q_L
    Q_U = args.Q_U
    n = args.n
    p = args.p
    rho = args.rho
    n_b = args.n_b
    ntrain = args.ntrain
    max_it = args.max_it
    j = args.idx
    for j in range(args.idx, args.idx + args.nrep):
        logs = {}
        print("Start simulation {}".format(j))
        # generate data
        seed = j
        logs['seed'] = seed
        np.random.seed(seed)
        X = np.random.randn(n, p)
        beta = np.zeros(p)
        sigma = 1
        C = np.tile(np.arange(1, n + 1), (n, 1))
        C_cov = np.power(rho, abs(C - C.T)) / (1 - rho ** 2) * sigma**2
        C_inv = np.linalg.inv(C_cov)
        epsilon = np.random.multivariate_normal(np.zeros(n), C_cov)
        Y = X @ beta + epsilon
        hat = X @ np.linalg.inv(X.T @ X) @ X.T
        resids = Y - hat @ Y
        dw_stat = durbin_watson(resids)
        if args.upper and dw_stat >= Q_U:
            print("reject")
            print("DW ", dw_stat, 'Q_L', Q_L, 'Q_U', Q_U)
        elif not args.upper and dw_stat <= Q_L:
            print("reject")
            print("DW ", dw_stat)
        else:
            continue
        logs['dw'] = dw_stat

        AR_class = AR_model(X, Y, Q_L=Q_L, Q_U=Q_U, upper=args.upper, basis_type=args.basis_type)

        rho_hat = (np.mean(resids[1:] * resids[:-1]) - np.mean(resids[1:]) * np.mean(resids[:-1])) / \
                  (np.mean(resids[:-1]**2) - np.mean(resids[:-1])**2)

        beta_hat = np.linalg.inv(X.T @ C_inv @ X) @ X.T @ C_inv @ Y
        if args.basis_type == 'residual':
            Z_data = AR_class.basis(resids)
            theta_data = rho_hat
        else:
            Z_data = AR_class.basis_linear(X, Y)
            theta_data = beta_hat

        logs['rho_hat'] = rho_hat
        logs['beta_hat'] = beta_hat

        # generate training data
        training_data = AR_class.gen_train_data(ntrain, n, beta_hat, rho_hat)
        Z_train = training_data[0]
        W_train = training_data[1]
        Gamma = training_data[2]
        target_var = np.diag(training_data[3])
        target_sd = np.sqrt(target_var)
        logs['target_sd'] = target_sd
        print("ones:", np.mean(W_train))
        logs['ones'] = np.mean(W_train)

        print("Start learning selection probability")
        net = None
        for it in range(max_it):
            print("recursion", it)
            net, flag, pr_data = learn_select_prob(Z_train, W_train, Z_data=torch.tensor(Z_data, dtype=torch.float),
                                                   net=net, thre=args.thre,
                                                   consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                   batch_size=args.batch_size, verbose=args.verbose, print_every=100)
            if flag == 1:
                print("Succeeded learning!")
                break
            if it == max_it - 1:
                break
            else:  # generate more data
                print("generate more data")
                training_data = AR_class.gen_train_data(ntrain=ntrain, n_b=n_b, beta_hat=beta_hat, rho_hat=rho_hat)
                Z_train_new = training_data[0]
                W_train_new = training_data[1]
                Z_train = np.concatenate([Z_train, Z_train_new])
                W_train = np.concatenate([W_train, W_train_new])
                print("fraction of positive data:", np.mean(W_train))
        print('pr_data', pr_data.item())
        logs['pr_data'] = pr_data.item()
        N_0 = Z_data - Gamma @ theta_data
        gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 201)
        target_theta = theta_data + gamma_list
        Sigma1 = np.linalg.inv(X.T @ C_inv @ X)
        interval_nn = np.zeros([p, 2])
        covered_nn = np.zeros(p)
        for k in range(p):
            Gamma_k = Sigma1[:, k] / Sigma1[k, k]
            target_theta_k = theta_data[k] + gamma_list[:, k]
            target_theta = theta_data + np.outer(gamma_list[:, k], Gamma_k)
            weight_val = get_weight(net, target_theta, N_0, Gamma)
            interval = get_CI(target_theta_k, weight_val, target_var[k], theta_data[k])
            interval_nn[k, :] = interval
            if interval[0] <= beta[k] <= interval[1]:
                covered_nn[k] = 1

        print("interval_nn", interval_nn)
        logs['covered_nn'] = covered_nn
        print("covered_nn", logs['covered_nn'])
        logs['interval_nn'] = interval_nn
        logs['width_nn'] = interval_nn[:, 1] - interval_nn[:, 0]

        # naive interval
        interval_naive = np.zeros([p, 2])
        covered_naive = np.zeros(p)
        for k in range(p):
            interval_naive[k, 0] = beta_hat[k] + norm.ppf(0.025) * target_sd[k]
            interval_naive[k, 1] = beta_hat[k] - norm.ppf(0.025) * target_sd[k]
            if interval_naive[k, 0] <= beta[k] <= interval_naive[k, 1]:
                covered_naive[k] = 1
        print("interval_naive", interval_naive)
        logs['covered_naive'] = covered_naive
        print("covered_naive", logs['covered_naive'])
        logs['width_naive'] = interval_naive[:, 1] - interval_naive[:, 0]

        path = open('{}_n_{}_p_{}_nb_{}_rho_{}_{}.pickle'.format(args.logname, n, p, n_b, rho, j), 'wb')
        pickle.dump(logs, path)
        path.close()


if __name__ == "__main__":
    main()
