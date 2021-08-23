"""
Two-sample test: increase sample size until significant
"""

import torch
import numpy as np
from scipy.stats import t
from scipy.stats import norm
from blackbox_selectinf.usecase.increase_size import two_sample
from importlib import reload
import blackbox_selectinf.usecase.increase_size
reload(blackbox_selectinf.usecase.increase_size)
from blackbox_selectinf.usecase.increase_size import two_sample
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import argparse
import pickle

parser = argparse.ArgumentParser(description='random effect')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n_1', type=int, default=50)
parser.add_argument('--n_2', type=int, default=50)
parser.add_argument('--m_1', type=int, default=50)
parser.add_argument('--m_2', type=int, default=50)
parser.add_argument('--mu_1', type=float, default=0.0)
parser.add_argument('--mu_2', type=float, default=0.0)
parser.add_argument('--alpha_0', type=float, default=0.1)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--max_it', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=5000)
parser.add_argument('--logname', type=str, default='log')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=5)
parser.add_argument('--cond_exact', action='store_false', default=True)
args = parser.parse_args()


def main():
    n_1 = args.n_1
    n_2 = args.n_2
    m_1 = args.m_1
    m_2 = args.m_2
    mu_1 = args.mu_1
    mu_2 = args.mu_2
    sigma = 1
    alpha_0 = args.alpha_0
    ntrain = args.ntrain
    j = args.idx
    for j in range(args.idx, args.idx + args.nrep):
        print("simulation", j)
        logs = {}
        logs['seed'] = j
        np.random.seed(j)
        X_1 = mu_1 + sigma * np.random.randn(n_1)
        X_2 = mu_2 + sigma * np.random.randn(n_2)
        s_all = np.sqrt(
            ((n_1 - 1) * np.std(X_1, ddof=1) + (n_2 - 1) * np.std(X_2, ddof=1)) / (n_1 + n_2 - 2)) * np.sqrt(
            1 / n_1 + 1 / n_2)
        t_stat = (np.mean(X_1) - np.mean(X_2)) / s_all
        cutoff = t.ppf(1 - alpha_0 / 2, n_1 + n_2 - 2)
        print(abs(t_stat), cutoff)
        k = 0
        if abs(t_stat) < cutoff:
            for i in range(100):
                A_1 = mu_1 + sigma * np.random.randn(m_1)
                A_2 = mu_2 + sigma * np.random.randn(m_2)
                X_1 = np.concatenate([X_1, A_1])
                X_2 = np.concatenate([X_2, A_2])
                N_1 = len(X_1)
                N_2 = len(X_2)
                s_all = np.sqrt(((N_1 - 1) * np.std(X_1, ddof=1) + (N_2 - 1) * np.std(X_2, ddof=1)) / (N_1 + N_2 - 2)) * np.sqrt(1 / N_1 + 1 / N_2)
                t_stat = (np.mean(X_1) - np.mean(X_2)) / s_all
                cutoff = t.ppf(1 - alpha_0 / 2, N_1 + N_2 - 2)
                if abs(t_stat) >= cutoff:
                    print(i, abs(t_stat))
                    k = i + 1
                    print("rejected after {} repeats, T={}".format(k, t_stat))
                    break

        if abs(t_stat) < cutoff:
            print("not rejected")
            continue
        # naive interval
        target_sd = s_all
        target_var = target_sd ** 2
        interval_naive = np.mean(X_1) - np.mean(X_2) + (t.ppf(0.025, N_1 + N_2 - 2) * target_sd, -t.ppf(0.025, N_1 + N_2 - 2) * target_sd)
        covered_naive = 0
        if interval_naive[0] <= mu_1 - mu_2 <= interval_naive[1]:
            print("naive covered", interval_naive)
            covered_naive = 1
        width_naive = interval_naive[1] - interval_naive[0]
        logs['interval_naive'] = interval_naive
        logs['covered_naive'] = covered_naive
        logs['width_naive'] = width_naive

        ts_class = two_sample(X_1, X_2, k, mu_1, mu_2, sigma, n_1, n_2, m_1, m_2, alpha_0)
        Z_data = ts_class.basis(X_1, X_2)
        # theta_data = ts_class.test_statistic(X_1, X_2)
        theta_data = np.mean(X_1) - np.mean(X_2)
        # generate training data
        training_data = ts_class.gen_train_data(ntrain, cond_exact=args.cond_exact)
        Z_train = training_data[0]
        W_train = training_data[1]
        Gamma = training_data[2]
        print(np.mean(W_train))
        logs['ones'] = np.mean(W_train)

        print("Start learning selection probability")
        net = None
        max_it = 1
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
                training_data = ts_class.gen_train_data(ntrain=ntrain)
                Z_train_new = training_data[0]
                W_train_new = training_data[1]
                Z_train = np.concatenate([Z_train, Z_train_new])
                W_train = np.concatenate([W_train, W_train_new])
                print("fraction of positive data:", np.mean(W_train))
        print('pr_data', pr_data.item())
        logs['pr_data'] = pr_data.item()
        N_0 = Z_data - Gamma * theta_data

        gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 201)
        target_theta = theta_data + gamma_list
        target_theta = target_theta.reshape(1, len(gamma_list))
        weight_val = get_weight(net, target_theta, N_0, Gamma.reshape(-1, 1))

        interval_nn, pvalue_nn = get_CI(target_theta, weight_val, target_var, theta_data, return_pvalue=True)
        print("interval_nn", interval_nn)
        logs['covered_nn'] = 0
        if interval_nn[0] <= mu_1 - mu_2 <= interval_nn[1]:
            logs['covered_nn'] = 1
        print("covered_nn", logs['covered_nn'])
        logs['width_nn'] = interval_nn[1] - interval_nn[0]
        logs['interval_nn'] = interval_nn

        path = open('{}_n_{}_m_{}_mu1_{}_mu2_{}_{}.pickle'.format(args.logname, n_1, m_1, mu_1, mu_2, j), 'wb')
        pickle.dump(logs, path)
        path.close()


if __name__ == "__main__":
    main()
