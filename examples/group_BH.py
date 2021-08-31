import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from blackbox_selectinf.usecase.BH import group_BH
from blackbox_selectinf.learning.learning import learn_select_prob, get_weight, get_CI
import torch
import matplotlib.pyplot as plt
import pickle
import argparse
from argparse import Namespace
import bisect

parser = argparse.ArgumentParser(description='group BH')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--I', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--n_b', type=int, default=100)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--max_it', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--logname', type=str, default='gBH')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=5)
parser.add_argument('--nonnull', default=False, action='store_true')
parser.add_argument('--cond_all', default=False, action='store_true')
parser.add_argument('--one_side', default=False, action='store_true')
parser.add_argument('--mean', default=False, action='store_true')
parser.add_argument('--fix_level', default=False, action='store_true')
parser.add_argument('--nullneg', default=False, action='store_true')
args = parser.parse_args()


def main():
    I = args.I
    m = args.m
    n = args.n
    n_b = args.n_b
    ntrain = args.ntrain
    alpha = args.alpha
    sigma = 1
    mu = np.zeros([m, I])
    if args.nonnull:
        mu[:, :I // 4] = .1
    if args.nullneg:
        mu[:, :I // 4] = -.1
    j = args.idx
    for j in range(args.idx, args.idx + args.nrep):
        seed = j
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        X = rng.standard_normal([m, I, n]) * sigma + np.tile(mu[:, :, None], (1, 1, n))
        X_bar = np.mean(X, 2)
        if args.one_side:
            pvals = norm.cdf(X_bar / sigma * np.sqrt(n))
        else:
            pvals = 2 * (1 - norm.cdf(abs(X_bar) / sigma * np.sqrt(n)))
        if args.mean:
            pvals_group_min = np.mean(pvals, 1)
            print(pvals_group_min)
            selected_family = np.where(pvals_group_min <= 0.5)[0]
        else:
            pvals_group_min = np.min(pvals, 1)
            selected_family = np.where(pvals_group_min <= 0.05)[0]
        selected_group = np.zeros([m, I])
        alpha_corrected = alpha * len(selected_family) / m
        for t in selected_family:
            bh = multipletests(pvals[t, :], alpha_corrected, method='fdr_bh')
            selected_group[t, :] = bh[0]
        num_selected = np.sum(selected_group)
        if num_selected < 1:
            print("select no group")
            continue
        print('select families', selected_family)
        print(np.sum(selected_group, 1))
        logs = {}
        logs['selected_group'] = selected_group
        logs['num_selected'] = num_selected
        bh_class = group_BH(X, alpha, sigma, args.one_side)
        Z_data = bh_class.basis(X)

        target_sd = sigma / np.sqrt(n)
        target_var = target_sd ** 2

        # generate training data
        training_data = bh_class.gen_train_data(ntrain, n_b, selected_family, mean=args.mean, fix_level=args.fix_level)
        Z_train = training_data[0]
        W_train = training_data[1]

        interval_nn = np.zeros([m, I, 2]) - 1
        width_nn = np.zeros([m, I]) - 1
        covered_nn = np.zeros([m, I]) - 1
        interval_naive = np.zeros([m, I, 2]) - 1
        width_naive = np.zeros([m, I]) - 1
        covered_naive = np.zeros([m, I]) - 1
        pr_datas = np.zeros([m, I]) - 1
        ones = np.zeros([m, I]) - 1
        for t in selected_family:
            for i in np.where(selected_group[t, :])[0]:
                W_train_i = W_train[:, np.where(selected_family == t)[0][0], i]
                print(t, i, np.mean(W_train_i))
                net = None
                net, flag, pr_data = learn_select_prob(Z_train, W_train_i,
                                                       Z_data=torch.tensor(Z_data, dtype=torch.float),
                                                       net=net, thre=args.thre,
                                                       consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                       batch_size=args.batch_size, verbose=args.verbose,
                                                       print_every=100)
                gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 101)
                theta_data = X_bar[t, i]
                target_theta = theta_data + gamma_list
                target_theta = target_theta.reshape(-1, 1)
                N_0 = np.copy(Z_data)
                N_0[t * I + i] = 0
                Gamma = np.zeros([I * m, 1])
                Gamma[t * I + i] = 1
                weight_val = get_weight(net, target_theta, N_0, Gamma)
                print(weight_val[50])
                interval = get_CI(target_theta, weight_val, target_var, theta_data)
                interval_nn[t, i, :] = interval
                width_nn[t, i] = interval[1] - interval[0]
                if interval[0] <= mu[t, i] <= interval[1]:
                    covered_nn[t, i] = 1
                else:
                    covered_nn[t, i] = 0
                pr_datas[t, i] = pr_data.item()
                ones[t, i] = np.mean(W_train_i)

                # naive interval
                interval_naive[t, i, :] = tuple((norm.ppf(0.025) * target_sd, -norm.ppf(0.025) * target_sd)) + theta_data
                if interval_naive[t, i, 0] <= mu[t, i] <= interval_naive[t, i, 1]:
                    covered_naive[t, i] = 1
                else:
                    covered_naive[t, i] = 0
                width_naive[t, i] = -2 * norm.ppf(0.025) * target_sd

        logs['interval_nn'] = interval_nn
        logs['width_nn'] = width_nn
        logs['covered_nn'] = covered_nn
        logs['interval_naive'] = interval_naive
        logs['width_naive'] = width_naive
        logs['covered_naive'] = covered_naive
        logs['pr_data'] = pr_datas
        logs['ones'] = ones
        print(covered_nn[selected_group == 1])
        print(covered_naive[selected_group == 1])
        path = open('{}_n_{}_I_{}_nb_{}_alpha_{}_{}.pickle'.format(args.logname, n, I, n_b, alpha, j), 'wb')
        pickle.dump(logs, path)
        path.close()


if __name__ == "__main__":
    main()
