import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from blackbox_selectinf.usecase.BH import BH
from blackbox_selectinf.learning.learning import learn_select_prob, get_weight, get_CI
import torch
import matplotlib.pyplot as plt
import pickle
import argparse
from argparse import Namespace
import bisect

parser = argparse.ArgumentParser(description='BH')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--I', type=int, default=50)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--n_b', type=int, default=100)
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
parser.add_argument('--nonnull', default=False, action='store_true')
parser.add_argument('--cond_all', default=False, action='store_true')
parser.add_argument('--one_side', default=False, action='store_true')
args = parser.parse_args()


# args = Namespace(I=20, alpha=0.5, batch_size=100, cond_all=True, consec_epochs=5, epochs=1000,
#                  idx=98, loadmodel=False, logname='log', max_it=1, modelname='model_', n=100,
#                  n_b=100, nonnull=False, nrep=1, ntrain=1000, savemodel=False, thre=0.99,
#                  verbose=True)


def main():
    I = args.I
    n = args.n
    n_b = args.n_b
    ntrain = args.ntrain
    alpha = args.alpha
    sigma = 1
    mu = np.zeros(I)
    if args.nonnull:
        mu[: I // 4] = .1
    j = args.idx
    for j in range(args.idx, args.idx + args.nrep):
        seed = j
        rng = np.random.default_rng(seed)
        X = rng.standard_normal([I, n]) * sigma + np.tile(mu[:, None], (1, n))
        X_bar = np.mean(X, 1)
        if args.one_side:
            pvals = norm.cdf(X_bar / sigma * np.sqrt(n))
        else:
            pvals = 2 * (1 - norm.cdf(abs(X_bar) / sigma * np.sqrt(n)))
        pvals_sort_index = np.argsort(pvals)
        pvals_sorted = pvals[pvals_sort_index]
        fdr_bh = multipletests(pvals_sorted, alpha, method='fdr_bh', is_sorted=True)
        selected_group = pvals_sort_index[np.where(fdr_bh[0])[0]]
        selected_group = np.sort(selected_group)
        num_selected = len(selected_group)
        if num_selected == 0:
            print("select no group")
            continue
        print('select', num_selected, 'groups')
        logs = {}
        logs['selected_group'] = selected_group
        logs['num_selected'] = num_selected
        bh_class = BH(X, alpha, sigma, args.one_side)
        Z_data = bh_class.basis(X)
        # generate training data
        training_data = bh_class.gen_train_data(ntrain, n_b)
        Z_train = training_data[0]
        W_train = training_data[1]

        # conditional selecting all the selected groups
        if args.cond_all:
            W_train_all = np.prod(W_train[:, selected_group], 1) * np.prod(1 - W_train[:, ~selected_group], 1)
            net = None
            net, flag, pr_data = learn_select_prob(Z_train, W_train_all, Z_data=torch.tensor(Z_data, dtype=torch.float),
                                                   net=net, thre=args.thre,
                                                   consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                   batch_size=args.batch_size, verbose=args.verbose, print_every=100)
        if args.cond_all:
            logs['pr_data'] = pr_data.item()
            logs['ones'] = np.mean(W_train_all)
        # train nn
        target_sd = sigma / np.sqrt(n)
        target_var = target_sd ** 2
        covered_nn = np.zeros(num_selected)
        covered_naive = np.zeros(num_selected)
        covered_true = np.zeros(num_selected)
        interval_nn = np.zeros([num_selected, 2])
        interval_naive = np.zeros([num_selected, 2])
        interval_true = np.zeros([num_selected, 2])
        pr_datas = np.zeros(num_selected)
        ones = np.zeros(num_selected)
        s = 0
        for i in selected_group:
            W_train_i = W_train[:, i]
            print(i, mu[i], X_bar[i], np.mean(W_train_i))
            if not args.cond_all:
                net = None
                net, flag, pr_data = learn_select_prob(Z_train, W_train_i, Z_data=torch.tensor(Z_data, dtype=torch.float),
                                                       net=net, thre=args.thre,
                                                       consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                       batch_size=args.batch_size, verbose=args.verbose, print_every=100)
            gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 101)
            theta_data = X_bar[i]
            target_theta = theta_data + gamma_list
            target_theta = target_theta.reshape(-1, 1)
            N_0 = np.copy(Z_data)
            N_0[i] = 0
            Gamma = np.zeros([I, 1])
            Gamma[i] = 1
            weight_val = get_weight(net, target_theta, N_0, Gamma)

            interval, pvalue_nn = get_CI(target_theta, weight_val, target_var, theta_data, return_pvalue=True)
            interval_nn[s, :] = interval
            if interval[0] <= mu[i] <= interval[1]:
                covered_nn[s] = 1
            pr_datas[s] = pr_data.item()
            ones[s] = np.mean(W_train_i)

            # naive interval
            interval_naive[s, :] = tuple((norm.ppf(0.025) * target_sd, -norm.ppf(0.025) * target_sd)) + theta_data
            if interval_naive[s, 0] <= mu[i] <= interval_naive[s, 1]:
                covered_naive[s] = 1

            # true interval
            prob_gamma = np.zeros(len(target_theta))
            for k in range(len(target_theta)):
                pvals_b = np.delete(pvals_sorted, np.where(pvals_sort_index == i)).tolist()
                if args.one_side:
                    new_p = norm.cdf(target_theta[k] / sigma * np.sqrt(n))
                else:
                    new_p = 2 * (1 - norm.cdf(abs(target_theta[k]) / sigma * np.sqrt(n)))[0]
                bisect.insort(pvals_b, new_p)
                pvals_b = np.array(pvals_b)
                insert_index = np.where(pvals_b == new_p)[0]
                bh = multipletests(pvals_b, alpha, method='fdr_bh', is_sorted=True)
                if bh[0][insert_index]:
                    prob_gamma[k] = 1
            interval = get_CI(target_theta, prob_gamma, target_var, theta_data)
            interval_true[s, :] = interval
            if interval[0] <= mu[i] <= interval[1]:
                covered_true[s] = 1
            if interval[0] <= mu[i] <= interval[1]:
                covered_nn[s] = 1
            plt.plot(target_theta, weight_val, label='nn')
            plt.plot(target_theta, prob_gamma, ls='--', label='true')
            plt.plot(np.ones(2) * theta_data, [0, 1])
            plt.legend()
            plt.savefig('bh_{}.png'.format(i))
            plt.close()
            s += 1

        if not args.cond_all:
            logs['pr_data'] = pr_datas
            logs['ones'] = ones
        logs['interval_nn'] = interval_nn
        logs['width_nn'] = interval_nn[:, 1] - interval_nn[:, 0]
        logs['covered_nn'] = covered_nn
        logs['interval_true'] = interval_true
        logs['width_true'] = interval_true[:, 1] - interval_true[:, 0]
        logs['covered_true'] = covered_true
        logs['interval_naive'] = interval_naive
        logs['width_naive'] = interval_naive[:, 1] - interval_naive[:, 0]
        logs['covered_naive'] = covered_naive
        print(logs)

        path = open('{}_n_{}_I_{}_nb_{}_alpha_{}_{}.pickle'.format(args.logname, n, I, n_b, alpha, j), 'wb')
        pickle.dump(logs, path)
        path.close()


if __name__ == "__main__":
    main()

# alpha_corrected = alpha * len(selected_group) / I
# for i in selected_group:
#     mt = multipletests(pvals[i, :], alpha=alpha_corrected, method='fdr_bh')
#     rej = mt[0]
#     selected_index[i, rej] = 1
#     if np.sum(rej) == 0:
#         # print(i, "no report")
#         continue
#     else:
#         print(i)



##
# I = 100
# n = 2
# nrep = 500
# groups_selected = []
# num_reported = []
# for r in range(nrep):
#     X = np.random.randn(I, n)  # + mu
#     pvals = 2 * (1 - norm.cdf(abs(X)))  # np.minimum(norm.cdf(X), 1 - norm.cdf(X)) * 2
#     # pvals = np.random.rand(I, n)
#     selected_group = np.where(np.min(pvals, 1) < 0.05)[0]
#     groups_selected.append(len(selected_group) / I)
#     alpha = 0.05
#     alpha_corrected = alpha * len(selected_group) / I
#
#     tmp = []
#     for i in selected_group:
#         # rej = pvals[i, :] <= alpha / n * len(selected_group) / I
#
#         mt = multipletests(pvals[i, :], alpha=alpha_corrected, method='fdr_bh')
#         rej = mt[0]
#         tmp.append(np.sum(rej))
#         # print(mt[0])
#     num_reported.append(np.mean(tmp))
# print(np.mean(groups_selected))
# print(np.mean(num_reported))
#
# pvals = np.concatenate([np.random.rand(n // 2), np.random.rand(n // 2) * .1])
# mt = multipletests(pvals=pvals, alpha=0.2, method='fdr_bh')
# mt[0]
# (mt[1] <= 0.2) * 1 - mt[0] * 1
