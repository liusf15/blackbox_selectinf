"""
Drop-the-loser with correlated data
"""

from importlib import reload
from blackbox_selectinf.usecase.DTL_corr import DTL_corr
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import numpy as np
import argparse
import pickle
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
from selectinf.distributions.discrete_family import discrete_family


parser = argparse.ArgumentParser(description='DTL')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--m_dep', type=int, default=10, help='m dependent')
parser.add_argument('--block_bootstrap', action='store_false', default=True)
parser.add_argument('--true_sigma', action='store_false', default=True)
parser.add_argument('--blocklength', type=int, default=20)
parser.add_argument('--selection', type=str, default='mean')
parser.add_argument('--uc', type=float, default=2)
parser.add_argument('--basis_type', type=str, default='naive')
parser.add_argument('--indep', action='store_true', default=False)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--m', type=int, default=500)
parser.add_argument('--n_b', type=int, default=1000)
parser.add_argument('--m_b', type=int, default=500)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=5000)
parser.add_argument('--logname', type=str, default='dtlts')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--nonnull', action='store_true', default=False)
args = parser.parse_args()


def main():
    n = args.n
    m = args.m
    n_b = args.n_b
    m_b = args.m_b
    K = args.K
    uc = args.uc
    selection = args.selection
    ntrain = args.ntrain
    mu_list = np.zeros(K)
    if args.nonnull:
        mu_list[:25] = .1
    m_dep = args.m_dep
    rho = 0.5
    blocklength = args.blocklength
    logs = [dict() for x in range(args.nrep)]
    for j in range(args.idx, args.idx + args.nrep):
        print("Starting simulation", j)
        logs[j - args.idx]['seed'] = j
        np.random.seed(j)
        # Stage 1
        X = np.zeros([K, n])
        for k in range(K):
            u = np.random.randn(n + m_dep)
            ep = 0
            for l in range(m_dep):
                ep += u[l:(l + n)]
            X[k, :] = mu_list[k] + ep

        X_bar = np.mean(X, 1)
        if selection == 'mean':
            max_rest = np.sort(X_bar)[-2]
            win_idx = np.argmax(X_bar)
        elif selection == "UC":
            UC = X_bar + uc * np.std(X, axis=1, ddof=1)
            win_idx = np.argmax(UC)
            max_rest = np.sort(UC)[-2]
        else:
            raise AssertionError("invalid selection")
        # Stage 2
        u = np.random.randn(m + m_dep)
        ep = 0
        for l in range(m_dep):
            ep += u[l:(l + m)]
        X_2 = mu_list[win_idx] + ep

        DTL_class = DTL_corr(X, X_2)
        Z_data = DTL_class.basis(X, X_2, args.basis_type)
        theta_data = DTL_class.theta_hat

        training_data = DTL_class.gen_train_data(ntrain=ntrain, n_b=n_b, m_b=m_b, basis_type=args.basis_type, blocklength=blocklength, remove_D0=args.indep)
        Z_train = training_data['Z_train']
        W_train = training_data['W_train']
        gamma = training_data['gamma']

        # train
        net = learn_select_prob(Z_train, W_train, num_epochs=args.epochs, batch_size=args.batch_size, verbose=True)
        pr_data = net(torch.tensor(Z_data, dtype=torch.float))
        print('pr_data', pr_data)
        logs[j - args.idx]['pr_data'] = pr_data
        if args.indep:
            gamma_D0 = training_data['gamma_D0']
            Z_data = Z_data - gamma_D0 @ DTL_class.D_0.reshape(1, )
        N_0 = Z_data - gamma @ theta_data.reshape(1, )

        sigma_sq = m_dep ** 2
        var_M_true = (1 / (n + m)) * sigma_sq
        var_0_true = (1 / n - 1 / (n + m)) * sigma_sq
        target_var = var_M_true
        target_sd = np.sqrt(target_var)
        gamma_list = np.linspace(-20 / np.sqrt(n_b + m_b), 20 / np.sqrt(n_b + m_b), 101)
        target_theta = theta_data + gamma_list
        target_theta = target_theta.reshape(1, 101)
        weight_val = get_weight(net, target_theta, N_0, gamma)
        interval_nn, pvalue_nn = get_CI(target_theta, weight_val, target_var, theta_data, return_pvalue=True)
        logs[j - args.idx]['interval_nn'] = interval_nn
        if interval_nn[0] <= mu_list[DTL_class.win_idx] <= interval_nn[1]:
            logs[j - args.idx]['covered_nn'] = 1
        else:
            logs[j - args.idx]['covered_nn'] = 0
        logs[j - args.idx]['width_nn'] = interval_nn[1] - interval_nn[0]
        logs[j - args.idx]['pvalue_nn'] = pvalue_nn

        ##################################################
        # check learning
        count = 0
        nb = 50
        X_pooled = np.concatenate([X[win_idx], X_2])
        pval = []
        num_blocks_1 = n_b // blocklength
        num_blocks_2 = m_b // blocklength
        for ell in range(int(nb / np.mean(W_train))):
            X_b = np.zeros([K, num_blocks_1 * blocklength])
            for k in range(K):
                if k != win_idx:
                    indices = np.random.choice(n - blocklength, num_blocks_1, replace=True)
                    for s in range(num_blocks_1):
                        idx = indices[s]
                        data_block = X[k, idx: idx + blocklength]
                        X_b[k, s * blocklength: (s + 1) * blocklength] = data_block
                if k == win_idx:
                    indices = np.random.choice(n + m - blocklength, num_blocks_1, replace=True)
                    for s in range(num_blocks_1):
                        idx = indices[s]
                        data_block = X_pooled[idx: idx + blocklength]
                        X_b[k, s * blocklength: (s + 1) * blocklength] = data_block
            if selection == 'mean':
                idx = np.argmax(np.mean(X_b, 1))
            else:
                idx = np.argmax(np.mean(X_b, 1) + uc * np.std(X_b, axis=1, ddof=1))
            if idx != win_idx:
                continue
            else:
                count += 1
                X_2_b = []
                indices = np.random.choice(n + m - blocklength, num_blocks_2, replace=True)
                for s in range(num_blocks_2):
                    idx = indices[s]
                    X_2_b.extend(X_pooled[idx: idx + blocklength])
                X_2_b = np.array(X_2_b)
                d_M = np.mean(np.concatenate([X_b[win_idx], X_2_b]))
                observed_target = d_M
                target_theta_0 = d_M + gamma_list
                target_theta_0 = target_theta_0.reshape(1, 101)
                target_val = target_theta_0
                weight_val = get_weight(net, target_theta_0, N_0, gamma)
                weight_val_2 = weight_val * norm.pdf((target_val - observed_target) / target_sd)
                exp_family = discrete_family(target_val.reshape(-1), weight_val_2.reshape(-1))
                hypothesis = theta_data
                pivot = exp_family.cdf((hypothesis - observed_target) / target_var, x=observed_target)
                pivot = 2 * min(pivot, 1 - pivot)
                pval.append(pivot)
                if count == nb:
                    break
        pval = np.array(pval)
        logs[j - args.idx]['pval'] = pval
        logs[j - args.idx]['false_rej'] = sum(pval <= 0.05) / len(pval)
        print(pval)
        print("reject:", sum(pval <= 0.05) / len(pval))

        ##################################################
        # true interval
        var_0 = var_0_true
        observed_target = theta_data
        gamma_list = np.linspace(-20 * target_sd, 20 * target_sd, 101)
        target_val = gamma_list + theta_data
        prob_gamma_true = []
        for gamma in gamma_list:
            if selection == "mean":
                prob_gamma_true.append(norm.sf((np.sort(X_bar)[-2] - theta_data - gamma) / np.sqrt(var_0)))
            if selection == "UC":
                tmp = []
                for i in range(1000):
                    tmp.append(np.sqrt(var_0) * np.random.randn(1)
                               + uc / np.sqrt(n - 1) * np.sqrt(np.random.chisquare(df=n - 1)))
                prob_gamma_true.append(sum(tmp > max_rest - theta_data - gamma) / len(tmp))
        interval_true, pvalue_true = get_CI(target_val, np.squeeze(prob_gamma_true), target_var, observed_target, return_pvalue=True)
        logs[j - args.idx]['interval_true'] = interval_true
        if interval_true[0] <= mu_list[DTL_class.win_idx] <= interval_true[1]:
            logs[j - args.idx]['covered_true'] = 1
        else:
            logs[j - args.idx]['covered_true'] = 0
        logs[j - args.idx]['width_true'] = interval_true[1] - interval_true[0]
        logs[j - args.idx]['pvalue_true'] = pvalue_true

        plt.figure()
        plt.plot(target_val, weight_val, label="nn")
        plt.plot(target_val, prob_gamma_true, label="truth")
        plt.legend()
        plt.savefig("{}_n_{}_m_{}_K_{}_nb_{}_{}.png".format(args.logname, n, m, K, n_b, j))


        ##################################################
        # stage 2 interval
        var_2 = sigma_sq / m
        interval_2 = tuple((norm.ppf(0.025) * np.sqrt(var_2), -norm.ppf(0.025) * np.sqrt(var_2)) + np.mean(X_2))

        logs[j - args.idx]['interval_2'] = interval_2
        if interval_2[0] <= mu_list[DTL_class.win_idx] <= interval_2[1]:
            logs[j - args.idx]['covered_2'] = 1
        else:
            logs[j - args.idx]['covered_2'] = 0
        logs[j - args.idx]['width_2'] = interval_2[1] - interval_2[0]
        pivot = norm.cdf(-np.mean(X_2) / np.sqrt(var_2))
        logs[j - args.idx]['pvalue_2'] = 2 * min(pivot, 1 - pivot)

        X_cat = np.concatenate([X[win_idx, :], X_2])
        interval_naive = tuple((norm.ppf(0.025) * np.sqrt(var_M_true), -norm.ppf(0.025) * np.sqrt(var_M_true)) + np.mean(X_cat))
        width_naive = -2 * norm.ppf(0.025) * np.sqrt(var_M_true)
        logs[j - args.idx]['interval_naive'] = interval_naive
        logs[j - args.idx]['width_naive'] = width_naive
        if interval_naive[0] <= mu_list[DTL_class.win_idx] <= interval_naive[1]:
            logs[j - args.idx]['covered_naive'] = 1
        else:
            logs[j - args.idx]['covered_naive'] = 0
        pivot = norm.cdf(-np.mean(X_cat) / np.sqrt(var_M_true))
        logs[j - args.idx]['pvalue_naive'] = 2 * min(pivot, 1 - pivot)

        logs[j - args.idx]['mu_true'] = mu_list[DTL_class.win_idx]

        path = open("{}_n_{}_m_{}_K_{}_nb_{}_{}.pickle".format(args.logname, n, m, K, n_b, j), 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()
    print(logs)


if __name__ == "__main__":
    main()
