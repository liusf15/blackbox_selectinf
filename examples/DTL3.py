"""
3 stage drop-the-loser
"""
from importlib import reload
import blackbox_selectinf.usecase.DTL3
reload(blackbox_selectinf.usecase.DTL3)
from blackbox_selectinf.usecase.DTL3 import DropTheLoser3
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import DTL_vae
import numpy as np
import argparse
import pickle
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
from selectinf.distributions.discrete_family import discrete_family
from argparse import Namespace


parser = argparse.ArgumentParser(description='DTL')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--n_1', type=int, default=500)
parser.add_argument('--n_2', type=int, default=500)
parser.add_argument('--n_3', type=int, default=100)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--logname', type=str, default='dtl3_')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--nonnull', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=2)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


def main():
    seed = args.idx
    n_1 = args.n_1
    n_2 = args.n_2
    n_3 = args.n_3
    K = args.K
    ntrain = args.ntrain
    mu_list = np.zeros(K)
    if args.nonnull:
        mu_list[:25] = .1
    logs = [dict() for x in range(args.nrep)]
    for j in range(args.idx, args.idx + args.nrep):
        print("Starting simulation", j)
        logs[j - args.idx]['seed'] = j
        np.random.seed(j)
        # Stage 1
        X_1 = np.zeros([K, n_1])
        for k in range(K):
            X_1[k, :] = np.random.randn(n_1) + mu_list[k]
        X_bar_1 = np.mean(X_1, 1)
        win_idx1 = np.argsort(X_bar_1)[::-1]
        win_idx1 = win_idx1[-10:]
        # Stage 2
        X_2 = np.random.randn(10, n_2) + np.tile(mu_list[win_idx1].reshape(-1, 1), (1, n_2))
        X_bar_2 = np.mean(X_2, 1)
        win_idx_in_2 = np.argmax(X_bar_2)
        win_idx = win_idx1[win_idx_in_2]
        # Stage 3
        X_3 = np.random.randn(n_3) + mu_list[win_idx]
        # put together
        X = []
        for k in range(K):
            if k not in win_idx1:
                X.append(X_1[k, :])
            elif k == win_idx:
                X.append(np.hstack([X_1[win_idx], X_2[win_idx_in_2], X_3]))
            else:
                X.append(np.hstack([X_1[k, :], X_2[list(win_idx1).index(k), :]]))

        DTL_class = DropTheLoser3(X, n_1, n_2, n_3, win_idx1, win_idx)
        Z_data = DTL_class.basis(X)
        theta_data = DTL_class.theta_hat

        print("Generate initial data")
        training_data = DTL_class.gen_train_data(ntrain=ntrain)
        Z_train = training_data['Z_train']
        W_train = training_data['W_train']
        gamma = training_data['gamma']
        print(np.mean(W_train))

        # train
        print("Start learning selection probability")
        net = None
        for ii in range(5):
            print("recursion", ii)
            print(Z_train.shape)
            net, flag, pr_data = learn_select_prob(Z_train, W_train, Z_data=torch.tensor(Z_data, dtype=torch.float), net=net, thre=args.thre,
                                                   consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                   batch_size=args.batch_size, verbose=args.verbose, print_every=100)
            if flag == 1:
                print("Succeeded learning!")
                break
            else:  # generate more data
                print("generate more data")
                training_data = DTL_class.gen_train_data(ntrain=ntrain)
                Z_train_new = training_data['Z_train']
                W_train_new = training_data['W_train']
                Z_train = np.concatenate([Z_train, Z_train_new])
                W_train = np.concatenate([W_train, W_train_new])

        logs[j - args.idx]['pr_data'] = pr_data
        logs[j - args.idx]['flag'] = flag
        N_0 = Z_data - gamma @ theta_data.reshape(1, )

        n_all = n_1 + n_2 + n_3
        target_var = 1 / n_all
        target_sd = np.sqrt(target_var)
        gamma_list = np.linspace(-20 / np.sqrt(n_all), 20 / np.sqrt(n_all), 101)
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
        print("pvalue", pvalue_nn)
        print(logs[j - args.idx]['covered_nn'])

        ##################################################
        # stage 2 interval
        interval_2 = tuple((norm.ppf(0.025) / np.sqrt(n_3), -norm.ppf(0.025) / np.sqrt(n_3)) + np.mean(X_3))

        logs[j - args.idx]['interval_2'] = interval_2
        if interval_2[0] <= mu_list[DTL_class.win_idx] <= interval_2[1]:
            logs[j - args.idx]['covered_2'] = 1
        else:
            logs[j - args.idx]['covered_2'] = 0
        logs[j - args.idx]['width_2'] = interval_2[1] - interval_2[0]
        pivot = norm.cdf(-np.sqrt(n_3) * np.mean(X_3))
        logs[j - args.idx]['pvalue_2'] = 2 * min(pivot, 1 - pivot)

        # X_cat = np.concatenate([X[win_idx, :], X_2])
        interval_naive = tuple((norm.ppf(0.025) / np.sqrt(n_all), -norm.ppf(0.025) / np.sqrt(n_all)) + theta_data)
        width_naive = -2 * norm.ppf(0.025) / np.sqrt(n_all)
        logs[j - args.idx]['interval_naive'] = interval_naive
        logs[j - args.idx]['width_naive'] = width_naive
        if interval_naive[0] <= mu_list[DTL_class.win_idx] <= interval_naive[1]:
            logs[j - args.idx]['covered_naive'] = 1
        else:
            logs[j - args.idx]['covered_naive'] = 0
        pivot = norm.cdf(-np.sqrt(n_all) * theta_data)
        logs[j - args.idx]['pvalue_naive'] = 2 * min(pivot, 1 - pivot)

        logs[j - args.idx]['mu_true'] = mu_list[DTL_class.win_idx]

        print(logs[j - args.idx])

        path = open("{}_n_{}_{}_{}_K_{}_{}.pickle".format(args.logname, n_1, n_2, n_3, K, j), 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()


if __name__ == "__main__":
    main()
